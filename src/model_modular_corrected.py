from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
import numpy as np

from utils.layers import DistanceAdj



class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=self.padding, dilation=dilation)
        self.act1 = QuickGELU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=self.padding, dilation=dilation)
        self.act2 = QuickGELU()
        self.drop2 = nn.Dropout(dropout)
        self.norm = LayerNorm(channels)

    def forward(self, x):
        residual = x
        x = x.permute(0, 2, 1)
        x = self.conv1(x)[:, :, :residual.shape[1]]
        x = x.permute(0, 2, 1)
        x = self.act1(x)
        x = self.drop1(x)
        x = x.permute(0, 2, 1)
        x = self.conv2(x)[:, :, :residual.shape[1]]
        x = x.permute(0, 2, 1)
        x = self.act2(x)
        x = self.drop2(x)
        x = x + residual
        x = self.norm(x)
        return x


class AcausalTCN(nn.Module):
    def __init__(self, channels, num_levels, kernel_size, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_levels):
            dilation = 2 ** i
            self.layers.append(ResidualTCNBlock(channels, kernel_size=kernel_size, dilation=dilation, dropout=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GraphResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
            ("dropout", nn.Dropout(dropout))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_bias: torch.Tensor):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_bias)[0]

    def forward(self, x_tuple):
        x, attn_bias = x_tuple
        x = x + self.attention(self.ln_1(x), attn_bias)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_bias)


class OptimalGraphTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.resblocks = nn.Sequential(*[GraphResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x, attn_bias):
        x = x.permute(1, 0, 2)
        x, _ = self.resblocks((x, attn_bias))
        x = x.permute(1, 0, 2)
        return x


class CLIPVAD_Modular_Parallel(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device,
                 tcn_levels: int,
                 tcn_kernel_size: int,
                 gt_layers: int,
                 gt_heads: int):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device
        self.tcn_module = AcausalTCN(visual_width, num_levels=tcn_levels, kernel_size=tcn_kernel_size)
        self.gt_module = OptimalGraphTransformer(width=visual_width, layers=gt_layers, heads=gt_heads)
        self.disAdj = DistanceAdj()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.fusion_mlp = nn.Sequential(
            nn.Linear(visual_width * 2, visual_width),
            QuickGELU(),
            LayerNorm(visual_width)
        )
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)
        self.clipmodel, _ = clip.load("ViT-B/16", device, jit=False)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False
        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)
        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1))
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2 / (x_norm_x + 1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            seq_len = [x.shape[1]] * x.shape[0]
        if isinstance(seq_len, torch.Tensor):
            seq_len_list = seq_len.cpu().numpy()
        else:
            seq_len_list = seq_len

        for i in range(len(seq_len_list)):
            valid_len = int(seq_len_list[i])
            if valid_len > 0:
                tmp = x2[i, :valid_len, :valid_len]
                adj2 = F.threshold(tmp, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :valid_len, :valid_len] = adj2
        return output

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        current_device = images.device
        position_ids = torch.arange(self.visual_length, device=current_device).unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        x = images + frame_position_embeddings

        x_temporal = self.tcn_module(x)

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1]).to(current_device)
        epsilon = 1e-8
        adj_bias = torch.log(adj + epsilon)
        disadj_bias = torch.log(disadj + epsilon)
        attention_bias = self.alpha * adj_bias + self.beta * disadj_bias
        num_heads = self.gt_module.resblocks[0].attn.num_heads
        seq_len = attention_bias.shape[1]
        attention_bias = attention_bias.unsqueeze(1).repeat(1, num_heads, 1, 1).view(-1, seq_len, seq_len)
        x_graph = self.gt_module(x, attention_bias)

        x_fused = torch.cat([x_temporal, x_graph], dim=-1)
        x_final = self.fusion_mlp(x_fused)

        return x_final

    def encode_textprompt(self, text):
        current_device = self.text_prompt_embeddings.weight.device
        word_tokens = clip.tokenize(text).to(current_device)
        word_embedding = self.clipmodel.token_embedding(word_tokens).type(self.clipmodel.dtype)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(current_device)).unsqueeze(0).repeat(
            [len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77, dtype=torch.long).to(current_device)
        for i in range(len(text)):
            actual_len = torch.count_nonzero(word_tokens[i])
            if actual_len >= 77 - self.prompt_prefix - 1:
                actual_len = 77 - self.prompt_prefix - 1
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + actual_len] = word_embedding[i,
                                                                                          1:actual_len]
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + actual_len] = word_embedding[i, actual_len - 1]
            text_tokens[i, self.prompt_prefix + actual_len] = word_tokens[i, actual_len - 1]
        x = text_embeddings + self.clipmodel.positional_embedding.type(self.clipmodel.dtype)
        x = x.permute(1, 0, 2)
        x = self.clipmodel.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clipmodel.ln_final(x)
        text_features = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.clipmodel.text_projection
        return text_features

    def forward(self, visual, padding_mask, text, lengths):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        text_features_ori = self.encode_textprompt(text)
        text_features = text_features_ori
        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / (visual_attn.norm(dim=-1, keepdim=True) + 1e-9)
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
        text_features = text_features_ori.unsqueeze(0)
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = visual_features / (visual_features.norm(dim=-1, keepdim=True) + 1e-9)
        text_features_norm = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-9)
        text_features_norm = text_features_norm.permute(0, 2, 1)
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07
        _ = 0
        return text_features_ori, logits1, logits2, _, _