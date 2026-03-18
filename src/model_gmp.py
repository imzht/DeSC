from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIPVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)
        self.gmp_head = nn.Linear(visual_width, self.num_class * 2)

        self.clipmodel, _ = clip.load("ViT-B/16", device, jit=False)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0
        return mask

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
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :valid_len, :valid_len] = adj2
        return output

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        current_device = images.device
        position_ids = torch.arange(self.visual_length, device=current_device).unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1]).to(current_device)
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)

        return x

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
            if actual_len <= 1: actual_len = 2

            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + actual_len - 1] = word_embedding[i,
                                                                                              1:actual_len - 1]
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + actual_len - 1] = word_embedding[i, actual_len - 1]
            text_tokens[i, self.prompt_prefix + actual_len - 1] = word_tokens[i, actual_len - 1]

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
        gmp_params = self.gmp_head(visual_features)
        pred_mu = torch.sigmoid(gmp_params[..., :self.num_class])
        pred_sigma = F.softplus(gmp_params[..., self.num_class:]) + 1e-6
        return text_features_ori, logits1, logits2, pred_mu, pred_sigma