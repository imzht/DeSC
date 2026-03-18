import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import argparse
from collections import OrderedDict
from clip import clip

from model_modular_corrected import CLIPVAD_Modular_Parallel as CLIPVAD_SOTA1_TCN_GT
from model_gmp import CLIPVAD as CLIPVAD_SOTA2_GCN_GMP
from TTA_dataset import FullVideoDataset, collate_fn_filter_none
from ucf_option import parser as option_parser
from utils.tools import get_batch_mask, get_prompt_text


def test_ensemble_TTA(model_sota1, model_sota2, testdataloader, maxlen, tta_stride, prompt_text, gt, device):

    model_sota1.to(device)
    model_sota1.eval()
    model_sota2.to(device)
    model_sota2.eval()
    list_prob_sota1_final = []
    list_prob_sota2_final = []
    list_prob_ensemble_final = []

    with torch.no_grad():
        for i, (visual, label, length) in enumerate(testdataloader):
            visual = visual.to(device)
            T = length
            if T == 0:
                continue
            final_scores_sota1 = torch.zeros(T, device=device)
            final_scores_sota2 = torch.zeros(T, device=device)
            count_matrix = torch.zeros(T, device=device)
            if T <= maxlen:
                start_indices = [0]
            else:
                start_indices = list(range(0, T - maxlen, tta_stride))
                if (T - maxlen) not in start_indices:
                    start_indices.append(T - maxlen)
            for start_idx in start_indices:
                end_idx = start_idx + maxlen
                chunk_len = min(T, maxlen) if T <= maxlen else maxlen
                if T <= maxlen:
                    end_idx = T
                chunk_visual = visual[start_idx: end_idx]
                padded_chunk = F.pad(chunk_visual, (0, 0, 0, maxlen - chunk_len)).unsqueeze(0)
                lengths_tensor = torch.tensor([chunk_len], device=device, dtype=torch.int)
                padding_mask = get_batch_mask(lengths_tensor, maxlen)
                _, logits1_sota1, _, _, _ = model_sota1(padded_chunk, padding_mask, prompt_text, lengths_tensor)
                prob_sota1 = torch.sigmoid(logits1_sota1.squeeze(0).squeeze(-1))[:chunk_len]
                _, logits1_sota2, _, _, _ = model_sota2(padded_chunk, padding_mask, prompt_text, lengths_tensor)
                prob_sota2 = torch.sigmoid(logits1_sota2.squeeze(0).squeeze(-1))[:chunk_len]
                final_scores_sota1[start_idx: end_idx] += prob_sota1
                final_scores_sota2[start_idx: end_idx] += prob_sota2
                count_matrix[start_idx: end_idx] += 1.0
            count_matrix[count_matrix == 0] = 1.0
            prob_sota1_final = (final_scores_sota1 / count_matrix).cpu().numpy()
            prob_sota2_final = (final_scores_sota2 / count_matrix).cpu().numpy()
            prob_ensemble_final = (prob_sota1_final + prob_sota2_final) / 2.0
            list_prob_sota1_final.append(prob_sota1_final)
            list_prob_sota2_final.append(prob_sota2_final)
            list_prob_ensemble_final.append(prob_ensemble_final)
    prob_sota1_np = np.concatenate(list_prob_sota1_final, axis=0)
    prob_sota2_np = np.concatenate(list_prob_sota2_final, axis=0)
    prob_ensemble_np = np.concatenate(list_prob_ensemble_final, axis=0)
    gt_frames = gt
    gt_len = len(gt_frames)
    prob_sota1_frames = np.repeat(prob_sota1_np, 16)[:gt_len]
    prob_sota2_frames = np.repeat(prob_sota2_np, 16)[:gt_len]
    prob_ensemble_frames = np.repeat(prob_ensemble_np, 16)[:gt_len]
    ROC_SOTA1 = roc_auc_score(gt_frames, prob_sota1_frames)
    ROC_SOTA2 = roc_auc_score(gt_frames, prob_sota2_frames)
    ROC_Ensemble = roc_auc_score(gt_frames, prob_ensemble_frames)
    print(f"AUC (TTA Ensemble):  {ROC_Ensemble:.6f}")

    return ROC_Ensemble, 0.0


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    option_parser.add_argument('--tta_stride', type=int, default=256, help='Stride for TTA sliding window.')
    args = option_parser.parse_args()
    PATH_SOTA1_TCN_GT = "../ucf_sensitivity_stream.pth"
    PATH_SOTA2_GCN_GMP = "../ucf_consistency_stream.pth"
    model_sota1 = CLIPVAD_SOTA1_TCN_GT(
        num_class=args.classes_num,
        embed_dim=args.embed_dim,
        visual_length=args.visual_length,
        visual_width=args.visual_width,
        prompt_prefix=args.prompt_prefix,
        prompt_postfix=args.prompt_postfix,
        device=device,
        tcn_levels=args.tcn_levels,
        tcn_kernel_size=args.tcn_kernel_size,
        gt_layers=args.graph_layers,
        gt_heads=args.graph_head
    )
    model_sota2 = CLIPVAD_SOTA2_GCN_GMP(
        num_class=args.classes_num,
        embed_dim=args.embed_dim,
        visual_length=args.visual_length,
        visual_width=args.visual_width,
        visual_head=args.visual_head,
        visual_layers=args.visual_layers,
        attn_window=args.attn_window,
        prompt_prefix=args.prompt_prefix,
        prompt_postfix=args.prompt_postfix,
        device=device
    )

    try:
        ckpt_sota1 = torch.load(PATH_SOTA1_TCN_GT, map_location=device)
        if 'model_state_dict' in ckpt_sota1:
            model_sota1.load_state_dict(ckpt_sota1['model_state_dict'], strict=False)
        else:
            model_sota1.load_state_dict(ckpt_sota1, strict=False)

        ckpt_sota2 = torch.load(PATH_SOTA2_GCN_GMP, map_location=device)
        if 'model_state_dict' in ckpt_sota2:
            model_sota2.load_state_dict(ckpt_sota2['model_state_dict'], strict=False)
        else:
            model_sota2.load_state_dict(ckpt_sota2, strict=False)
    except Exception as e:
        exit()
    label_map = dict({'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson', 'Assault': 'Assault',
                      'Burglary': 'Burglary', 'Explosion': 'Explosion', 'Fighting': 'Fighting',
                      'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery', 'Shooting': 'Shooting',
                      'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing', 'Vandalism': 'Vandalism'})

    testdataset = FullVideoDataset(list_path=args.test_list, label_map=label_map)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False, collate_fn=collate_fn_filter_none)
    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    test_ensemble_TTA(
        model_sota1,
        model_sota2,
        testdataloader,
        args.visual_length,
        args.tta_stride,
        prompt_text,
        gt,
        device
    )