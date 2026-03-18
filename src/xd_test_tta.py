import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import xd_option
from utils.tools import get_batch_mask, get_prompt_text

class SmartVideoDataset(Dataset):
    def __init__(self, list_path, target_dim=512):
        self.df = pd.read_csv(list_path)
        self.target_dim = target_dim

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = self.df.loc[index]['path']
        try:
            feature = np.load(path)
            if feature.ndim == 2:
                shape = feature.shape
                if shape[1] == self.target_dim:
                    pass
                elif shape[0] == self.target_dim:
                    feature = feature.transpose(1, 0)
                elif shape[0] in [256, 1024, 2048] and shape[1] not in [256, 1024, 2048]:
                    feature = feature.transpose(1, 0)
            feature = torch.from_numpy(feature).float()
            label = 0
            return feature, label, feature.shape[0]
        except Exception as e:
            return torch.zeros(1, self.target_dim), 0, 0

def collate_fn_safe(batch):
    batch = [b for b in batch if b[2] > 0]
    if not batch: return torch.empty(0), 0, 0
    return batch[0]

try:
    from model_modular_corrected import CLIPVAD_Modular_Parallel as CLIPVAD_SOTA1
    from model_multigmp import CLIPVAD as CLIPVAD_SOTA2
except ImportError:
    exit()

def test_ensemble_TTA_Smart(model_sota1, model_sota2, testdataloader, maxlen, tta_stride, prompt_text, gt_labels, device):
    model_sota1.to(device).eval()
    model_sota2.to(device).eval()

    list_s1_ap1, list_s1_ap2 = [], []
    list_s2_ap1, list_s2_ap2 = [], []
    list_ens_ap1, list_ens_ap2 = [], []


    with torch.no_grad():
        for i, (visual, label, length) in enumerate(testdataloader):
            if length == 0: continue

            visual = visual.to(device)  # (T, D)
            T, D = visual.shape

            if D != 512:
                if T == 512:
                    visual = visual.transpose(1, 0)
                    T, D = visual.shape
                if D != 512 and i == 0:
                    print(f"Video 0 dimension is {D}, model expects 512.")

            visual_t = visual.permute(1, 0).unsqueeze(0)
            visual_resized = F.interpolate(visual_t, size=maxlen, mode='linear', align_corners=True)
            padded = visual_resized.permute(0, 2, 1)
            lens = torch.tensor([maxlen], device=device, dtype=torch.int)
            mask = get_batch_mask(lens, maxlen)
            _, l1_s1, l2_s1, _, _ = model_sota1(padded, mask, prompt_text, lens)
            s1_ap1_256 = torch.sigmoid(l1_s1.squeeze(0).squeeze(-1))
            s1_ap2_256 = 1.0 - F.softmax(l2_s1, dim=-1)[0, :, 0]
            _, l1_s2, l2_s2, _, _ = model_sota2(padded, mask, prompt_text, lens)
            s2_ap1_256 = torch.sigmoid(l1_s2.squeeze(0).squeeze(-1))
            s2_ap2_256 = 1.0 - F.softmax(l2_s2, dim=-1)[0, :, 0]
            def resize_back(score_256, target_T):
                score_256 = score_256.view(1, 1, -1)
                score_T = F.interpolate(score_256, size=target_T, mode='linear', align_corners=True)
                return score_T.view(-1).cpu().numpy()
            v_s1_ap1 = resize_back(s1_ap1_256, T)
            v_s1_ap2 = resize_back(s1_ap2_256, T)
            v_s2_ap1 = resize_back(s2_ap1_256, T)
            v_s2_ap2 = resize_back(s2_ap2_256, T)
            list_s1_ap1.append(v_s1_ap1)
            list_s1_ap2.append(v_s1_ap2)
            list_s2_ap1.append(v_s2_ap1)
            list_s2_ap2.append(v_s2_ap2)
            list_ens_ap1.append((v_s1_ap1 + v_s2_ap1) / 2.0)
            list_ens_ap2.append((v_s1_ap2 + v_s2_ap2) / 2.0)

    def calc_ap(preds, gt):
        s = np.concatenate(preds, axis=0)
        f = np.repeat(s, 16)
        m = min(len(f), len(gt))
        return average_precision_score(gt[:m], f[:m])
    print(f"AP (TTA Ensemble):  {calc_ap(list_ens_ap2, gt_labels):.6f}")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = xd_option.parser
    parser.add_argument('--tta_stride', type=int, default=256)
    args = parser.parse_args()

    PATH_SOTA1_TCN_GT = "../xd_sensitivity_stream.pth"
    PATH_SOTA2_GCN_GMP = "../xd_consistency_stream.pth"


    model_sota1 = CLIPVAD_SOTA1(
        num_class=args.classes_num, embed_dim=args.embed_dim, visual_length=args.visual_length,
        visual_width=args.visual_width, prompt_prefix=args.prompt_prefix, prompt_postfix=args.prompt_postfix,
        device=device, tcn_levels=args.tcn_levels, tcn_kernel_size=args.tcn_kernel_size,
        gt_layers=args.graph_layers, gt_heads=args.graph_head
    )

    model_sota2 = CLIPVAD_SOTA2(
        num_class=args.classes_num, embed_dim=args.embed_dim, visual_length=args.visual_length,
        visual_width=args.visual_width, visual_head=args.visual_head, visual_layers=args.visual_layers,
        attn_window=args.attn_window, prompt_prefix=args.prompt_prefix, prompt_postfix=args.prompt_postfix,
        device=device, n_components=5
    )

    try:
        model_sota1.load_state_dict(torch.load(PATH_SOTA1_TCN_GT, map_location=device)['model_state_dict'], strict=False)
        model_sota2.load_state_dict(torch.load(PATH_SOTA2_GCN_GMP, map_location=device)['model_state_dict'], strict=False)
    except Exception as e:
        model_sota1.load_state_dict(torch.load(PATH_SOTA1_TCN_GT, map_location=device), strict=False)
        model_sota2.load_state_dict(torch.load(PATH_SOTA2_GCN_GMP, map_location=device), strict=False)


    label_map = {'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'}
    prompt_text = get_prompt_text(label_map)
    gt_labels = np.load(args.gt_path)

    test_dataset = SmartVideoDataset(args.test_list, target_dim=args.visual_width)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_safe)

    test_ensemble_TTA_Smart(
        model_sota1, model_sota2, test_loader, args.visual_length, args.tta_stride,
        prompt_text, gt_labels, device
    )