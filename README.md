# DeSC: Decoupled Sensitivity–Consistency Framework for Weakly Supervised Video Anomaly Detection

## Overview  
DeSC addresses the inherent **sensitivity–stability trade-off** in unified anomaly detection frameworks by decoupling optimization into two specialized streams:  
- **Temporal Sensitivity Stream**: trained to capture transient anomalies.  
- **Semantic Consistency Stream**: trained to preserve long-term semantic coherence.  

Predictions are fused via a **collaborative inference mechanism**, yielding balanced detection of both abrupt and sustained anomalies.

> **State-of-the-art results**:  
> • **UCF-Crime**: 89.37% AUC  
> • **XD-Violence**: 87.18% AP

---

## Model Checkpoints  
Two expert models per dataset are provided (sensitivity + consistency streams):

| Dataset      | Sensitivity Stream  | Consistency Stream  |
|--------------|------------------------------|---------------------------|
| **UCF-Crime**    | [Download (OneDrive)](https://drive.google.com/file/d/11Vn3FE9kS3FK09HK2VOvE_Vxxi65HzgY/view?usp=sharing) | [Download (OneDrive)](https://drive.google.com/file/d/1F7Pa8-Nl0I47jHus_r4BJwS7b2HwLIrx/view?usp=sharing) |
| **XD-Violence**  | [Download (OneDrive)](https://drive.google.com/file/d/1O-_7LscNeYeJlLm-p8YmICq3LV_DELnJ/view?usp=sharing) | [Download (OneDrive)](https://drive.google.com/file/d/1R7X_LQUoqwgbIZAKUuoNEl1dRTk6zZGl/view?usp=sharing) |

---

## Setup  
Before testing, update the following files with your local paths:

### 1. Feature List Files  
Modify the CLIP feature paths in:
- `list/ucf_CLIP_rgb.csv`
- `list/ucf_CLIP_rgbtest.csv`
- `list/xd_CLIP_rgb.csv`
- `list/xd_CLIP_rgbtest.csv`

### 2. Configuration Files  
Set dataset-specific parameters in:
- `src/ucf_option.py`
- `src/xd_option.py`

### 3. Test Scripts  
In the following test scripts, assign your downloaded checkpoint paths to the corresponding variables:

- `ucf_test_tta.py` (for UCF-Crime)
- `xd_test_tta.py` (for XD-Violence)

Specifically, update these lines inside each script:
```python
PATH_SOTA1_TCN_GT = "/path/to/sensitivity_stream.pth"
PATH_SOTA2_GCN_GMP = "/path/to/consistency_stream.pth"
```

## Inference
Run the corresponding test script:
```
python ucf_test_tta.py
```
```
python xd_test_tta.py
```
The output includes AUC/AP.
