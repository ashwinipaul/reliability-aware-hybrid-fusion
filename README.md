# Reliability-Aware Hybrid Fusion Framework

Reference implementation for:

**A Reliability-Aware Hybrid Fusion Framework for Robust Multi-Source Histopathology Image Classification under Domain Heterogeneity and Missing-Source Conditions**

This repository implements the architecture and experimental settings described in the manuscript: source-specific ResNet-50 encoders, adaptive attention-based feature fusion, source-wise classification, entropy-based reliability estimation with temperature scaling, reliability-aware decision fusion, missing-source inference, the joint training objective, five-seed evaluation, patient-wise splitting, and the reported augmentation/training settings.



## 1. Repository structure

```text
reliability-aware-hybrid-fusion/
├── README.md
├── requirements.txt
├── LICENSE
├── configs/
│   ├── camelyon16.yaml
│   ├── camelyon17.yaml
│   └── breakhis.yaml
├── data/
│   └── README.md
├── manifests/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── augmentations.py
│   ├── calibration.py
│   ├── config.py
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── missing_source.py
│   ├── ablation.py
│   ├── statistical.py
│   ├── split.py
│   ├── utils.py
│   ├── wsi_patches.py
│   └── smoke_test.py
└── scripts/
    ├── train_camelyon16.sh
    ├── train_camelyon17.sh
    └── train_breakhis.sh
```

## 2. Environment

The manuscript reports PyTorch 2.1, ResNet-50 ImageNet initialization, AdamW, batch size 32, 100 epochs, early stopping patience 15, and cosine annealing.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use a Python version supported by the selected PyTorch 2.1 wheels (Python 3.10/3.11 is recommended for this repository).

For WSI handling, OpenSlide system libraries may also be required by your operating system.

## 3. Data organization

The model expects a **paired multi-source manifest**. Each row represents one multi-source sample and contains the label plus one image path for each source.

Example:

```csv
sample_id,label,patient_id,source_0,source_1,source_2,source_3
p001,0,p001,data/source0/p001.png,data/source1/p001.png,data/source2/p001.png,data/source3/p001.png
p002,1,p002,data/source0/p002.png,,data/source2/p002.png,data/source3/p002.png
```

Blank source fields are interpreted as missing sources.

This manifest abstraction is intentional. The manuscript defines sources as heterogeneous institutions/devices/staining/magnification levels, but it does not provide the exact sample-pairing construction. The manifest therefore makes that experimental choice explicit instead of silently inventing one.

For BreakHis, a natural source definition is magnification (40x/100x/200x/400x) **only when the same sample can be validly paired across sources**. Do not fabricate pairs where the dataset does not contain corresponding observations.

For Camelyon16/Camelyon17, use the actual source/center assignment and patch-level pairing used in your experiment.

## 4. Download data

Obtain datasets directly from their official sources:

- CAMELYON data: https://camelyon17.grand-challenge.org/Data/
- CAMELYON16 data: https://camelyon16.grand-challenge.org/Data/
- BreaKHis: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

Follow the current license/usage terms of each dataset. Do not place dataset images in this repository.

## 5. Configuration

The three YAML files mirror the manuscript's stated settings:

- ResNet-50 ImageNet pretrained
- embedding dimension 1024
- two-layer attention fusion
- batch size 32
- AdamW
- 100 epochs
- early stopping patience 15
- cosine annealing
- five independent seeds
- 224x224 patches
- augmentation: random crop, horizontal flip, rotation, stain augmentation, color jitter, Gaussian noise

The manuscript text does not expose numerical values for the learning rate, weight decay, λ1, λ2, or temperature. The YAML files therefore expose these values explicitly so they can be replaced with the validated experimental values. The runnable defaults are placeholders and are **not claims about the unpublished historical values**.

## 6. Prepare splits

The code performs patient-wise splitting. A 70/10/20 split is used for CAMELYON datasets as stated in the manuscript.

```bash
python -m src.split \
  --manifest manifests/camelyon16.csv \
  --output manifests/camelyon16_split.csv \
  --train-ratio 0.70 --val-ratio 0.10 --seed 42
```

The output contains a `split` column with `train`, `val`, or `test`.

## 7. Train

```bash
python -m src.train --config configs/camelyon16.yaml --seed 42
python -m src.train --config configs/camelyon17.yaml --seed 42
python -m src.train --config configs/breakhis.yaml --seed 42
```

The trainer saves:

```text
outputs/<dataset>/
├── best.pt
├── history.csv
├── config.yaml
└── test_metrics.json
```

## 8. Evaluate

```bash
python -m src.evaluate \
  --config configs/camelyon16.yaml \
  --checkpoint outputs/camelyon16/best.pt
```

Metrics include accuracy, precision, recall, F1, and ROC-AUC where ROC-AUC is defined for the target labels.

## 9. Missing-source evaluation

The framework dynamically renormalizes feature-attention and reliability weights over sources that are actually available. No imputation or retraining is performed.

```bash
python -m src.missing_source \
  --config configs/camelyon17.yaml \
  --checkpoint outputs/camelyon17/best.pt
```

For a four- or five-source experiment, the evaluator tests:

- no source missing
- every one-source-missing combination
- every two-source-missing combination

For fewer sources, only valid combinations are evaluated.

## 10. Ablation

The implementation supports:

1. feature-level fusion only
2. decision-level fusion only
3. hybrid fusion without reliability modeling
4. full proposed framework

```bash
python -m src.ablation \
  --config configs/breakhis.yaml \
  --checkpoint outputs/breakhis/best.pt
```

The ablation runner evaluates a trained full-model checkpoint under these aggregation modes; it does not invent historical checkpoint results.

## 11. WSI patch extraction

`src/wsi_patches.py` provides 224x224 non-overlapping patch extraction from OpenSlide-readable WSIs.

```bash
python -m src.wsi_patches \
  --wsi path/to/slide.tif \
  --output-dir data/patches/slide_001 \
  --patch-size 224 \
  --level 0 \
  --tissue-threshold 0.50
```

The manuscript specifies patch size and non-overlapping sampling, but not the exact tissue threshold, stain algorithm, or XML-to-patch label rule. Those are therefore configurable rather than silently invented.

## 12. Five independent runs

The manuscript reports five repetitions with different random seeds and mean ± standard deviation.

```bash
for seed in 42 52 62 72 82; do
  python -m src.train --config configs/camelyon17.yaml --seed $seed
done
```

## 13. Method implementation

### Feature-level fusion

For source features `f_s`, a two-layer MLP produces attention scores. Softmax normalization gives:

`alpha_s = softmax(score_s)`

and:

`f_fused = sum_s alpha_s f_s`

### Reliability-aware decision fusion

For calibrated source probabilities `p_s`, predictive entropy is:

`H_s = -sum_c p_sc log(p_sc)`

Reliability is implemented as:

`r_s = 1 / (H_s + epsilon)`

and reliability weights are softmax-normalized before weighted probability aggregation.

Temperature scaling is applied before entropy computation.

### Missing sources

Only available sources contribute to both fusion stages. Weights are renormalized over the available set.

### Training loss

The implementation follows:

`L = L_fused + lambda1 * L_source + lambda2 * L_reg`

where `L_source` is the mean source-wise cross-entropy and `L_reg` regularizes adaptive attention weights.

## 14. Important reproducibility limitations

Before submitting a code-availability artifact as “reproducible code for the paper,” record:

- exact dataset release/version
- exact source definitions
- exact sample pairing
- exact patient IDs in each split
- exact WSI level and patch-labeling policy
- exact stain augmentation parameters
- calibrated temperature per source/dataset
- exact learning rate and weight decay
- exact λ1 and λ2
- exact baseline implementations and checkpoints
- exact five random seeds
- exact hardware/software versions

The manuscript also describes both a 70/10/20 patient-wise split and five-fold cross-validation. These are not fully reconciled in the text. The supplied code defaults to the explicitly stated 70/10/20 holdout plus repeated seeds; if the actual experiment used five folds, preserve those fold assignments in the final release instead.

## 15. Smoke test

A synthetic smoke test checks the data loader and complete model forward pass without downloading medical datasets:

```bash
python -m src.smoke_test
```

## 16. Citation

If you release this implementation, cite the associated manuscript and the original CAMELYON/BreaKHis dataset publications as appropriate.
