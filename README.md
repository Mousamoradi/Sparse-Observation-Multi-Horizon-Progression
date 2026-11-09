# Sparse-Observation Multi-Horizon Glaucoma Progression Prediction

Multimodal deep learning framework for forecasting glaucoma progression at **2, 3, and 4 years** from only **two clinical visits** (T₀, T₁ within a one-year observation window). The pipeline integrates circumpapillary retinal nerve fiber layer (cpRNFL) OCT thickness maps, visual field total-deviation (VF TD) plots, and clinical covariates, and introduces a **Temporally-Consistent Multi-Horizon (TC-MH) loss** that enforces monotone risk across prediction horizons.

<p align="center">
  <img src="figures/fig1_study_design.png" width="900" alt="Study design: sparse-observation multi-horizon framework with TC-MH loss">
</p>

**Figure 1.** Study design. **(A)** A representative fast progressor showing structure–function discordance — near-flat cpRNFL slope (−0.82 μm/year) despite rapid MD decline (−1.90 dB/year) — motivating a multimodal approach. **(B)** Sparse-observation multi-horizon framework: two observed visits at T₀ and T₀+1 year predict progression labels at T₂, T₃, and T₄ (years 2, 3, 4), with a monotonicity constraint P(Y₂) ≤ P(Y₃) ≤ P(Y₄). **(C)** Multimodal architecture: a shared ConvNeXt-V2 image encoder embeds cpRNFL and VF TD at each visit, a tabular MLP encodes clinical covariates, a two-layer Bi-LSTM produces a temporal representation c_final, and a horizon-conditioned shared prediction head outputs three sigmoid probabilities. The **TC-MH loss** combines weighted BCE with a monotonicity penalty: `L = L_BCE + λ · L_mono`.

---

## Method summary

The framework has two contributions over standard multimodal Bi-LSTM baselines:

1. **Horizon-conditioned shared prediction head.** Instead of three independent output heads, a single shared head is conditioned on a learned horizon embedding (Y₂ / Y₃ / Y₄). This ties horizon-specific predictions to a common latent risk representation.

2. **Temporally-Consistent Multi-Horizon (TC-MH) loss.** A monotonicity penalty `L_mono = mean(ReLU(P_h − P_{h+1})²)` is added to the masked BCE loss, discouraging clinically implausible non-monotone risk trajectories (e.g. P(Y₂) > P(Y₃)).

The proposed model (`ConvNeXt-V2 + TC-MH`) is compared against four backbone baselines (ConvNeXt-V2, ViT-Base, MobileNet-V2, EfficientNet-B0) under **identical cohort, splits, and training hyperparameters**.

For **external validation** at sites that export only the 12 clock-hour cpRNFL thicknesses, a conditional generative decoder (Step 4) produces the cpRNFL thickness map from those values; the trained progression model is applied without retraining.

---

## Repository structure

```
.
├── data_preparation.py           # Step 1: match VF/cpRNFL, QC, MD-slope labels
├── sequence_generation.py        # Step 2: build (T0, T1) sparse-observation sequences
├── src/
│   └── Training_code/
│       ├── train_baseline.py     # Step 3a: ConvNeXt / ViT / MobileNet / EfficientNet
│       └── train_tcmh.py         # Step 3b: proposed ConvNeXt-V2 + TC-MH model
├── train_clock2map_generative.py # Step 4: clock-hour → cpRNFL-map decoder (external validation)
├── figures/                      # Study-design and results figures
├── requirements.txt
├── LICENSE
└── README.md
```

`train_baseline.py` and `train_tcmh.py` share the same dataset, cohort filter, patient-level `GroupKFold` cross-validation, tabular preprocessing, and optimizer schedule. The only differences between them are the model class and the loss, so any performance gain from TC-MH is attributable to the method — not to data or training-setup drift.

---

## Requirements

Python ≥ 3.10 with:

```
torch>=2.0
torchvision
timm
pandas
numpy
scikit-learn
Pillow
scipy
matplotlib
tqdm
```

Install:

```bash
pip install -r requirements.txt
```

---

## Input data

The pipeline expects longitudinal glaucoma records with, at minimum:

- `patient_id`, `eye`, `exam_date`
- `md` (mean deviation), `avg_rnfl` (mean cpRNFL thickness)

Optional columns used downstream:

- `false_positive_rate`, `signal_strength` — QC filters
- `age`, `sex`, `race` — tabular covariates
- `vfi`, `vf_progression`, `Severity`, `gl_subtype` — labels and stratification

Column names may need to be adjusted inside the scripts to match your dataset.

The repository does **not** ship patient data; access is subject to institutional review and data-sharing agreements.

---

## Step 1 — Data preparation

`data_preparation.py` matches VF and cpRNFL records by patient, eye, and date, applies QC filters, computes per-eye MD slope, and assigns progression categories.

```bash
python data_preparation.py \
    --vf_path   /path/to/vf_data.csv \
    --rnfl_path /path/to/rnfl_data.csv \
    --output_path prepared_data.csv
```

**Output.** `prepared_data.csv` — matched VF–cpRNFL visits with MD-slope-based progression labels.

---

## Step 2 — Sequence generation

`sequence_generation.py` converts prepared longitudinal records into sparse-observation sequences using two input visits (T₀, T₁) and assigns multi-horizon labels for years 2, 3, and 4. Sequences with partial follow-up are **retained** with missing horizons masked, rather than discarded.

```bash
python sequence_generation.py \
    --input_path  prepared_data.csv \
    --output_path sequences.csv
```

**Output.** `sequences.csv` containing per-sequence T₀/T₁ image paths, tabular covariates, and horizon labels `label_y2`, `label_y3`, `label_y4` (NaN where follow-up is unavailable).

---

## Step 3a — Baseline training

`src/Training_code/train_baseline.py` trains one of four multimodal Bi-LSTM baselines (shared-weight image encoder + tabular concatenation + three independent output heads) with masked multi-horizon BCE.

```bash
python src/Training_code/train_baseline.py \
    --sequences_path sequences.csv \
    --image_dir      /path/to/images \
    --backbone       convnext \
    --output_dir     ./results_baseline
```

**Supported backbones:** `convnext`, `vit`, `mobilenet`, `efficientnet`.

## Step 3b — Proposed TC-MH training

`src/Training_code/train_tcmh.py` trains the proposed ConvNeXt-V2 + TC-MH model with a horizon-conditioned shared head and masked TC-MH loss. All other components are identical to the baseline script.

```bash
python src/Training_code/train_tcmh.py \
    --sequences_path sequences.csv \
    --image_dir      /path/to/images \
    --output_dir     ./results_tcmh \
    --lambda_mono    0.5
```

**Output** (both scripts). Per-fold best checkpoints and a JSON summary of AUROC, accuracy, and F1 at each horizon.

---

## Step 4 — Clock-hour-to-map decoder (external validation)

Some sites export only the 12 clock-hour mean cpRNFL thicknesses (3.46 mm measurement circle), not the 200 × 200 thickness map that the image encoder expects. `train_clock2map_generative.py` trains a conditional generative decoder on development-site scans that maps those 12 values to a 200 × 200 cpRNFL thickness map (μm, 6 × 6 mm field, right-eye orientation). The map is then rendered as a training-style cpRNFL image (thickness-to-image conversion: `convert_array_to_pil`) and passed to the progression model, which is not retrained.

**Design.**

- *Conditioning:* standardized clock hours (MLP embedding), a geometric pseudo-map (the 12 values painted on the 3.46 mm circle) and coordinate channels, supplied at every decoder scale.
- *Generator:* 25 → 50 → 100 → 200 px up-sampling CNN with FiLM conditioning; output = softplus(head + pseudo-map), in μm.
- *Loss:* masked MSE + 0.5 × masked Sobel-gradient L1 + 0.02 × hinge adversarial loss (conditional PatchGAN). Set `ADV_WEIGHT = 0` for a pure regression decoder.
- *Evaluation:* patient-grouped 80/20 split; early stopping (patience 5) on masked RMSE in a 5% patient-grouped validation split of the development part. RMSE and SSIM against a mean-map baseline, and the agreement between the generated map's ring-sector means and the input clock hours, are reported on the held-out 20%. The final decoder is then refit on all scans for the selected number of epochs.

**Input.** Paths are read from environment variables (see the CONFIG block at the top of the script):

- `C2M_META_CSV` (default `$C2M_DATA_ROOT/metadata.csv`) — one row per scan: `clockhour1` … `clockhour12` (μm), `datadir`, `id`, and optionally `righteye` (1 = OD, 0 = OS) and `signalstrength` (scans below 7 are dropped).
- `C2M_SCANS_DIR` (default `$C2M_DATA_ROOT/scans`) — one folder per scan with `segmentation_rnfl_to_gcl.csv` and `segmentation_ilm.csv` (200 × 200 boundary maps in axial pixels; thickness = their difference × 1.955 μm), optionally `cup_coordinates_x.csv` / `cup_coordinates_y.csv`.

```bash
export C2M_DATA_ROOT=/path/to/oct_export     # metadata.csv and scans/
export C2M_OUT_DIR=/path/to/out              # weights, scaler, report, example figure
export C2M_CACHE_DIR=/path/to/cache          # patient-level arrays: keep outside this repository
python train_clock2map_generative.py
```

The map cache needs about 16 GB per 200k scans (float16; `CACHE_DTYPE = np.uint8` halves it), and the build is chunked and resumable. Set `MAX_SCANS = 5000` for a smoke test. `scikit-image` is optional (SSIM is skipped without it).

**Output** (in `C2M_OUT_DIR`).

- `clock2map_generative_all_available.pth` — generator weights
- `decoder_generative_clock2map_all_available.npz` — clock scaler, disc mask, configuration and the weights' SHA-256
- `decoder_generative_report_all_available.json` — held-out metrics and training history
- `generative_examples_heldout.png` — real | clock hours | generated, held-out examples

**Decoding.**

```python
from train_clock2map_generative import GenerativeDecoder

dec = GenerativeDecoder("decoder_generative_clock2map_all_available.npz")   # finds the .pth next to it
m = dec.decode(clocks_um, eye="OD")   # clocks_um: 12 values, hour 1 … hour 12 (μm)
# m is a 200 × 200 map (μm) in right-eye orientation.
# For a left eye pass eye="OS", then mirror the result back: m = m[:, ::-1]
```

**Notes.**

- The generated maps are population-typical syntheses consistent with the 12 clock-hour values. Fine texture added by the adversarial term is invented, so the maps are not reconstructions of an individual scan and should be reported as generated wherever they are used.
- Left eyes are converted to right-eye orientation by mirroring the map and permuting the clock hours (h → 12 − h, `mirror_clocks_os`). This assumes an export whose clock-hour numbering is mirrored between eyes; check your export's numbering (ring-sector means vs clock columns) before training.
- The decoder artifact holds network weights, the clock scaler and a population disc mask, and no individual map; the training cache does, so keep `C2M_CACHE_DIR` and `C2M_OUT_DIR` outside the repository. Differential privacy is not implemented. The repository ships code only, with no scans, maps or trained weights.

---

## Results

<p align="center">
  <img src="figures/fig2_subgroup_analysis.png" width="900" alt="Subgroup analysis across race, sex, and age with prediction uncertainty">
</p>

**Figure 2.** Subgroup performance and prediction uncertainty across demographic strata and severity subgroups (S1, S2, S3). Accuracy and F1 are reported for **(A)** race (Asian, Black, White), **(B)** sex (Female, Male), and **(C)** age (< 60, 60–70, > 70) across five model variants: **ConvNeXt + TC-MH** (proposed), ConvNeXt, ViT, MobileNet, and EfficientNet. The right column shows prediction uncertainty per subgroup. The proposed ConvNeXt + TC-MH model achieves the highest accuracy and F1 in nearly every subgroup–severity combination while maintaining consistently low uncertainty relative to baselines.

---

## Notes on reproducibility

- Patient-level `GroupKFold` cross-validation is used in both scripts to prevent eye-level and patient-level leakage across folds.
- Random seeds for Python, NumPy, and PyTorch are set at the start of each run.
- Missing images are replaced with zero tensors at load time; missing horizon labels are masked out of both the BCE and the monotonicity penalty.
- Tabular features are z-scored per fold using training-set statistics only.

---

## Data availability

The clinical dataset used in this study, including the scans used to train the clock-hour-to-map decoder (Step 4), is not publicly available due to institutional privacy and data-sharing restrictions. Users should prepare their own dataset in the format described in **Input data** above.

---

## License

MIT License. See [`LICENSE`](LICENSE).

---

## Citation

If you use this code, please cite:

> Zebardast, Nazlee, Mousa Moradi, Jerry Cao-Xue, Asahi Fujita, Daniel Liebman, Alessandro Jammal, Mengyu Wang, Tobias Elze, and Mohammad Eslami. **"Sparse-Observation Multi-Horizon Glaucoma Progression Forecasting with Biologically Constrained Temporal Consistency: A Glaucoma Case Study."** (2026).

---

## Contact

Mousa Moradi: `mmoradi2@meei.harvard.edu`
Harvard Ophthalmology AI Lab, Schepens Eye Research Institute / Mass Eye and Ear / Harvard Medical School.
