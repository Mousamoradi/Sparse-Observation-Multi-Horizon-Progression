# Multi-Horizon-Glaucoma-Progression-Prediction-using-limited-input-visits

This repository contains code for sparse-observation, multimodal deep learning for longitudinal prediction of glaucoma progression from only two clinical visits. The pipeline integrates circumpapillary retinal nerve fiber layer (cpRNFL) OCT data, visual field (VF) data, and clinical covariates to generate prediction sequences and train multi-horizon models for forecasting progression at 2, 3, and 4 years. The repository currently includes three main scripts for data preparation, sequence generation, and model training. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

## Overview

The workflow is organized into three stages:

1. **Data preparation**
   - Matches VF and cpRNFL records by patient, eye, and exam date
   - Applies quality-control filters
   - Computes mean deviation (MD) slope
   - Assigns progression categories for downstream analysis :contentReference[oaicite:3]{index=3}

2. **Sequence generation**
   - Creates sparse-observation two-visit sequences `(T0, T1)`
   - Assigns horizon-specific labels for years 2, 3, and 4
   - Masks missing horizons instead of discarding partially observed sequences :contentReference[oaicite:4]{index=4}

3. **Model training**
   - Trains a multimodal Bi-LSTM framework with shared-weight image encoders
   - Supports ConvNeXt, ViT, MobileNet, and EfficientNet backbones
   - Uses masked multi-horizon binary cross-entropy loss
   - Performs patient-level 5-fold cross-validation to prevent leakage :contentReference[oaicite:5]{index=5}

## Repository Structure

```text
.
├── data_preparation.py
├── sequence_generation.py
├── model_training.py
└── README.md
```
## Requirements

Recommended environment:

Python 3.10+
PyTorch
torchvision
pandas
numpy
scikit-learn
Pillow

Depending on your environment, you may also want:

scipy
statsmodels
matplotlib
jupyter

Example installation:
pip install torch torchvision pandas numpy scikit-learn pillow scipy statsmodels matplotlib
Input Data

The scripts expect structured CSV files containing longitudinal glaucoma data. At minimum, your data should include:

patient_id
eye
exam_date
md
avg_rnfl

Additional columns used by the pipeline may include:

false_positive_rate
signal_strength
age
sex
race
vfi
vf_progression
Severity
gl_subtype

You may need to adjust column names in the scripts to match your local dataset.

** Step 1: Data Preparation

data_preparation.py loads VF and cpRNFL CSV files, matches records within a date window, applies quality filters, computes MD slope, and assigns progression categories.

Example
python data_preparation.py \
  --vf_path /path/to/vf_data.csv \
  --rnfl_path /path/to/rnfl_data.csv \
  --output_path prepared_data.csv
Output

A prepared CSV file containing matched VF-cpRNFL records, MD slope estimates, and progression categories.

Step 2: Sequence Generation

sequence_generation.py converts prepared longitudinal data into sparse-observation sequences using two input visits (T0, T1) and assigns multi-horizon labels for years 2, 3, and 4. Missing horizon labels are masked rather than excluded.

Example
python sequence_generation.py \
  --input_path prepared_data.csv \
  --output_path sequences.csv
Output

A sequences.csv file containing:

T0/T1 dates
baseline and follow-up covariates
subtype and severity metadata
horizon labels label_y2, label_y3, and label_y4
Step 3: Model Training

model_training.py trains the multimodal Bi-LSTM model using the generated sequences and image files. It supports multiple pretrained image backbones and performs patient-level cross-validation.

Example
python model_training.py \
  --sequences_path sequences.csv \
  --image_dir /path/to/images \
  --backbone convnext \
  --output_dir ./results
Supported backbones
convnext
vit
mobilenet
efficientnet
Output

The training script saves:

best model checkpoints for each fold
cross-validation metrics in JSON format
Notes
The model expects paired cpRNFL and VF images at both T0 and T1.
Missing image files are currently replaced with zero tensors during loading.
Patient-level grouped cross-validation is used to reduce data leakage across folds.
Horizon labels are masked when follow-up is unavailable within the tolerance window.
Reproducibility

The training script includes random seed initialization for Python, NumPy, and PyTorch to improve reproducibility.

## Data Availability

This repository does not include patient data.

Because the study uses clinical ophthalmic data, raw data may be subject to institutional review board, privacy, and data-sharing restrictions. Users should prepare their own dataset in a compatible format.

## License

This project is licensed under the MIT License.


## Citation

If you use this code, please cite:

Moradi, M., Cao-Xue, J., Eslami, M., Wang, M., Elze, T. and Zebardast, N., 2025. Multimodal Deep Learning for Longitudinal Prediction of Glaucoma Progression Using Sequential RNFL, Visual Field, and Clinical Data. medRxiv, pp.2025-10.

Contact

mmoradi2@meei.harvard.edu
