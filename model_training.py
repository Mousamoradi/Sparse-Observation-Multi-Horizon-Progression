"""
model_training.py
==================
Training pipeline for multi-horizon glaucoma progression prediction using a
multimodal Bi-LSTM framework with four backbone architectures.

Architecture:
    - Shared-weight image encoders (ConvNeXt-V2 / ViT-base / MobileNet-V2 /
      EfficientNet-B0) encode cpRNFL and VF TD images at T0 and T1.
    - Per-visit embeddings are concatenated with tabular covariates and passed
      to a two-layer Bi-LSTM (256 hidden units, dropout 0.3).
    - A three-output sigmoid head predicts progression at years 2, 3, and 4.
    - Masked multi-horizon BCE loss ignores unavailable horizon labels.
    - Patient-level five-fold cross-validation prevents data leakage.

Reference:
    Moradi et al., "Multi-Horizon Glaucoma Progression Prediction from Minimal
    Longitudinal Data: A Reliability-Aware Multimodal Deep Learning Framework"
    IEEE TBME, 2025.

Usage:
    python model_training.py --sequences_path sequences.csv \
                              --image_dir /path/to/images \
                              --backbone convnext \
                              --output_dir ./results
"""

import argparse
import random
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from PIL import Image


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

HORIZONS       = [2, 3, 4]
IMAGE_SIZE     = 224
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]
TABULAR_FEATS  = ["t0_md", "t1_md", "t0_rnfl", "t1_rnfl", "t0_age"]
CAT_FEATS      = ["t0_sex", "t0_race"]

BACKBONE_DIMS  = {
    "convnext":    1024,
    "vit":         768,
    "mobilenet":   1280,
    "efficientnet": 1024,
}

# ─────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────

class GlaucomaSequenceDataset(Dataset):
    """
    PyTorch Dataset for paired (T0, T1) multimodal glaucoma sequences.

    Each sample provides:
      - cpRNFL image at T0 and T1  (3 × 224 × 224)
      - VF TD image at T0 and T1   (3 × 224 × 224)
      - Tabular covariates          (float tensor)
      - Label vector [y2, y3, y4]  (float tensor, NaN for masked horizons)
    """

    def __init__(self,
                 df: pd.DataFrame,
                 image_dir: str,
                 scaler: StandardScaler = None,
                 fit_scaler: bool = False,
                 augment: bool = False):
        self.df        = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.augment   = augment

        # Image transforms
        aug_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ] if augment else []

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            *aug_transforms,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Tabular preprocessing
        tab_cols = TABULAR_FEATS + [f"{c}_{v}" for c in CAT_FEATS
                                    for v in df[c].dropna().unique()]
        df_encoded = pd.get_dummies(df, columns=CAT_FEATS, drop_first=False)
        self.tab_cols = [c for c in df_encoded.columns if c in tab_cols or
                         any(c.startswith(f"{cat}_") for cat in CAT_FEATS)]
        tab_data = df_encoded[self.tab_cols].fillna(0).values.astype(np.float32)

        if fit_scaler:
            self.scaler = StandardScaler()
            self.tab_data = self.scaler.fit_transform(tab_data)
        elif scaler is not None:
            self.scaler   = scaler
            self.tab_data = scaler.transform(tab_data)
        else:
            self.scaler   = None
            self.tab_data = tab_data

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, filepath: str) -> torch.Tensor:
        path = self.image_dir / filepath
        if not path.exists():
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        rnfl_t0 = self._load_image(str(row.get("rnfl_img_t0", "")))
        rnfl_t1 = self._load_image(str(row.get("rnfl_img_t1", "")))
        vf_t0   = self._load_image(str(row.get("vf_img_t0",   "")))
        vf_t1   = self._load_image(str(row.get("vf_img_t1",   "")))

        tab = torch.tensor(self.tab_data[idx], dtype=torch.float32)

        labels = torch.tensor(
            [row.get(f"label_y{h}", float("nan")) for h in HORIZONS],
            dtype=torch.float32
        )

        return {
            "rnfl_t0": rnfl_t0, "rnfl_t1": rnfl_t1,
            "vf_t0":   vf_t0,   "vf_t1":   vf_t1,
            "tabular": tab,
            "labels":  labels,
        }


# ─────────────────────────────────────────────
# 2. Backbone factory
# ─────────────────────────────────────────────

def build_backbone(name: str, pretrained: bool = True) -> nn.Module:
    """Return a pretrained image encoder with the classification head removed."""
    name = name.lower()
    if name == "convnext":
        model = models.convnext_base(pretrained=pretrained)
        model.classifier = nn.Identity()
    elif name == "vit":
        model = models.vit_b_16(pretrained=pretrained)
        model.heads = nn.Identity()
    elif name == "mobilenet":
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier = nn.Identity()
    elif name == "efficientnet":
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: '{name}'. "
                         f"Choose from: {list(BACKBONE_DIMS.keys())}")
    return model


# ─────────────────────────────────────────────
# 3. Model
# ─────────────────────────────────────────────

class MultimodalBiLSTM(nn.Module):
    """
    Multimodal Bi-LSTM for multi-horizon glaucoma progression prediction.

    For each visit (T0, T1):
      1. Encode cpRNFL and VF images via shared-weight backbone.
      2. Concatenate image embeddings with tabular covariates.
    Pass the two-visit sequence to a Bi-LSTM and decode to three sigmoid outputs.
    """

    def __init__(self,
                 backbone_name:  str,
                 tab_dim:        int,
                 lstm_hidden:    int = 256,
                 lstm_layers:    int = 2,
                 dropout:        float = 0.3,
                 pretrained:     bool  = True):
        super().__init__()

        img_dim  = BACKBONE_DIMS[backbone_name.lower()]
        self.encoder = build_backbone(backbone_name, pretrained=pretrained)

        # Per-visit feature dimension: 2 images × img_dim + tabular
        visit_dim = 2 * img_dim + tab_dim

        self.lstm = nn.LSTM(
            input_size=visit_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        # Prediction head: 512 → 128 → 3
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, len(HORIZONS)),
            nn.Sigmoid(),
        )

    def encode_visit(self,
                     rnfl: torch.Tensor,
                     vf:   torch.Tensor,
                     tab:  torch.Tensor) -> torch.Tensor:
        """Encode one visit: [rnfl_emb | vf_emb | tabular]."""
        e_rnfl = self.encoder(rnfl)
        e_vf   = self.encoder(vf)
        return torch.cat([e_rnfl, e_vf, tab], dim=-1)

    def forward(self, batch: dict) -> torch.Tensor:
        tab = batch["tabular"]

        v0 = self.encode_visit(batch["rnfl_t0"], batch["vf_t0"], tab)   # (B, visit_dim)
        v1 = self.encode_visit(batch["rnfl_t1"], batch["vf_t1"], tab)   # (B, visit_dim)

        seq, _ = self.lstm(torch.stack([v0, v1], dim=1))  # (B, 2, 2*hidden)
        out = self.dropout(seq[:, -1, :])                 # last timestep
        return self.head(out)                             # (B, 3)


# ─────────────────────────────────────────────
# 4. Masked multi-horizon BCE loss
# ─────────────────────────────────────────────

class MaskedMultiHorizonBCE(nn.Module):
    """
    Binary cross-entropy loss that ignores NaN labels.

    For each sample and horizon, the loss is computed only when the
    corresponding label is available (not NaN). This implements the
    masked multi-horizon supervision described in the paper.
    """

    def __init__(self, class_weights: torch.Tensor = None):
        super().__init__()
        self.bce = nn.BCELoss(reduction="none")
        self.class_weights = class_weights  # (num_classes,) for imbalance

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask   = ~torch.isnan(targets)                     # (B, H)
        safe_t = targets.clone()
        safe_t[~mask] = 0.0                                # dummy — will be masked out

        loss = self.bce(preds, safe_t)                     # (B, H)

        if self.class_weights is not None:
            w = torch.where(safe_t == 1,
                            self.class_weights[1],
                            self.class_weights[0])
            loss = loss * w

        loss = loss * mask.float()
        return loss.sum() / mask.float().sum().clamp(min=1.0)


# ─────────────────────────────────────────────
# 5. Training loop
# ─────────────────────────────────────────────

def train_one_epoch(model:      nn.Module,
                    loader:     DataLoader,
                    optimizer:  torch.optim.Optimizer,
                    criterion:  nn.Module,
                    device:     torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        preds  = model(batch)
        loss   = criterion(preds, batch["labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model:    nn.Module,
             loader:   DataLoader,
             device:   torch.device) -> dict:
    """Compute AUROC, accuracy, and F1 averaged across available horizons."""
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = model(batch).cpu().numpy()
        lbls  = batch["labels"].cpu().numpy()
        all_preds.append(preds)
        all_labels.append(lbls)

    preds  = np.vstack(all_preds)    # (N, H)
    labels = np.vstack(all_labels)   # (N, H)

    metrics = {}
    aucs, accs, f1s = [], [], []

    for i, h in enumerate(HORIZONS):
        mask = ~np.isnan(labels[:, i])
        if mask.sum() < 10:
            continue
        y_true = labels[mask, i].astype(int)
        y_prob = preds[mask, i]
        y_pred = (y_prob >= 0.5).astype(int)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")

        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, zero_division=0)

        metrics[f"auc_y{h}"]  = auc
        metrics[f"acc_y{h}"]  = acc
        metrics[f"f1_y{h}"]   = f1
        aucs.append(auc); accs.append(acc); f1s.append(f1)

    metrics["auc_mean"] = float(np.nanmean(aucs)) if aucs else float("nan")
    metrics["acc_mean"] = float(np.nanmean(accs)) if accs else float("nan")
    metrics["f1_mean"]  = float(np.nanmean(f1s))  if f1s  else float("nan")
    return metrics


# ─────────────────────────────────────────────
# 6. Cross-validation
# ─────────────────────────────────────────────

def run_cross_validation(df:           pd.DataFrame,
                          image_dir:   str,
                          backbone:    str,
                          n_folds:     int   = 5,
                          epochs:      int   = 200,
                          lr:          float = 2e-5,
                          batch_size:  int   = 16,
                          weight_decay:float = 1e-2,
                          warmup_epochs:int  = 10,
                          patience:    int   = 11,
                          output_dir:  str   = "./results") -> list:
    """
    Patient-level K-fold cross-validation.

    Fold assignments are stratified by patient ID to prevent patient-level
    contamination, temporal leakage, and label correlation across sequences
    of the same eye.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(42)

    # Patient-level groups for leakage-free splitting
    groups  = df["patient_id"].values
    splitter = GroupKFold(n_splits=n_folds)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(df, groups=groups), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{n_folds}")
        print(f"{'='*60}")

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val   = df.iloc[val_idx].reset_index(drop=True)

        # Compute class weights from training fold
        y_all    = pd.concat([df_train[f"label_y{h}"] for h in HORIZONS]).dropna()
        n_pos    = (y_all == 1).sum()
        n_neg    = (y_all == 0).sum()
        w_pos    = n_neg / n_pos if n_pos > 0 else 1.0
        cw       = torch.tensor([1.0, w_pos], dtype=torch.float32).to(device)

        train_ds = GlaucomaSequenceDataset(df_train, image_dir, fit_scaler=True, augment=True)
        val_ds   = GlaucomaSequenceDataset(df_val,   image_dir,
                                            scaler=train_ds.scaler, augment=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                   num_workers=4, pin_memory=True)

        tab_dim = train_ds.tab_data.shape[1]
        model   = MultimodalBiLSTM(backbone_name=backbone, tab_dim=tab_dim).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                       weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        criterion = MaskedMultiHorizonBCE(class_weights=cw)

        best_auc, patience_ctr = 0.0, 0
        best_state = None

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            if epoch <= warmup_epochs:
                scheduler.step()

            val_metrics = evaluate(model, val_loader, device)
            auc_mean    = val_metrics.get("auc_mean", 0.0)

            print(f"  Epoch {epoch:3d} | loss {train_loss:.4f} | "
                  f"val AUC {auc_mean:.4f} | "
                  f"acc {val_metrics.get('acc_mean', 0):.4f} | "
                  f"F1 {val_metrics.get('f1_mean', 0):.4f}")

            if auc_mean > best_auc:
                best_auc   = auc_mean
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        # Save best model for this fold
        ckpt_path = Path(output_dir) / f"fold{fold}_{backbone}_best.pt"
        torch.save(best_state, ckpt_path)
        print(f"  Best val AUC: {best_auc:.4f} → saved to {ckpt_path}")

        # Final evaluation with best weights
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        final_metrics = evaluate(model, val_loader, device)
        final_metrics["fold"] = fold
        fold_results.append(final_metrics)

    # Aggregate across folds
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS (mean ± std)")
    print("=" * 60)
    results_df = pd.DataFrame(fold_results)
    for col in ["auc_mean", "acc_mean", "f1_mean"]:
        if col in results_df.columns:
            print(f"  {col}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")

    results_path = Path(output_dir) / f"{backbone}_cv_results.json"
    results_df.to_json(results_path, orient="records", indent=2)
    print(f"\nFull results saved → {results_path}")

    return fold_results


# ─────────────────────────────────────────────
# 7. Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train multimodal Bi-LSTM for multi-horizon glaucoma progression prediction."
    )
    parser.add_argument("--sequences_path", required=True,
                        help="Path to sequences.csv from sequence_generation.py.")
    parser.add_argument("--image_dir",      required=True,
                        help="Root directory containing cpRNFL and VF images.")
    parser.add_argument("--backbone",       default="convnext",
                        choices=list(BACKBONE_DIMS.keys()),
                        help="Image encoder backbone (default: convnext).")
    parser.add_argument("--output_dir",     default="./results",
                        help="Directory to save checkpoints and results.")
    parser.add_argument("--n_folds",        type=int,   default=5)
    parser.add_argument("--epochs",         type=int,   default=200)
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--weight_decay",   type=float, default=1e-2)
    parser.add_argument("--warmup_epochs",  type=int,   default=10)
    parser.add_argument("--patience",       type=int,   default=11)
    args = parser.parse_args()

    df = pd.read_csv(args.sequences_path)
    print(f"Loaded {len(df):,} sequences from {args.sequences_path}")

    run_cross_validation(
        df            = df,
        image_dir     = args.image_dir,
        backbone      = args.backbone,
        n_folds       = args.n_folds,
        epochs        = args.epochs,
        lr            = args.lr,
        batch_size    = args.batch_size,
        weight_decay  = args.weight_decay,
        warmup_epochs = args.warmup_epochs,
        patience      = args.patience,
        output_dir    = args.output_dir,
    )
