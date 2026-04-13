"""
Regenerate features_processed.pkl from the raw XLSX dataset.

Reproduces the exact feature engineering pipeline from notebooks/02_features.ipynb
using PCOS_data_without_infertility.xlsx (42-feature dataset).

Also extracts imputation medians and scaler params into model_metadata.json
so ml_model.py has a git-tracked fallback when the pkl is missing.
"""

import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

RAW_PATH = DATA_RAW / "PCOS_data_without_infertility.xlsx"
PKL_PATH = DATA_PROC / "features_processed.pkl"
META_PATH = MODELS_DIR / "model_metadata.json"
FEAT_META_PATH = DATA_PROC / "feature_meta.json"

TARGET_COL = "PCOS (Y/N)"


def safe_col(df: pd.DataFrame, *candidates: str) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def main() -> None:
    if not RAW_PATH.exists():
        print(f"ERROR: Raw data not found at {RAW_PATH}")
        print("Download it with: kagglehub.dataset_download('prasoonkottarathil/polycystic-ovary-syndrome-pcos')")
        sys.exit(1)

    # -- Load --
    df_raw = pd.read_excel(RAW_PATH, sheet_name="Full_new")
    print(f"Loaded {df_raw.shape[0]} rows x {df_raw.shape[1]} columns from {RAW_PATH.name}")

    # -- Drop non-informative columns --
    drop_cols = [
        c for c in df_raw.columns
        if df_raw[c].dtype == object
        or "sl" in c.lower()
        or "patient" in c.lower()
        or "file" in c.lower()
    ]
    drop_cols = [c for c in drop_cols if c != TARGET_COL]
    print(f"Dropping: {drop_cols}")
    df = df_raw.drop(columns=drop_cols, errors="ignore").copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"After drop: {df.shape[0]} rows x {df.shape[1]} columns")

    # -- Feature engineering --
    lh_col = safe_col(df, "LH(mIU/mL)", "LH", "lh")
    fsh_col = safe_col(df, "FSH(mIU/mL)", "FSH", "fsh")
    if lh_col and fsh_col:
        df["LH_FSH_ratio"] = df[lh_col] / (df[fsh_col] + 1e-6)
        print(f"Engineered: LH_FSH_ratio  (from '{lh_col}' / '{fsh_col}')")

    fl_l = safe_col(df, "Follicle No. (L)", "Follicle No.(L)")
    fl_r = safe_col(df, "Follicle No. (R)", "Follicle No.(R)")
    if fl_l and fl_r:
        df["follicle_asymmetry"] = (df[fl_l] - df[fl_r]).abs()
        df["follicle_total"] = df[fl_l] + df[fl_r]
        print(f"Engineered: follicle_asymmetry, follicle_total")

    # -- Select features --
    all_features = [c for c in df.columns if c != TARGET_COL]

    high_missing = [c for c in all_features if df[c].isnull().mean() > 0.5]
    if high_missing:
        print(f"Dropping (>50% missing): {high_missing}")
        all_features = [c for c in all_features if c not in high_missing]

    low_var = [c for c in all_features if df[c].std() < 1e-6]
    if low_var:
        print(f"Dropping (near-zero variance): {low_var}")
        all_features = [c for c in all_features if c not in low_var]

    print(f"\nSelected {len(all_features)} features")

    # Cross-check against model_metadata.json
    if META_PATH.exists():
        with open(META_PATH) as f:
            meta = json.load(f)
        expected = meta.get("feature_names", [])
        if expected and set(expected) != set(all_features):
            missing_in_data = set(expected) - set(all_features)
            extra_in_data = set(all_features) - set(expected)
            if missing_in_data:
                print(f"WARNING: Model expects features not in data: {missing_in_data}")
            if extra_in_data:
                print(f"NOTE: Data has features not in model (will be dropped): {extra_in_data}")
            all_features = [f for f in expected if f in all_features]
            print(f"Using {len(all_features)} features matching model_metadata.json order")
        else:
            all_features = list(expected)
            print("Features match model_metadata.json")

    # -- Impute --
    X = df[all_features].copy()
    y = df[TARGET_COL].copy()

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=all_features,
        index=X.index,
    )

    valid_idx = y.dropna().index
    X_imputed = X_imputed.loc[valid_idx]
    y = y.loc[valid_idx].astype(int)

    print(f"After imputation: {X_imputed.shape[0]} rows x {X_imputed.shape[1]} features")
    print(f"Label distribution: {dict(y.value_counts().sort_index())}")

    # -- Split --
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print(f"Train: {X_train.shape[0]} rows  |  PCOS rate: {y_train.mean():.3f}")
    print(f"Test : {X_test.shape[0]} rows  |  PCOS rate: {y_test.mean():.3f}")

    # -- Scale --
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=all_features,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=all_features,
        index=X_test.index,
    )

    # -- Save pkl --
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    payload = {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "feature_names": all_features,
        "scaler": scaler,
        "imputer": imputer,
        "target_col": TARGET_COL,
    }
    with open(PKL_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved pkl -> {PKL_PATH}  ({PKL_PATH.stat().st_size / 1024:.1f} KB)")

    # -- Save feature_meta.json --
    feat_meta = {
        "feature_names": all_features,
        "n_features": len(all_features),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "pcos_rate_train": float(y_train.mean()),
        "pcos_rate_test": float(y_test.mean()),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
    }
    with open(FEAT_META_PATH, "w") as f:
        json.dump(feat_meta, f, indent=2)
    print(f"Saved feature_meta.json -> {FEAT_META_PATH}")

    # -- Update model_metadata.json with imputation/scaling stats --
    if META_PATH.exists():
        with open(META_PATH) as f:
            model_meta = json.load(f)
    else:
        model_meta = {}

    model_meta["imputation_medians"] = dict(
        zip(all_features, imputer.statistics_.tolist())
    )
    model_meta["scaler_means"] = dict(
        zip(all_features, scaler.mean_.tolist())
    )
    model_meta["scaler_stds"] = dict(
        zip(all_features, scaler.scale_.tolist())
    )

    with open(META_PATH, "w") as f:
        json.dump(model_meta, f, indent=2)
    print(f"Updated model_metadata.json with imputation_medians, scaler_means, scaler_stds")

    # -- Verify --
    print(f"\n{'='*50}")
    print("Verification:")
    print(f"  Features: {len(all_features)}")
    print(f"  Imputer medians (first 5): {dict(list(zip(all_features[:5], imputer.statistics_[:5].tolist())))}")
    print(f"  Scaler means (first 5): {dict(list(zip(all_features[:5], scaler.mean_[:5].tolist())))}")
    print(f"  Scaler stds (first 5): {dict(list(zip(all_features[:5], scaler.scale_[:5].tolist())))}")
    print(f"{'='*50}")
    print("Done!")


if __name__ == "__main__":
    main()
