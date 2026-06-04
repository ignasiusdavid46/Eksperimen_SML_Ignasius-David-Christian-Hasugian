"""
automate_Ignasius-David-Christian-Hasugian.py

Skrip otomasi preprocessing Wine Quality Dataset.
Mengonversi seluruh langkah eksperimen dari notebook menjadi pipeline
yang dapat dijalankan secara otomatis (via CLI atau GitHub Actions).

Penggunaan:
    python automate_Ignasius-David-Christian-Hasugian.py
    python automate_Ignasius-David-Christian-Hasugian.py --input ../winequality-red_raw.csv --output-dir ./winequality_preprocessing

Output:
    winequality_preprocessing/
        winequality_train.csv
        winequality_test.csv
"""

import argparse
import os
import sys
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Konstanta
FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]
TARGET_RAW    = "quality"
TARGET        = "quality_label"
TEST_SIZE     = 0.2
RANDOM_STATE  = 42
THRESHOLD     = 6   # quality >= THRESHOLD → label 1 (good)


# Step Functions

def load_data(path: str) -> pd.DataFrame:
    """Step 1 – Memuat dataset dari file CSV (sep=';' untuk format UCI)."""
    log.info(f"Memuat dataset dari: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    # Dataset UCI menggunakan separator titik koma
    try:
        df = pd.read_csv(path, sep=';')
        if df.shape[1] < 5:        # fallback jika ternyata pakai koma
            df = pd.read_csv(path, sep=',')
    except Exception:
        df = pd.read_csv(path)
    log.info(f"Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def create_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2 – Membuat binary label: quality >= 6 → 1 (good), lainnya → 0."""
    df = df.copy()
    df[TARGET] = (df[TARGET_RAW] >= THRESHOLD).astype(int)
    good     = df[TARGET].sum()
    not_good = len(df) - good
    log.info(f"Binary label dibuat (threshold={THRESHOLD}): Good={good}, Not Good={not_good}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3 – Menghapus baris duplikat."""
    before = len(df)
    df = df.drop_duplicates()
    log.info(f"Duplikat dihapus: {before - len(df)} baris ({before} → {len(df)})")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4 – Mengisi missing values dengan median per fitur."""
    total = df[FEATURES].isnull().sum().sum()
    if total == 0:
        log.info("Tidak ada missing values, langkah ini dilewati.")
        return df
    df = df.copy()
    for col in FEATURES:
        n = df[col].isnull().sum()
        if n > 0:
            med = df[col].median()
            df[col].fillna(med, inplace=True)
            log.info(f"  {col}: {n} missing → diisi median ({med:.4f})")
    log.info(f"Total {total} missing values ditangani.")
    return df


def cap_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Step 5 – Outlier capping dengan metode IQR (Winsorization)."""
    df = df.copy()
    total_clipped = 0
    for col in FEATURES:
        Q1    = df[col].quantile(0.25)
        Q3    = df[col].quantile(0.75)
        IQR   = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        total_clipped += n_out
    log.info(f"Outlier capping selesai: {total_clipped} nilai di-clip")
    return df


def split_and_normalize(df: pd.DataFrame):
    """
    Step 6+7 – Train-test split (stratified) lalu StandardScaler.
    Scaler hanya di-fit pada training set untuk mencegah data leakage.
    """
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    log.info(f"Train-test split: train={len(X_train)}, test={len(X_test)}")

    scaler      = StandardScaler()
    X_train_sc  = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURES)
    X_test_sc   = pd.DataFrame(scaler.transform(X_test),      columns=FEATURES)
    log.info("Standarisasi fitur selesai (StandardScaler)")

    return X_train_sc, X_test_sc, y_train, y_test, scaler


def save_output(X_train, y_train, X_test, y_test, output_dir: str):
    """Step 8 – Menyimpan hasil preprocessing ke folder output."""
    os.makedirs(output_dir, exist_ok=True)

    train_df          = X_train.copy()
    train_df[TARGET]  = y_train.values
    test_df           = X_test.copy()
    test_df[TARGET]   = y_test.values

    train_path = os.path.join(output_dir, "winequality_train.csv")
    test_path  = os.path.join(output_dir, "winequality_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)

    log.info(f"Train set disimpan: {train_path}  {train_df.shape}")
    log.info(f"Test  set disimpan: {test_path}   {test_df.shape}")
    return train_path, test_path


# Pipeline Utama

def preprocess_pipeline(input_path: str, output_dir: str) -> dict:
    """Menjalankan seluruh pipeline preprocessing secara berurutan."""
    log.info("Preprocessing Pipeline - Wine Quality Dataset")

    df = load_data(input_path)
    df = create_binary_label(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = cap_outliers_iqr(df)

    X_train, X_test, y_train, y_test, scaler = split_and_normalize(df)
    train_path, test_path = save_output(X_train, y_train, X_test, y_test, output_dir)

    log.info("Preprocessing selesai.")
    log.info(f"  Output folder: {output_dir}")

    return {
        "train_path": train_path,
        "test_path":  test_path,
        "scaler":     scaler,
        "features":   FEATURES,
        "target":     TARGET,
    }


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automate preprocessing – Wine Quality Dataset"
    )
    parser.add_argument(
        "--input",
        default="../winequality-red_raw.csv",
        help="Path ke raw CSV dataset (default: ../winequality-red_raw.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="./winequality_preprocessing",
        help="Folder output hasil preprocessing (default: ./winequality_preprocessing)",
    )
    args = parser.parse_args()

    result = preprocess_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
    )
    print("\nFile yang dihasilkan:")
    print(f"  - {result['train_path']}")
    print(f"  - {result['test_path']}")
