# src/utils.py
"""
Utility functions for EMIPredict AI project.
Includes:
- Data I/O helpers
- Metric computations
- Logging utilities
- Visualization support
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)


# ============================================================
# 📦 FILE OPERATIONS
# ============================================================

def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV with error handling."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {file_path} | Shape: {df.shape}")
    return df


def save_csv(df: pd.DataFrame, file_path: str):
    """Save DataFrame to CSV."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"💾 Saved DataFrame → {file_path} | Shape: {df.shape}")


def save_model(model, file_path: str):
    """Save ML model using joblib."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"✅ Model saved at {file_path}")


def load_model(file_path: str):
    """Load ML model from joblib file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Model file not found: {file_path}")
    model = joblib.load(file_path)
    print(f"✅ Loaded model from {file_path}")
    return model


def save_json(data: dict, file_path: str):
    """Save dictionary as JSON."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"💾 JSON saved to {file_path}")


def load_json(file_path: str) -> dict:
    """Load dictionary from JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ JSON file not found: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


# ============================================================
# 📊 METRICS CALCULATION
# ============================================================

def classification_metrics(y_true, y_pred) -> dict:
    """Compute accuracy, precision, recall, and F1-score."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }


def regression_metrics(y_true, y_pred) -> dict:
    """Compute regression metrics: MAE, MSE, RMSE, R2."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred)
    }


# ============================================================
# 🧠 VISUALIZATION HELPERS
# ============================================================

def plot_feature_importance(model, feature_names, top_n=10, save_path=None):
    """Plot top N feature importances for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print("⚠️ Model does not have feature_importances_. Skipping plot.")
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(feature_names)[idx], importances[idx])
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"📊 Feature importance plot saved → {save_path}")
    plt.show()


def plot_predictions(y_true, y_pred, title="Actual vs Predicted", save_path=None):
    """Plot regression predictions."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linestyle='--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"📈 Prediction plot saved → {save_path}")
    plt.show()


# ============================================================
# 🕒 LOGGING HELPERS
# ============================================================

def timestamp() -> str:
    """Return formatted timestamp."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def log_message(message: str, log_file: str = None):
    """Log a message with timestamp to console and optional log file."""
    msg = f"[{timestamp()}] {message}"
    print(msg)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
