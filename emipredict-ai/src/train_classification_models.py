import os
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# NOTE: these imports assume the other modules you already created exist in src/
from src.data_preprocessing import load_dataset, clean_data, encode_categorical, split_data
from src.feature_engineering import create_financial_ratios, generate_risk_features, interaction_features
from src.mlflow_tracking import set_tracking_uri, start_run, log_params, log_metrics, log_model_sklearn


def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)
    metrics = {
        'accuracy': float(accuracy_score(y_val, preds)),
        'precision': float(precision_score(y_val, preds, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_val, preds, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_val, preds, average='weighted', zero_division=0))
    }
    return metrics


def build_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def prepare_features(df):
    # Apply feature engineering steps (keeps everything in dataframe form)
    df = create_financial_ratios(df)
    df = generate_risk_features(df)
    df = interaction_features(df)
    return df


def main(args):
    # Configure MLflow tracking (None => local mlruns file-based)
    set_tracking_uri(args.mlflow_uri)

    # 1) Load and basic cleaning
    df = load_dataset(args.data_path)
    df = clean_data(df)

    # 2) Feature engineering
    df = prepare_features(df)

    # 3) Check target
    target_col = args.target_col
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # 4) Encode categoricals (this function will label-encode string cols)
    df = encode_categorical(df)

    # 5) Split dataset into train/val/test (uses your split_data util)
    X_train, X_val, X_test, y_class_train, y_class_val, y_class_test, *_ = split_data(
        df, target_col_class=target_col
    )

    # 6) Build scaler for numeric features only and scale
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = build_scaler(X_train[numeric_cols])

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_train_scaled[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])

    # 7) Models to train
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
    }

    results = {}
    os.makedirs(args.output_dir, exist_ok=True)

    for name, model in models.items():
        # Start and track run in MLflow
        with start_run(run_name=f'classification_{name}'):
            log_params({'model': name})

            # Fit model
            model.fit(X_train_scaled.values, y_class_train.values)

            # Evaluate on validation set
            metrics = evaluate_model(model, X_val_scaled.values, y_class_val.values)
            log_metrics(metrics)

            # Save artifacts locally: model + scaler + metadata
            art_dir = os.path.join(args.output_dir, name)
            os.makedirs(art_dir, exist_ok=True)
            joblib.dump(model, os.path.join(art_dir, 'model.joblib'))
            joblib.dump(scaler, os.path.join(art_dir, 'scaler.joblib'))
            with open(os.path.join(art_dir, 'meta.json'), 'w') as f:
                json.dump({'numeric_cols': numeric_cols, 'target_col': target_col}, f)

            # Try to log model to MLflow (sklearn wrapper)
            try:
                log_model_sklearn(model, artifact_path=f"{name}_sklearn")
            except Exception as e:
                print('Warning: MLflow model log failed:', e)

            results[name] = metrics
            print(f"Trained {name} -> {metrics}")

    # Save a small training summary for quick reference
    with open(os.path.join(args.output_dir, 'training_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print('All classification models trained and logged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=os.path.join('..', 'data', 'EMI_dataset.csv'))
    parser.add_argument('--target_col', type=str, default='emi_eligibility')
    parser.add_argument('--mlflow_uri', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=os.path.join('..', 'artifacts', 'classification'))
    args = parser.parse_args()
    main(args)
