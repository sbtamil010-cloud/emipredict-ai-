import os
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.data_preprocessing import load_dataset, clean_data, encode_categorical, split_data
from src.feature_engineering import create_financial_ratios, generate_risk_features, interaction_features
from src.mlflow_tracking import set_tracking_uri, start_run, log_params, log_metrics, log_model_sklearn


def evaluate_regression(model, X_val, y_val):
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    mape = np.mean(np.abs((y_val - preds) / (y_val + 1))) * 100
    return {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2), 'mape': float(mape)}


def build_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def prepare_features(df):
    df = create_financial_ratios(df)
    df = generate_risk_features(df)
    df = interaction_features(df)
    return df


def main(args):
    set_tracking_uri(args.mlflow_uri)

    # Load and preprocess data
    df = load_dataset(args.data_path)
    df = clean_data(df)

    # Feature engineering
    df = prepare_features(df)

    target_col = args.target_col
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Encode categorical variables
    df = encode_categorical(df)

    # Split data for regression
    X_train, X_val, X_test, *_ , y_reg_train, y_reg_val, y_reg_test = split_data(df, target_col_reg=target_col)

    # Scale numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = build_scaler(X_train[numeric_cols])
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_train_scaled[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])

    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        'xgboost': XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=0)
    }

    results = {}
    os.makedirs(args.output_dir, exist_ok=True)

    for name, model in models.items():
        with start_run(run_name=f'regression_{name}'):
            log_params({'model': name})

            model.fit(X_train_scaled.values, y_reg_train.values)
            metrics = evaluate_regression(model, X_val_scaled.values, y_reg_val.values)
            log_metrics(metrics)

            art_dir = os.path.join(args.output_dir, name)
            os.makedirs(art_dir, exist_ok=True)
            joblib.dump(model, os.path.join(art_dir, 'model.joblib'))
            joblib.dump(scaler, os.path.join(art_dir, 'scaler.joblib'))

            with open(os.path.join(art_dir, 'meta.json'), 'w') as f:
                json.dump({'numeric_cols': numeric_cols, 'target_col': target_col}, f)

            try:
                log_model_sklearn(model, artifact_path=f"{name}_sklearn")
            except Exception as e:
                print('Warning: MLflow model log failed:', e)

            results[name] = metrics
            print(f"Trained {name} -> {metrics}")

    with open(os.path.join(args.output_dir, 'training_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print('All regression models trained and logged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=os.path.join('..', 'data', 'EMI_dataset.csv'))
    parser.add_argument('--target_col', type=str, default='max_monthly_emi')
    parser.add_argument('--mlflow_uri', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=os.path.join('..', 'artifacts', 'regression'))
    args = parser.parse_args()
    main(args)
