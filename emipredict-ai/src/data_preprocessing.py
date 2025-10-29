import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_dataset(file_path):
    """Load the EMI dataset from a CSV file."""
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def clean_data(df):
    """Clean dataset: handle missing values, duplicates, and outliers."""
    df = df.drop_duplicates()

    # Fill numeric columns with median
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical columns with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    print("Data cleaning completed: missing values handled, duplicates removed.")
    return df

def encode_categorical(df):
    """Encode categorical columns using LabelEncoder."""
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    print("Categorical encoding completed.")
    return df

def scale_features(X_train, X_test):
    """Scale numeric features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def split_data(df, target_col_class='emi_eligibility', target_col_reg='max_monthly_emi', test_size=0.2, val_size=0.1, random_state=42):
    """Split dataset into train, validation, and test sets for both classification and regression."""
    X = df.drop([target_col_class, target_col_reg], axis=1)
    y_class = df[target_col_class]
    y_reg = df[target_col_reg]

    X_train, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
        X, y_class, y_reg, test_size=(test_size + val_size), random_state=random_state
    )

    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_class_val, y_class_test, y_reg_val, y_reg_test = train_test_split(
        X_temp, y_class_temp, y_reg_temp, test_size=relative_val_size, random_state=random_state
    )

    print(f"Data split completed: {X_train.shape[0]} train, {X_val.shape[0]} validation, {X_test.shape[0]} test samples.")
    return X_train, X_val, X_test, y_class_train, y_class_val, y_class_test, y_reg_train, y_reg_val, y_reg_test

def preprocess_pipeline(file_path):
    """Full preprocessing pipeline: load, clean, encode, split, and scale."""
    df = load_dataset(file_path)
    df = clean_data(df)
    df = encode_categorical(df)

    X_train, X_val, X_test, y_class_train, y_class_val, y_class_test, y_reg_train, y_reg_val, y_reg_test = split_data(df)

    X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)

    print("Preprocessing pipeline completed successfully.")
    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test,
        'y_class_train': y_class_train,
        'y_class_val': y_class_val,
        'y_class_test': y_class_test,
        'y_reg_train': y_reg_train,
        'y_reg_val': y_reg_val,
        'y_reg_test': y_reg_test,
        'scaler': scaler
    }

if __name__ == "__main__":
    file_path = '../data/EMI_dataset.csv'
    data = preprocess_pipeline(file_path)