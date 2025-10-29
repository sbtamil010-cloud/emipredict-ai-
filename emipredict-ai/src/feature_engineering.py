import pandas as pd
import numpy as np

def create_financial_ratios(df):
    """Generate key financial ratios and derived features."""
    df['debt_to_income_ratio'] = (df['current_emi_amount'] + df['existing_loans']) / (df['monthly_salary'] + 1)
    df['expense_to_income_ratio'] = (df['groceries_utilities'] + df['other_monthly_expenses'] + df['travel_expenses']) / (df['monthly_salary'] + 1)
    df['affordability_index'] = (df['monthly_salary'] - (df['groceries_utilities'] + df['other_monthly_expenses'] + df['current_emi_amount'])) / (df['monthly_salary'] + 1)
    df['savings_ratio'] = (df['bank_balance'] + df['emergency_fund']) / (df['monthly_salary'] + 1)
    print("✅ Financial ratios generated successfully.")
    return df

def generate_risk_features(df):
    """Create synthetic risk scoring and stability indicators."""
    df['credit_risk_score'] = (df['credit_score'] / 850) * 0.6 + (df['affordability_index'] * 0.4)
    df['employment_stability_score'] = np.where(df['years_of_employment'] >= 5, 1, df['years_of_employment'] / 5)
    df['risk_category'] = pd.cut(df['credit_risk_score'], bins=[0, 0.4, 0.7, 1.0], labels=['High', 'Medium', 'Low'])
    print("✅ Risk features created successfully.")
    return df

def interaction_features(df):
    """Develop interaction terms between key variables."""
    df['income_x_credit'] = df['monthly_salary'] * df['credit_score']
    df['loan_burden'] = df['existing_loans'] + df['current_emi_amount']
    df['dependents_x_expenses'] = df['dependents'] * df['other_monthly_expenses']
    df['risk_weighted_income'] = df['monthly_salary'] * (df['credit_score'] / 850)
    print("✅ Interaction features generated.")
    return df

def feature_engineering_pipeline(file_path):
    """Run full feature engineering pipeline on dataset."""
    df = pd.read_csv(file_path)
    print(f"Loaded dataset for feature engineering with shape {df.shape}.")

    df = create_financial_ratios(df)
    df = generate_risk_features(df)
    df = interaction_features(df)

    print("🎯 Feature engineering pipeline completed.")
    return df

if __name__ == "__main__":
    file_path = '../data/EMI_dataset.csv'
    engineered_df = feature_engineering_pipeline(file_path)
    engineered_df.to_csv('../data/EMI_dataset_engineered.csv', index=False)
    print("Saved engineered dataset as EMI_dataset_engineered.csv")