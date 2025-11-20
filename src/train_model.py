import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, confusion_matrix
import xgboost as xgb
import os

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_PATH = os.path.join(BASE_DIR, "data", "Telco-Churn.csv")

# --- Configuration ---
# Optimal threshold determined during model tuning (based on maximizing F1-Score)
OPTIMAL_THRESHOLD = 0.5503 
# Class weight ratio (Non-Churn/Churn) to balance the imbalance and prioritize Recall
# Ratio = (Number of Non-Churners) / (Number of Churners)
SCALE_POS_WEIGHT = 7300 / 2700  # Approx. 73%/27% ratio for the training set

# --- 1. Data Loading and Feature Engineering ---
def load_and_engineer_data(filepath='data/Telco-Churn.csv'):
    """Loads the data and applies all final feature engineering steps."""
    try:
        # NOTE: Adjust filepath if 'Telco-Churn.csv' is in a different location
        df = pd.read_csv(filepath) 
    except FileNotFoundError:
        # Raise a clear error if the data file isn't found
        raise FileNotFoundError(f"Data file not found at {filepath}. Please ensure 'Telco-Churn.csv' is in the specified path.")

    # Data Cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Service features for composite score
    servicce_features = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                         'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

    # Handle 'No internet service' and 'No phone service'
    for col in servicce_features:
        if col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                   'StreamingTV', 'StreamingMovies']:
            df[col] = df[col].replace({'No internet service': 'No', 'Yes': 'Yes', 'No': 'No'})

    # Re-map Yes/No features to 1/0
    cols_to_map = ['Partner', 'Dependents', 'PaperlessBilling'] + servicce_features
    for col in cols_to_map:
         if df[col].dtype == 'object':
             df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Engineered Feature 1: Number of Services
    df["num_services"] = df[servicce_features].sum(axis=1)

    # Engineered Feature 2: Tenure Group (Recreating the category used in analysis)
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 72],
        labels=["0-1yrs", "1-2yrs", "2-4yrs", "4-6yrs"], right=False, include_lowest=True
    )

    return df

# --- 2. Preprocessing Pipeline Definition ---
def define_pipeline(df):
    """Defines the ColumnTransformer and the final model pipeline."""
    
    # Define feature lists
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'num_services']
    
    # Identify all remaining categorical/object features (excluding customerID)
    categorical_cols = [col for col in df.select_dtypes(include='object').columns.tolist() if col != 'customerID'] \
                       + ['SeniorCitizen', 'Partner', 'Dependents', 'PaperlessBilling', 'tenure_group']

    # Preprocessor: Standardize numerical features and One-Hot Encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough' # Keep any unlisted columns (which should just be Churn and customerID at this point)
    )

    # Final XGBoost Model (using tuned parameters)
    xgb_tuned = xgb.XGBClassifier(
        objective='binary:logistic',
        learning_rate=0.05,  
        max_depth=5,
        n_estimators=200,
        scale_pos_weight=SCALE_POS_WEIGHT, # Use the global weight configuration
        use_label_encoder=False, 
        eval_metric='logloss',
        random_state=42
    )

    # Full Model Pipeline combining preprocessing and classification
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_tuned)
    ])
    
    return final_pipeline

# --- 3. Training and Evaluation ---
def train_and_evaluate(df, pipeline, threshold):
    """Trains the model and evaluates performance on the test set."""
    
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']
    
    # Split data into training and testing sets (using 15% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    print("Training final XGBoost model...")
    pipeline.fit(X_train, y_train)
    
    # Predict probabilities for thresholding
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Apply the optimal custom threshold to get binary predictions
    y_pred_threshold = (y_proba >= threshold).astype(int)

    # --- Evaluation ---
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print("\n" + "="*70)
    print("          FINAL XGBOOST MODEL PERFORMANCE (TEST SET)")
    print(f"          Custom Threshold (Max F1): {threshold:.4f}")
    print("="*70 + "\n")
    
    print(f"ROC AUC Score: {roc_auc:.4f}\n")
    
    # Print detailed metrics
    print("Full Classification Report:")
    print(classification_report(y_test, y_pred_threshold))
    
    print("Confusion Matrix (Actual vs. Predicted):")
    print(confusion_matrix(y_test, y_pred_threshold))
    print("\n(Rows: Actual [0: No Churn, 1: Churn], Columns: Predicted [0: No Churn, 1: Churn])")


if __name__ == '__main__':
    print("Starting Telco Churn Production Pipeline...")
    
    # Load and Engineer Data
    data = load_and_engineer_data(filepath='Telco-Churn.csv') # Assuming data is in the same directory for this script example

    # Define the final model pipeline
    final_model_pipe = define_pipeline(data)

    # Train and Evaluate using the optimal threshold
    train_and_evaluate(data, final_model_pipe, OPTIMAL_THRESHOLD)

    print("\nPipeline execution complete. Model ready for deployment.")