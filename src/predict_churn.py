"""
Simple inference script.

Usage (batch CSV):
    python src/predict.py --input data/new_customers.csv --output data/churn_preds.csv

Usage (single sample via JSON file):
    python src/predict.py --input data/sample_input.csv --output data/sample_output.csv
"""
import argparse
import os
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "telco_churn_pipeline.pkl")

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(path)

def predict_batch(model, df):
    # Expect df has same features used during training (customerID may be present)
    X = df.copy()
    if 'customerID' in X.columns:
        ids = X['customerID']
    else:
        ids = pd.Series(range(len(X)), name='customerID')
    X = X.drop(columns=['customerID'], errors='ignore')
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)  # default 0.5; change if you saved optimal threshold
    out = pd.DataFrame({'customerID': ids, 'churn_probability': proba, 'churn_prediction': pred})
    return out

def main(args):
    model = load_model()
    df = pd.read_csv(args.input)
    out = predict_batch(model, df)
    out.to_csv(args.output, index=False)
    print(f"Saved predictions to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=False, default="data/churn_predictions.csv", help="Path to save predictions CSV")
    args = parser.parse_args()
    main(args)
