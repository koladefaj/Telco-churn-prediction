# 📞 Telco Customer Churn Prediction: XGBoost Pipeline

> An end-to-end Machine Learning project to identify high-risk Telco customers and predict churn using an optimized XGBoost Classifier.

## 🎯 Project Goal

The primary objective of this project is to maximize the **Recall** for the churn class (True Positives). Identifying the maximum number of potential churners allows the business to deploy proactive retention campaigns effectively.

The final model is an **XGBoost Classifier** optimized for the **F1-Score** at an adjusted classification threshold.

---

## ✨ Key Features & Pipeline Overview

* **Comprehensive Data Handling:** Robust cleaning and imputation for the `TotalCharges` feature.
* **Feature Engineering:** Creation of highly predictive signals like `tenure_group`, `is_new_customer`, and `num_services`.
* **Full ML Pipeline:** Uses `sklearn`'s `Pipeline` and `ColumnTransformer` for automated scaling (`StandardScaler`) and encoding (`OneHotEncoder`).
* **Imbalanced Learning:** XGBoost is trained using the **`scale_pos_weight`** hyperparameter to account for class imbalance and prioritize the minority (churn) class.
* **Threshold Optimization:** The model uses a fixed, F1-optimized classification threshold ($\approx 0.5503$) instead of the default $0.5$ to balance Precision and Recall for maximum business impact.

---

## 🛠️ Setup and Installation

### Prerequisites

* Python 3.8+
* The project assumes the data file `Telco-Churn.csv` is located in a structure accessible by the scripts (e.g., `data/Telco-Churn.csv` or in the same directory as `train_model.py`).

### Local Environment Setup

1.  **Clone the Repository:**
    ```bash
    git clone <YOUR_REPO_URL>
    cd <your-subfolder-name>
    ```

2.  **Install Dependencies:**
    The project relies on standard ML libraries. You can generate a `requirements.txt` from the imported libraries:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn xgboost
    ```

---

## 🚀 Usage

### 1. Training the Model (`train_model.py`)

The `train_model.py` script executes the full pipeline, from data loading and feature engineering to training and evaluation.

To run the training process and see the final metrics:
```bash
python train_model.py
```
### 2. Making Predictions (`predict_churn.py`)
The `predict_churn.py` script is designed for production inference, loading the trained model (assuming it's saved as models/telco_churn_pipeline.pkl) and applying it to new data.

Usage Example:

To predict churn for customers listed in data/new_customers.csv and save the results to data/predictions.csv:
```bash
python predict_churn.py --input data/new_customers.csv --output data/predictions.csv
```
