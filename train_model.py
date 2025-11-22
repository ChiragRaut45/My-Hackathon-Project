# train_model.py
"""
Train XGBoost model on the scaled (0–1) dataset.
Outputs:
  - model/xgb_model.pkl
  - model/label_encoder.pkl

Run:
  python train_model.py
"""

import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, recall_score, accuracy_score

# ---------- CONFIG ----------
DATA_PATH = "data/medical_dataset_scaled_0_1.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- helpers ----------
def load_data(path):
    return pd.read_csv(path)

# ---------- training ----------
def train(df):
    target_col = "Disease"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # encode target labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print(f"[OK] Saved label encoder. Classes: {list(le.classes_)}")

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    # XGBoost classifier - tuned defaults for multiclass tabular
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    print("[INFO] Training XGBoost...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # evaluation
    y_pred = model.predict(X_test)
    print("\n====== MODEL PERFORMANCE ======")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Recall (macro):", recall_score(y_test, y_pred, average="macro"))

    # save model
    joblib.dump(model, os.path.join(MODEL_DIR, "xgb_model.pkl"))
    print(f"[OK] Saved model to {os.path.join(MODEL_DIR, 'xgb_model.pkl')}")

    return model, le

if __name__ == "__main__":
    print("[INFO] Loading dataset:", DATA_PATH)
    df = load_data(DATA_PATH)
    print("[INFO] Dataset shape:", df.shape)
    train(df)
