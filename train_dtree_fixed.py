# train_decision_tree_customer.py
# Decision Tree churn classifier with robust preprocessing + class balancing.

import os
import pickle
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

INPUT_CSV = "customer_sales_raw.csv"     # produced by your EDA step
TARGET    = "churn"                      # 0/1

NUM_COLS = ["price", "quantity", "total_value", "age", "tenure_months"]
CAT_COLS = ["gender", "region", "segment", "product_name", "category", "sentiment"]

MODEL_PATH = Path("models/customer_churn_dtree.pkl")

def main():
    df = pd.read_csv(INPUT_CSV)
    if TARGET not in df.columns:
        raise RuntimeError(f"Target column '{TARGET}' not found in {INPUT_CSV}")

    X = df[NUM_COLS + CAT_COLS].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocess: impute numerics (median), categoricals (mode) + one-hot
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUM_COLS),
            ("cat", Pipeline(steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CAT_COLS),
        ]
    )

    dt = DecisionTreeClassifier(random_state=42)

    pipe = Pipeline([
        ("prep", preprocess),
        ("clf", dt),
    ])

    param_grid = {
        "clf__max_depth": [4, 6, 8, 10, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 5, 20, 50],
        "clf__class_weight": [None, "balanced"]
    }

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=5,
        verbose=0
    )

    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print("Best params:", grid.best_params_)

    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.3f}")

    try:
        y_proba = best.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {auc:.3f}")
    except Exception:
        print("Predict_proba not available (unexpected for DecisionTreeClassifier).")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best, f)

    print(f"\n✅ Decision Tree model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
