# train_kmeans_customer.py
# Train KMeans on RAW numeric features so inference matches app.py inputs.

import os
import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from pathlib import Path

# Must match app.py KMEANS_ORDER
NUM_COLS = ["price", "quantity", "total_value", "age", "tenure_months"]

INPUT_CSV  = "customer_sales_raw.csv"    # produced by your EDA step
MODEL_PATH = Path("models/customer_kmeans.pkl")

def main():
    df = pd.read_csv(INPUT_CSV)

    # Keep only numeric features (RAW scale to match API inputs)
    X = df[NUM_COLS].copy()

    # Impute numerics (median) to prevent NaNs
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    # Optional: if total_value ~= price*quantity dominates, you can drop it:
    # NUM_COLS.remove("total_value"); X_imp = imputer.fit_transform(df[NUM_COLS])

    # Train KMeans
    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    km.fit(X_imp)

    # Diagnostics
    labels = km.labels_
    print("Cluster assignments (first 10):", labels[:10].tolist())
    print("\nCluster sizes:\n", pd.Series(labels).value_counts().sort_index())
    print("\nInertia (sum of squared distances):", km.inertia_)
    if "churn" in df.columns:
        print("\nCluster vs Churn:\n", pd.crosstab(labels, df.loc[X.index, "churn"]))
    if "segment" in df.columns:
        print("\nCluster vs Segment:\n", pd.crosstab(labels, df.loc[X.index, "segment"]))

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(km, f)

    print(f"✅ KMeans model saved to {MODEL_PATH}")
    print("✅ IMPORTANT: Ensure app.py KMEANS_ORDER matches:", NUM_COLS)

if __name__ == "__main__":
    main()
