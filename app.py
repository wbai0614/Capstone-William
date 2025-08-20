# app.py

from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
import os, pickle, numpy as np, pandas as pd

app = Flask(__name__)

# -----------------------------
# Model paths
# -----------------------------
LOGREG_PATH = "models/customer_churn_logreg.pkl"
DTREE_PATH  = "models/customer_churn_dtree.pkl"
SVM_PATH    = "models/customer_churn_svm.pkl"
KMEANS_PATH = "models/customer_kmeans.pkl"
LINREG_PATH = "models/sales_linear_reg.pkl"   # <- regression model

def _require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model file: {path}")

for p in [LOGREG_PATH, DTREE_PATH, SVM_PATH, KMEANS_PATH, LINREG_PATH]:
    _require(p)

with open(LOGREG_PATH, "rb") as f:  logreg_pipe = pickle.load(f)
with open(DTREE_PATH,  "rb") as f:  dtree_pipe  = pickle.load(f)
with open(SVM_PATH,    "rb") as f:  svm_pipe    = pickle.load(f)
with open(KMEANS_PATH, "rb") as f:  kmeans_model= pickle.load(f)
with open(LINREG_PATH, "rb") as f:  linreg_pipe = pickle.load(f)

# -----------------------------
# Feature schemas (match training)
# -----------------------------
# Churn/classifiers & KMeans numeric base
NUM_COLS = ["price", "quantity", "total_value", "age", "tenure_months"]
CAT_COLS = ["gender", "region", "segment", "product_name", "category", "sentiment"]

# KMeans expects ONLY numeric features (already standardized at training time)
KMEANS_ORDER = NUM_COLS[:]  # order matters for list input

# Linear Regression (target = total_value) → exclude total_value from inputs
REG_NUM_COLS = ["price", "quantity", "age", "tenure_months"]
REG_CAT_COLS = CAT_COLS[:]  # same categoricals as above

# -----------------------------
# Helpers
# -----------------------------
def bad_request(msg, code=400):
    return jsonify({"error": msg}), code

def ensure_keys(payload: dict, keys: list, context: str):
    missing = [k for k in keys if k not in payload]
    if missing:
        raise BadRequest(f"Missing keys for {context}: {missing}")

def predict_classifier_row(pipe, features_dict, model_name: str):
    ensure_keys(features_dict, NUM_COLS + CAT_COLS, model_name)
    X_row = pd.DataFrame([{k: features_dict[k] for k in NUM_COLS + CAT_COLS}])
    y_pred = int(pipe.predict(X_row)[0])
    try:
        proba = float(pipe.predict_proba(X_row)[0, 1])
    except Exception:
        proba = None
    return y_pred, proba

def predict_kmeans_row(features):
    if isinstance(features, dict):
        ensure_keys(features, KMEANS_ORDER, "kmeans")
        vals = [features[k] for k in KMEANS_ORDER]
    elif isinstance(features, list):
        if len(features) != len(KMEANS_ORDER):
            raise BadRequest(f"For 'kmeans' list input, expected {len(KMEANS_ORDER)} values in order {KMEANS_ORDER}.")
        vals = features
    else:
        raise BadRequest("For 'kmeans', 'features' must be a dict or a list.")
    arr = np.array(vals, dtype=float).reshape(1, -1)
    cluster = int(kmeans_model.predict(arr)[0])
    return cluster, vals

def predict_linreg_row(features_dict):
    # Regression uses a different schema (no total_value in inputs)
    ensure_keys(features_dict, REG_NUM_COLS + REG_CAT_COLS, "linreg")
    X_row = pd.DataFrame([{k: features_dict[k] for k in REG_NUM_COLS + REG_CAT_COLS}])
    y_hat = float(linreg_pipe.predict(X_row)[0])
    return y_hat

# -----------------------------
# Endpoints
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Customer Sales ML API is running (logreg, dtree, svm, kmeans, linreg)"}), 200

@app.route("/schema", methods=["GET"])
def schema():
    return jsonify({
        "models": {
            "logreg_churn": {
                "model_type": "logreg",
                "required_fields": NUM_COLS + CAT_COLS,
                "notes": "Pipeline with impute + scale/one-hot + LogisticRegression.",
                "example_payload": {
                    "model_type": "logreg",
                    "features": {
                        "price": 45000, "quantity": 2, "total_value": 90000,
                        "age": 50, "tenure_months": 44,
                        "gender": "Female", "region": "West", "segment": "Corporate",
                        "product_name": "Projector", "category": "Electronics", "sentiment": "Positive"
                    }
                }
            },
            "decision_tree_churn": {
                "model_type": "dtree",
                "required_fields": NUM_COLS + CAT_COLS,
                "notes": "Pipeline with impute + one-hot + DecisionTree.",
                "example_payload": {
                    "model_type": "dtree",
                    "features": {
                        "price": 12000, "quantity": 2, "total_value": 24000,
                        "age": 40, "tenure_months": 37,
                        "gender": "Female", "region": "West", "segment": "Small Business",
                        "product_name": "Desk", "category": "Furniture", "sentiment": "Negative"
                    }
                }
            },
            "svm_churn": {
                "model_type": "svm",
                "required_fields": NUM_COLS + CAT_COLS,
                "notes": "Pipeline with impute + scale + one-hot + SVM(probability=True).",
                "example_payload": {
                    "model_type": "svm",
                    "features": {
                        "price": 45000, "quantity": 4, "total_value": 180000,
                        "age": 33, "tenure_months": 25,
                        "gender": "Male", "region": "South", "segment": "Corporate",
                        "product_name": "Projector", "category": "Electronics", "sentiment": "Neutral"
                    }
                }
            },
            "kmeans_clusters": {
                "model_type": "kmeans",
                "required_numeric_fields": KMEANS_ORDER,
                "notes": "Numeric-only. Dict (order-free) or list (order matters).",
                "example_payload_dict": {
                    "model_type": "kmeans",
                    "features": {"price": 12000, "quantity": 2, "total_value": 24000, "age": 40, "tenure_months": 37}
                },
                "example_payload_list": {
                    "model_type": "kmeans",
                    "features": [12000, 2, 24000, 40, 37],
                    "order": KMEANS_ORDER
                }
            },
            "linreg_sales": {
                "model_type": "linreg",
                "required_fields": REG_NUM_COLS + REG_CAT_COLS,
                "notes": "Predicts total_value (sales) from features. Do NOT include total_value in inputs.",
                "example_payload": {
                    "model_type": "linreg",
                    "features": {
                        "price": 45000, "quantity": 2, "age": 50, "tenure_months": 44,
                        "gender": "Female", "region": "West", "segment": "Corporate",
                        "product_name": "Projector", "category": "Electronics", "sentiment": "Positive"
                    }
                }
            }
        }
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST JSON:
    {
      "model_type": "logreg" | "dtree" | "svm" | "kmeans" | "linreg",
      "features": { ... }  # dict for classifiers/linreg; dict or list for kmeans
    }
    """
    try:
        data = request.get_json(force=True, silent=False)
        if not data:
            return bad_request("Expected JSON body.")
        model_type = data.get("model_type")
        features = data.get("features")

        if model_type not in {"logreg", "dtree", "svm", "kmeans", "linreg"}:
            return bad_request("Invalid model_type. Use 'logreg', 'dtree', 'svm', 'kmeans', or 'linreg'.")
        if features is None:
            return bad_request("Missing 'features' in request body.")

        if model_type == "logreg":
            if not isinstance(features, dict): return bad_request("For 'logreg', 'features' must be a dict.")
            y_pred, proba = predict_classifier_row(logreg_pipe, features, "logreg")
            return jsonify({"model_type":"logreg","input":features,"prediction":y_pred,"probability_of_churn":proba}), 200

        if model_type == "dtree":
            if not isinstance(features, dict): return bad_request("For 'dtree', 'features' must be a dict.")
            y_pred, proba = predict_classifier_row(dtree_pipe, features, "dtree")
            return jsonify({"model_type":"dtree","input":features,"prediction":y_pred,"probability_of_churn":proba}), 200

        if model_type == "svm":
            if not isinstance(features, dict): return bad_request("For 'svm', 'features' must be a dict.")
            y_pred, proba = predict_classifier_row(svm_pipe, features, "svm")
            return jsonify({"model_type":"svm","input":features,"prediction":y_pred,"probability_of_churn":proba}), 200

        if model_type == "kmeans":
            cluster, ordered_vals = predict_kmeans_row(features)
            return jsonify({"model_type":"kmeans","order":KMEANS_ORDER,"features":ordered_vals,"prediction_cluster":cluster}), 200

        if model_type == "linreg":
            if not isinstance(features, dict): return bad_request("For 'linreg', 'features' must be a dict.")
            y_hat = predict_linreg_row(features)
            return jsonify({"model_type":"linreg","input":features,"predicted_sales_value": y_hat}), 200

        return bad_request("Unhandled model_type.")

    except BadRequest as e:
        return bad_request(str(e))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    POST JSON:
    {
      "model_type": "logreg" | "dtree" | "svm" | "kmeans" | "linreg",
      "rows": [ {features...}, ... ]  # dicts for classifiers/linreg; dicts or lists for kmeans
    }
    """
    try:
        data = request.get_json(force=True, silent=False)
        if not data: return bad_request("Expected JSON body.")
        model_type = data.get("model_type")
        rows = data.get("rows")

        if model_type not in {"logreg", "dtree", "svm", "kmeans", "linreg"}:
            return bad_request("Invalid model_type.")
        if not isinstance(rows, list) or len(rows) == 0:
            return bad_request("'rows' must be a non-empty list.")

        outputs = []

        if model_type in {"logreg", "dtree", "svm"}:
            pipe = {"logreg": logreg_pipe, "dtree": dtree_pipe, "svm": svm_pipe}[model_type]
            for row in rows:
                if not isinstance(row, dict):
                    return bad_request(f"For '{model_type}', each row must be a dict.")
                ensure_keys(row, NUM_COLS + CAT_COLS, f"{model_type} batch")
            X_df = pd.DataFrame([{k: r[k] for k in NUM_COLS + CAT_COLS} for r in rows])
            preds = pipe.predict(X_df).astype(int).tolist()
            try:
                probas = pipe.predict_proba(X_df)[:, 1].astype(float).tolist()
            except Exception:
                probas = [None] * len(rows)
            for i in range(len(rows)):
                outputs.append({"input": rows[i], "prediction": preds[i], "probability_of_churn": probas[i]})

        elif model_type == "kmeans":
            for row in rows:
                cluster, ordered_vals = predict_kmeans_row(row)
                outputs.append({"order": KMEANS_ORDER, "features": ordered_vals, "prediction_cluster": cluster})

        else:  # linreg
            for row in rows:
                if not isinstance(row, dict):
                    return bad_request("For 'linreg', each row must be a dict.")
                ensure_keys(row, REG_NUM_COLS + REG_CAT_COLS, "linreg batch")
            X_df = pd.DataFrame([{k: r[k] for k in REG_NUM_COLS + REG_CAT_COLS} for r in rows])
            y_hat = linreg_pipe.predict(X_df).astype(float).tolist()
            for i in range(len(rows)):
                outputs.append({"input": rows[i], "predicted_sales_value": y_hat[i]})

        return jsonify({"model_type": model_type, "results": outputs}), 200

    except BadRequest as e:
        return bad_request(str(e))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


#if __name__ == "__main__":
    # Local dev only; use gunicorn/waitress in production
    #app.run(host="0.0.0.0", port=5000, debug=True)
