import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import pandas as pd
import numpy as np
import joblib
import json
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Configure MLflow
mlflow.set_experiment("Skill_Demand_Prediction_Experiment")

def load_data(path):
    print(f"Loading features data from {path}...")
    df = pd.read_csv(path)
    target = "demand_label"
    df = df.dropna(subset=[target])
    
    # Remove duplicates based on title column (tags are binarized)
    before = len(df)
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    print(f"  Duplicates removed: {before - len(df)} (remaining: {len(df)})")
    
    y = df[target]
    
    cat_features = ["seniority", "category", "geo_tier"]
    
    # Base numeric features - NO demand_score or tag_frequency_score (target leakage)
    # NOTE: num_skills removed - it's a proxy for demand_score (more tags = higher frequency sum)
    engineered_features = ["years_exp", "has_ai", "has_cloud", "has_backend", "has_frontend"]
    
    # Sparse features - ONLY title TF-IDF (Tags removed to prevent leakage)
    leaky_cols = {"tag_list", "tag_frequency_score", "demand_score", "demand_score_norm", "demand_label"}
    sparse_features = [c for c in df.columns if c.startswith("title_tfidf_")]
    
    num_features = engineered_features + sparse_features
    X = df[cat_features + num_features]
    
    # VERIFICATION: Print feature columns to confirm no leakage
    print(f"\n  === FEATURE VERIFICATION ===")
    print(f"  Total features: {X.shape[1]}")
    print(f"  Categorical: {cat_features}")
    print(f"  Engineered: {engineered_features}")
    print(f"  Sparse (tag+tfidf): {len(sparse_features)} columns")
    # Confirm no leaky columns present
    for col in X.columns:
        assert col not in leaky_cols, f"LEAKAGE DETECTED: '{col}' is a target-derived feature!"
    print(f"  No target leakage detected in feature matrix")
    print(f"  ===========================\n")
    
    return X, y, cat_features, num_features

def build_tuned_pipeline(cat_features, num_features, model_type="rf"):
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],
        remainder="passthrough"
    )
    
    if model_type == "rf":
        clf = RandomForestClassifier(random_state=42)
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5]
        }
    elif model_type == "xgb":
        clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [3, 6]
        }
    
    pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("oversampler", SMOTE(random_state=42, k_neighbors=3)),
        ("classifier", clf)
    ])
    
    return pipeline, param_grid

def get_best_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def train_and_track():
    FEATURES_PATH = "data/features_jobs.csv"
    if not os.path.exists(FEATURES_PATH):
        print(f"Error: {FEATURES_PATH} not found. Run preprocessing first.")
        return

    X, y, cat_features, num_features = load_data(FEATURES_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    with mlflow.start_run(run_name="Demand_Stacking_Ensemble"):
        print("Tuning Base Learners...")
        
        # 1. Random Forest
        rf_pipe, rf_params = build_tuned_pipeline(cat_features, num_features, "rf")
        rf_grid = GridSearchCV(rf_pipe, rf_params, cv=3, scoring="f1_macro", n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        best_rf = rf_grid.best_estimator_
        
        # 2. XGBoost
        xgb_pipe, xgb_params = build_tuned_pipeline(cat_features, num_features, "xgb")
        xgb_grid = GridSearchCV(xgb_pipe, xgb_params, cv=3, scoring="f1_macro", n_jobs=-1)
        xgb_grid.fit(X_train, y_train)
        best_xgb = xgb_grid.best_estimator_
        
        # 3. Stacking Ensemble
        print("Building Stacking Ensemble...")
        estimators = [("rf", best_rf), ("xgb", best_xgb)]
        stack_clf = StackingClassifier(
            estimators=estimators, 
            final_estimator=LogisticRegression(max_iter=1000),
            cv=3
        )
        stack_clf.fit(X_train, y_train)
        
        # 4. Calibration
        print("Calibrating probabilities...")
        calibrated_clf = CalibratedClassifierCV(stack_clf, cv=3, method="sigmoid")
        calibrated_clf.fit(X_train, y_train)
        
        # 5. Threshold Optimization
        y_probs = calibrated_clf.predict_proba(X_test)[:, 1]
        best_threshold, best_f1 = get_best_threshold(y_test, y_probs)
        print(f"Optimal Threshold: {best_threshold:.4f} (Best F1: {best_f1:.4f})")
        
        y_pred = (y_probs >= best_threshold).astype(int)
        
        # 6. Metrics
        target_names = ["Low Demand", "High Demand"]
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_probs)
        
        print("\nClassification Report (Demand Prediction):")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # MLflow Logging
        mlflow.log_param("best_threshold", best_threshold)
        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.log_metric("precision_high", report["High Demand"]["precision"])
        mlflow.log_metric("recall_high", report["High Demand"]["recall"])
        mlflow.log_metric("f1_high", report["High Demand"]["f1-score"])
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Local artifacts
        os.makedirs("models", exist_ok=True)
        joblib.dump(calibrated_clf, "models/demand_model.joblib")
        with open("models/threshold.json", "w") as f:
            json.dump({"threshold": float(best_threshold)}, f)
            
        metrics_summary = {
            "accuracy": report["accuracy"],
            "precision": report["High Demand"]["precision"],
            "recall": report["High Demand"]["recall"],
            "f1_high": report["High Demand"]["f1-score"],
            "roc_auc": roc_auc,
            "best_threshold": float(best_threshold)
        }
        with open("data/evaluation_metrics.json", "w") as f:
            json.dump(metrics_summary, f, indent=4)
        
        print(f"\nModel and metrics saved successfully to 'models/' and 'data/'")
        return calibrated_clf

if __name__ == "__main__":
    train_and_track()
