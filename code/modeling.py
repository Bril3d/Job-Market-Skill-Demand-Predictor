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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Configure MLflow
mlflow.set_experiment("Salary_Prediction_Experiment")

def load_data(path):
    print(f"Loading processed data from {path}...")
    df = pd.read_csv(path)
    target = "is_high_salary"
    df = df.dropna(subset=[target])
    y = df[target]
    cat_features = ["seniority", "category", "geo_tier"]
    num_features = [c for c in df.columns if (c.startswith("tag_") and c != "tag_list") or c.startswith("title_tfidf_")]
    X = df[cat_features + num_features]
    return X, y, cat_features, num_features

def build_tuned_pipeline(cat_features, num_features, model_type="rf"):
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],
        remainder="passthrough"
    )
    
    if model_type == "rf":
        clf = RandomForestClassifier(random_state=42)
        param_grid = {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5]
        }
    elif model_type == "xgb":
        clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
        param_grid = {
            "classifier__n_estimators": [50, 100],
            "classifier__learning_rate": [0.01, 0.1],
            "classifier__max_depth": [3, 6]
        }
    
    pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", clf)
    ])
    
    return pipeline, param_grid

def train_and_track():
    X, y, cat_features, num_features = load_data("data/processed_jobs.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    with mlflow.start_run(run_name="Stacking_Ensemble_Tuning"):
        print("Tuning Base Learners...")
        # 1. Tune Random Forest
        rf_pipe, rf_params = build_tuned_pipeline(cat_features, num_features, "rf")
        rf_grid = GridSearchCV(rf_pipe, rf_params, cv=3, scoring="f1_macro", n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        best_rf = rf_grid.best_estimator_
        mlflow.log_params({f"rf_best_{k}": v for k, v in rf_grid.best_params_.items()})
        
        # 2. Tune XGBoost
        xgb_pipe, xgb_params = build_tuned_pipeline(cat_features, num_features, "xgb")
        xgb_grid = GridSearchCV(xgb_pipe, xgb_params, cv=3, scoring="f1_macro", n_jobs=-1)
        xgb_grid.fit(X_train, y_train)
        best_xgb = xgb_grid.best_estimator_
        mlflow.log_params({f"xgb_best_{k}": v for k, v in xgb_grid.best_params_.items()})
        
        # 3. Build Stacking Ensemble with Tuned Models
        print("Building Stacking Ensemble...")
        estimators = [("rf", best_rf), ("xgb", best_xgb)]
        stack_clf = StackingClassifier(
            estimators=estimators, 
            final_estimator=LogisticRegression(class_weight="balanced"),
            cv=5
        )
        
        stack_clf.fit(X_train, y_train)
        
        # 4. Evaluation
        y_pred = stack_clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        roc_auc = roc_auc_score(y_test, stack_clf.predict_proba(X_test)[:, 1])
        
        print(f"\nF1 (Macro): {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        # Log Metrics to MLflow
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Log Confusion Matrix as an artifact
        cm = confusion_matrix(y_test, y_pred)
        cm_path = "data/confusion_matrix.json"
        with open(cm_path, "w") as f:
            json.dump(cm.tolist(), f)
        mlflow.log_artifact(cm_path)
        
        # Log Model
        mlflow.sklearn.log_model(stack_clf, "salary_model_stacking")
        
        # Persistent save for app
        os.makedirs("models", exist_ok=True)
        joblib.dump(stack_clf, "models/salary_model.joblib")
        
        print("\nModel and metrics logged to MLflow successfully.")
        return stack_clf

if __name__ == "__main__":
    train_and_track()
