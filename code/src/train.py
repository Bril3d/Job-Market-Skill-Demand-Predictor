import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def load_data(path):
    print(f"Loading processed data from {path}...")
    df = pd.read_csv(path)
    
    # Identify target and drop rows with missing target
    target = "is_high_salary"
    df = df.dropna(subset=[target])
    
    y = df[target]
    
    # Feature selection
    # Categorical features
    cat_features = ["seniority", "category", "geo_tier"]
    
    # Numerical features (tags and tfidf)
    # Exclude intermediate string columns like 'tag_list'
    num_features = [c for c in df.columns if (c.startswith("tag_") and c != "tag_list") or c.startswith("title_tfidf_")]
    
    X = df[cat_features + num_features]
    
    return X, y, cat_features, num_features

def build_pipeline(cat_features, num_features, model_type="rf"):
    # Preprocessor for categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
        ],
        remainder="passthrough"
    )
    
    # Select base model
    if model_type == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "xgb":
        clf = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss")
    else:
        raise ValueError("Invalid model type")
    
    # Create the imbalanced-learn pipeline
    pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", clf)
    ])
    
    return pipeline

def train_stacking_ensemble(X, y, cat_features, num_features):
    print("Building Stacking Ensemble...")
    
    # Base learners
    rf_pipe = build_pipeline(cat_features, num_features, "rf")
    xgb_pipe = build_pipeline(cat_features, num_features, "xgb")
    
    # Stacking ensemble
    estimators = [
        ("rf", rf_pipe),
        ("xgb", xgb_pipe)
    ]
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight="balanced"),
        cv=5
    )
    
    # Evaluation via cross-validation
    print("Evaluating with 5-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        stacking_clf, X, y, 
        cv=skf, 
        scoring=["accuracy", "precision", "recall", "f1_macro", "roc_auc"]
    )
    
    metrics = {
        "accuracy": np.mean(cv_results["test_accuracy"]),
        "precision": np.mean(cv_results["test_precision"]),
        "recall": np.mean(cv_results["test_recall"]),
        "f1_macro": np.mean(cv_results["test_f1_macro"]),
        "roc_auc": np.mean(cv_results["test_roc_auc"])
    }
    
    print("\n--- Cross-Validation Metrics ---")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")
    
    # Final training on all data
    print("\nTraining final stacking ensemble on the full dataset...")
    stacking_clf.fit(X, y)
    
    return stacking_clf, metrics

def save_model(model, metrics, path="models/salary_model.joblib"):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, path)
    
    with open("data/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Model saved to {path}")
    print("Metrics saved to data/evaluation_metrics.json")

if __name__ == "__main__":
    X, y, cat_features, num_features = load_data("data/processed_jobs.csv")
    
    if len(y.unique()) < 2:
        print("Error: Target variable needs at least 2 classes (found 1). Check salary parsing in Phase 2.")
    else:
        # Debug: Split for a single classification report
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        model, metrics = train_stacking_ensemble(X_train, y_train, cat_features, num_features)
        
        print("\n--- Holdout Set Evaluation ---")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        save_model(model, metrics)
