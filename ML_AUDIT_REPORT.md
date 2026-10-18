# Machine Learning Engineering Audit & Remediation Report

**Repository**: [Job-Market-Skill-Demand-Predictor](https://github.com/Bril3d/Job-Market-Skill-Demand-Predictor)  
**Date**: October 2026  
**Auditor**: Senior Machine Learning Engineer / Antigravity Agent  
**Status**: All Critical Leakages Resolved & Verified (14/14 Pytest Tests Passing)

---

## 1. Executive Summary & Audit Overview

A comprehensive audit was performed on the data pipeline, feature engineering, model training, MLflow tracking runs, and inference serving of the SkillDemand AI platform.

The audit revealed three critical methodological flaws in the previous implementation:
1. **Decision Threshold Snooping**: The decision threshold ($0.3626$) was optimized directly against the **test set labels** (`y_test`), artificially inflating reported High-Demand recall from **87.27% $\to$ 98.18%** and F1-score from **87.27% $\to$ 92.31%**.
2. **Global TF-IDF Vocabulary Leakage**: `TfidfVectorizer.fit_transform()` was executed globally across all scraped jobs before data splitting, leaking out-of-sample test vocabulary and document frequencies into the feature matrix.
3. **Circular Target Shortcut Features**: The binary target (`demand_label`) was synthetically generated as $\sum_{t \in \text{tags}} \text{count}(t) > \text{median}$. Input features such as `has_cloud`, `has_backend`, and `has_frontend` were extracted from the exact same tag strings, with `has_cloud` having a **0.00% occurrence in Class 0**, allowing the model to trivially reconstruct the target rule rather than learning market signals.

### Remediation Summary
* Built an end-to-end `ColumnTransformer` that confines TF-IDF vocabulary fitting and OneHot encodings strictly to training folds.
* Preserved `SMOTE` oversampling within `imblearn.pipeline.Pipeline`, ensuring test and validation sets remain untouched.
* Completely removed circular tag shortcut features from the modeling feature matrix.
* Implemented out-of-fold training 5-fold cross-validation threshold tuning (`get_best_threshold_from_oof`), ensuring the holdout test set is evaluated only once on unseen data.
* Executed a fair benchmark of **Dummy Baselines, Logistic Regression, Random Forest, XGBoost, and Stacking Classifier**, demonstrating that regularized **Logistic Regression ($C=2.0$)** achieves the highest generalization performance (**0.9373 ROC-AUC** vs 0.8865 for Stacking).
* Added 6 automated regression and leakage tests in `tests/test_leakage_and_evaluation.py`. All 14 tests in the repository pass.

---

## 2. Summary of Code Changes & Files Modified

| File Path | Component | Changes Made |
| :--- | :--- | :--- |
| [`code/preprocessing.py`](code/preprocessing.py) | Data Cleaning & Preprocessing | Removed global `TfidfVectorizer.fit_transform()`. Eliminated circular tag shortcut features (`has_cloud`, `has_backend`, `has_frontend`, `has_ai`, `num_skills`). Enforced title deduplication to produce 520 clean, unique job postings. |
| [`code/modeling.py`](code/modeling.py) | Training & Evaluation Pipeline | Built an end-to-end `ColumnTransformer` (TF-IDF on `title`, OneHot on categorical features, passthrough on experience). Enforced SMOTE inside `ImbPipeline`. Tuned threshold ($0.5181$) strictly on 5-fold out-of-fold training CV. Executed fair multi-model comparison. Logged all runs and model artifacts to MLflow (`sqlite:///mlflow.db`). |
| [`code/predict.py`](code/predict.py) | Standalone Inference | Converted input handling to pass structured DataFrames directly into the serialized pipeline artifact (`models/demand_model.joblib`), eliminating manual matrix stitching. |
| [`code/app.py`](code/app.py) | FastAPI Backend Engine | Aligned `run_prediction` with the clean feature schema (`title`, `seniority`, `category`, `geo_tier`, `years_exp`). Retained backward compatibility for request payloads. |
| [`tests/test_leakage_and_evaluation.py`](tests/test_leakage_and_evaluation.py) | Automated Test Suite | Created 6 automated unit tests validating: zero leaky columns, TF-IDF vocabulary isolation, SMOTE inference invariance, train-only threshold selection, deterministic splitting, and inference consistency. |
| [`data/evaluation_metrics.json`](data/evaluation_metrics.json) | Metrics Artifact | Replaced invalid metrics with genuine, verified holdout test metrics. |
| [`README.md`](README.md) | Documentation | Updated benchmark comparisons, removed the snooped 98.18% recall claim, and detailed the architectural justification for the linear model. |
| [`.gitignore`](.gitignore) | Version Control Hygiene | Added patterns for temporary scratch scripts, dumps, and test artifacts. |

---

## 3. Dataset Pipeline & Class Distributions

| Pipeline Stage | Rows | Columns | Class 0 (Low Demand) | Class 1 (High Demand) | % High Demand | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Raw Scraped Data** | 864 | 8 | N/A | N/A | N/A | Initial RemoteOK postings |
| **Cleaned Intermediate** | 752 | 12 | N/A | N/A | N/A | Deduplicated on `title, company, tags` (112 dropped) |
| **Clean Feature Matrix** | 520 | 6 | 243 | 277 | 53.27% | Deduplicated by `title` to prevent train-test contamination |
| **Training Partition (80%)** | 416 | 5 | 194 (46.63%) | 222 (53.37%) | 53.37% | Stratified split (`random_state=42`) |
| **Holdout Test Partition (20%)** | 104 | 5 | 49 (47.12%) | 55 (52.88%) | 52.88% | Untouched test partition |

---

## 4. Before / After Benchmark Evaluation

All models in the remediated pipeline were evaluated under identical conditions on the **untouched holdout test set** ($N = 104$: 55 High Demand, 49 Low Demand). The decision threshold was chosen strictly using 5-fold cross-validation on `X_train`:

| Model Architecture | Threshold Strategy | Threshold | Test Accuracy | Precision (High) | Recall (High) | F1-Score (High) | Macro F1 | ROC AUC | Confusion Matrix `[TN, FP, FN, TP]` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BEFORE: Stacking (Claimed in README)** | **Test Set Snooped** | **0.3627** | **91.35%** | **87.10%** | **98.18%** | **92.31%** | **91.21%** | **0.9625** | `[41, 8, 1, 54]` |
| *BEFORE: Stacking (Default Thresh)* | *Default* | *0.5000* | *86.54%* | *87.27%* | *87.27%* | *87.27%* | *86.49%* | *0.9625* | `[42, 7, 7, 48]` |
| | | | | | | | | |
| **AFTER (LEAK-FREE PROTOCOL)**: | | | | | | | | |
| **Dummy (Most Frequent)** | Baseline | 1.0000 | 52.88% | 52.88% | 100.00% | 69.18% | 34.59% | 0.5000 | `[0, 49, 0, 55]` |
| **Dummy (Stratified)** | Baseline | Random | 52.88% | 52.88% | 100.00% | 69.18% | 34.59% | 0.4666 | `[0, 49, 0, 55]` |
| **Random Forest (Tuned + SMOTE)** | Train CV OOF | 0.4597 | 77.88% | 83.33% | 72.73% | 77.67% | 77.72% | 0.8879 | `[41, 8, 15, 40]` |
| **XGBoost (Tuned + SMOTE)** | Train CV OOF | 0.4157 | 81.73% | 80.00% | 87.27% | 83.48% | 81.60% | 0.8785 | `[37, 12, 7, 48]` |
| **Stacking (RF + XGB + SMOTE)** | Train CV OOF | 0.4696 | 79.81% | 82.69% | 78.18% | 80.37% | 79.76% | 0.8865 | `[40, 9, 12, 43]` |
| **Logistic Regression (Selected Best)** | **Train CV OOF** | **0.5181** | **84.62%** | **88.24%** | **81.82%** | **84.91%** | **84.61%** | **0.9373** | `[43, 6, 10, 45]` |

---

## 5. Architectural Insight: Why Linear Outperforms Ensembles Here

The fair comparison demonstrated that **Regularized Logistic Regression ($C=2.0$) decisively outperformed Random Forest, XGBoost, and Stacking Classifier across Accuracy, Precision, F1-Score, and ROC-AUC**.

### Root Cause Analysis:
1. **Feature Space Geometry**: The title text is encoded as a 200-dimensional sparse TF-IDF feature vector. Linear models find maximum-margin separating hyperplanes exceptionally well in sparse spaces. Decision trees, in contrast, partition data with axis-aligned splits, which causes them to overfit sparse individual token indicators on small sample sizes.
2. **Sample Size ($N = 416$)**: Complex ensembles like Stacking Classifier incur high parameter overhead. On small datasets, stacking adds variance without providing non-linear gains over text representations.
3. **Occam's Razor**: Selecting Logistic Regression provides superior out-of-sample generalization (**0.9373 ROC-AUC**), lower latency ($<1\text{ ms}$ per inference), and direct model interpretability.

### 95% Wilson Score Confidence Intervals ($N=104$):
* **Accuracy (88/104)**: $84.62\%$ ($95\%\text{ CI: } [76.5\%, 90.3\%]$)
* **Precision (45/51)**: $88.24\%$ ($95\%\text{ CI: } [76.7\%, 94.5\%]$)
* **Recall (45/55)**: $81.82\%$ ($95\%\text{ CI: } [69.7\%, 89.8\%]$)

---

## 6. Target Definition Limitations & Defensible Alternatives

### 6.1 Limitations in Current Definition
The target `demand_label` is generated by:
1. Summing the corpus-wide frequencies of all tags present on a job posting (`demand_score`).
2. Splitting by the median score.

This introduces:
* **Tag Volume Bias**: Postings with 6 or 7 tags received high demand scores purely due to tag quantity ($\rho = 0.915$), while niche or emerging roles with a single tag (e.g., `AI Engineer`) were labeled Low Demand (`y=0`).
* **Heuristic Artifacts**: The target reflects posting tag frequency on RemoteOK rather than true economic demand (e.g., hiring velocity or compensation premiums).

### 6.2 Defensible Alternatives for Future Iterations
1. **Salary Premium Classification**: Predict whether a job posting offers top-quartile compensation within its category.
2. **Posting Velocity / Application Competition**: Predict days-active on board or application count.
3. **External Labor Indices**: Calibrate skill demand against external benchmarks such as BLS Occupational Outlook or Stack Overflow annual developer surveys.

---

## 7. Verification & Automated Test Results

The test suite was executed via pytest:
```bash
.venv\Scripts\python -m pytest tests/ -v
```

### Test Output:
```text
tests/test_api.py::test_health_endpoint PASSED                           [  7%]
tests/test_api.py::test_metrics_endpoint PASSED                          [ 14%]
tests/test_api.py::test_insights_endpoint PASSED                         [ 21%]
tests/test_api.py::TestPredictionEndpoint::test_single_prediction_high_demand PASSED [ 28%]
tests/test_api.py::TestPredictionEndpoint::test_single_prediction_low_demand PASSED [ 35%]
tests/test_api.py::TestPredictionEndpoint::test_batch_prediction PASSED  [ 42%]
tests/test_api.py::TestPredictionEndpoint::test_invalid_api_key PASSED   [ 50%]
tests/test_api.py::test_history_endpoint PASSED                          [ 57%]
tests/test_leakage_and_evaluation.py::test_feature_matrix_no_leaky_columns PASSED [ 64%]
tests/test_leakage_and_evaluation.py::test_tfidf_pipeline_isolation PASSED [ 71%]
tests/test_leakage_and_evaluation.py::test_smote_isolation_during_inference PASSED [ 78%]
tests/test_leakage_and_evaluation.py::test_threshold_selection_training_only PASSED [ 85%]
tests/test_leakage_and_evaluation.py::test_reproducibility_data_split PASSED [ 92%]
tests/test_leakage_and_evaluation.py::test_inference_consistency PASSED  [100%]

====================== 14 passed in 5.22s ======================
```

---

## 8. Exact Commands to Reproduce

```bash
# 1. Clean data and generate leak-free feature matrix
python code/preprocessing.py

# 2. Run 5-fold CV benchmark, threshold selection, and MLflow tracking
python code/modeling.py

# 3. Test standalone inference
python code/predict.py

# 4. Run automated test suite
python -m pytest tests/ -v

# 5. Launch MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# 6. Launch FastAPI backend
uvicorn code.app:app --host 127.0.0.1 --port 8001
```
