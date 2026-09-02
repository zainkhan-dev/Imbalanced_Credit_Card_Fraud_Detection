# 💳 Credit Card Fraud Detection — Domain-Specific Imbalanced Classifier

A fraud detection pipeline built on the Kaggle Credit Card Fraud dataset, focused on properly
handling severe class imbalance (0.17% fraud), tracking experiments with MLflow, and deploying
a working Streamlit app. This README also documents a real threshold-calibration bug discovered
during evaluation — included deliberately, because diagnosing it was as valuable as the model itself.

## 1. Problem & Dataset

- **Dataset:** Kaggle Credit Card Fraud (`creditcardfraud`), 284,807 transactions, 30 features
  (`Time`, `V1`–`V28` PCA-transformed, `Amount`).
- **Class balance:** 284,315 legitimate (99.83%) vs. 492 fraud (0.17%) — a severe imbalance.
- **Why accuracy is the wrong metric:** a model predicting "not fraud" for every transaction
  scores 99.83% accuracy while catching zero fraud. This project uses **F1 (fraud class)**,
  **AUC-ROC**, and — more importantly — **AUC-PR (average precision)** as primary metrics,
  since AUC-PR is far less inflated by the large negative class.
- **Split strategy:** chronological (ordered by `Time`) rather than random, to avoid leaking
  future transaction patterns into training and to mirror real-world deployment (train on past,
  predict on future). Note: the dataset spans only 48 hours, so this is primarily a methodological
  best practice rather than a defense against major temporal drift.

## 2. Approach

Three logistic regression variants were trained and compared:

| Variant | Technique |
|---|---|
| Baseline | No imbalance handling |
| Class-Weighted | `class_weight='balanced'` |
| SMOTE | Synthetic minority oversampling (applied strictly to the training split only, after the train/test split, to avoid leakage) |

`Amount` and `Time` were each scaled independently with their own `StandardScaler` (fit only on
training data). `V1`–`V28` are already PCA-transformed/standardized and left as-is.

## 3. Key Finding: Threshold Saturation in Resampled Models

The most important result of this project wasn't which model "won" — it was discovering that
naive threshold optimization is actively misleading for resampled/reweighted models.

When searching for each model's F1-optimal decision threshold, class-weighted and SMOTE logistic
regression both converged on thresholds of **0.9999–1.0000** — i.e., only flagging a transaction
as fraud when the model was almost perfectly certain.

A full threshold sweep confirmed this wasn't a fluke:

| Threshold | Class-Weighted F1 | SMOTE F1 |
|---|---|---|
| 0.50 | 0.131 | 0.127 |
| 0.80 | 0.381 | 0.329 |
| 0.90 | 0.502 | 0.472 |
| 0.95 | 0.555 | 0.556 |
| 0.98 | 0.629 | 0.600 |
| 0.999 | 0.727 | 0.731 |
| 0.9999 | **0.806** | **0.762** |

F1 increases **monotonically** all the way to the saturation point, with no interior local
maximum — meaning there is no genuine "best" threshold to discover below it. This is a known
pathology: SMOTE oversampling (and heavy class weighting) pushes the training classes toward
near-linear separability, which causes an unregularized logistic regression's coefficients — and
therefore its predicted probabilities — to blow up toward 0/1. The resulting "optimal" threshold
is a statistical artifact backed by very few predictions (as few as ~60 flagged transactions out
of 56,962 test rows), not a reliable operating point.

**Conclusion:** AUC-PR (threshold-independent) was adopted as the primary comparison metric.
F1/precision/recall are additionally reported at one fixed, shared threshold (T=0.5) across all
three models for illustrative — not optimality — purposes.

## 4. Results

**Primary comparison (threshold-independent):**

| Model | AUC-PR | AUC-ROC |
|---|---|---|
| Baseline | 0.7402 | 0.9755 |
| Class-Weighted | 0.7620 | 0.9863 |
| SMOTE | 0.7682 | 0.9852 |

AUC-PR differences (~0.02–0.03) are small relative to the test set's fraud count (only 75
positive cases) and were not treated as a decisive ranking — no confidence intervals were
computed, and a ~3-point gap on 75 positives is well within plausible sampling noise.

**Illustrative fixed-threshold comparison (T = 0.50):**

| Model | Precision | Recall | F1 | Flagged |
|---|---|---|---|---|
| Baseline | 0.933 | 0.560 | 0.700 | 45 |
| Class-Weighted | 0.071 | 0.893 | 0.131 | 945 |
| SMOTE | 0.068 | 0.893 | 0.127 | 979 |

**Baseline's own natural F1-optimal threshold (T = 0.17):**

- Precision: 0.773, Recall: 0.680, **F1: 0.7234**, AUC-PR: 0.7402, support: 66 flagged

## 5. Model Selection

**Deployed model: Baseline Logistic Regression, decision threshold T = 0.17.**

Justification: AUC-PR is comparable across all three variants and not reliably distinguishable
given the small fraud sample size. Baseline is the only variant with a genuine, well-supported F1
optimum and smoothly calibrated probabilities across its full range — class-weighted and SMOTE
both exhibit unresolved probability saturation that makes any of their thresholds either
statistically fragile (near 1.0) or effectively arbitrary (if capped lower, F1 is still climbing
at the cap, so the cap becomes the answer rather than a discovered optimum). Baseline is therefore
the only variant considered safely deployable.

**Limitation:** at T=0.17, baseline still misses ~32% of fraud cases (recall 0.68–0.80 depending
on threshold). Future work — see Section 8 — should explore non-linear models less prone to this
class of calibration failure.

## 6. Experiment Tracking (MLflow)

All three variants are logged as separate MLflow runs under experiment
`Credit_Card_Fraud_Detection`, each with:

- **Params:** model type, `class_weight`, `smote_applied`, evaluation strategy, fixed threshold, `C`, `random_state`
- **Metrics:** `auc_pr`, `auc_roc`, precision/recall/F1 at the fixed shared threshold
- **Artifacts:** precision-recall curve (PNG), serialized model
- **Tags:** a `note` field documenting the saturation finding on the class-weighted and SMOTE
  runs, and `primary_comparison_metric: auc_pr` on all runs

Run locally:
```bash
mlflow ui
```
then open `http://127.0.0.1:5000` to compare runs side by side.

## 7. Deployment (Streamlit)

`app.py` provides two ways to interact with the deployed baseline model:

- **Single Sample Demo:** score one of 10 preloaded real test-set transactions (5 fraud, 5
  legitimate), with an adjustable decision threshold slider.
- **CSV Batch Scoring:** upload a CSV matching the expected schema; get per-row fraud
  probabilities, flags, and a downloadable scored CSV. Uploads missing required columns are
  rejected with a clear error message rather than failing silently.

The model and its two feature scalers (`Amount`, `Time` — fit and persisted separately) are
exported once from the MLflow-logged run via `joblib`, so the deployed app has no runtime
dependency on the MLflow tracking store itself.

Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 8. Limitations & Future Work

- Recall tops out around 68–80% depending on threshold — a meaningful share of fraud is still
  missed at any deployable operating point for this model family.
- Logistic regression's probability saturation under resampling is a known limitation of linear
  models on near-separable data; tree-based ensembles (XGBoost, LightGBM) with proper resampling
  are a natural next step, as they don't share this exact failure mode.
- No confidence intervals were computed for AUC-PR given the small positive class (75 test
  cases) — bootstrapped CIs would sharpen future model comparisons.
- The Streamlit app currently ships with a fixed set of 10 sample transactions; a larger,
  randomized sample pool would improve demo variety.

## 9. Project Structure
.
├── app.py                              # Streamlit application
├── baseline_logistic_regression.joblib # Exported deployed model
├── amount_scaler.joblib                # Fitted StandardScaler for Amount
├── time_scaler.joblib                  # Fitted StandardScaler for Time
├── sample_transactions.csv             # 10 real test-set rows for the demo tab
├── requirements.txt
└── README.md

## 10. Setup
git clone https://github.com/zainkhan-dev/Imbalanced_Credit_Card_Fraud_Detection.git
cd Imbalanced_Credit_Card_Fraud_Detection
pip install -r requirements.txt
streamlit run app.py