"""
P(Deal Won) lead/opportunity scoring model.

Compares Logistic Regression, Random Forest, and XGBoost, all under a
STRICT TEMPORAL split (train on older deals, test on newer ones - never a
random shuffle split, which would leak future market conditions/patterns
into the past and overstate real-world performance). Probabilities are then
calibrated (Platt scaling for LR is native; isotonic/sigmoid calibration
applied to RF and XGBoost) so that "70% win probability" really does mean
~70% of such deals close, which the downstream revenue forecast depends on.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, average_precision_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from data_io import PROCESSED_DIR, ROOT  # noqa: E402

FIG_DIR = os.path.join(ROOT, "reports", "figures")
MODEL_DIR = os.path.join(ROOT, "models", "artifacts")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(PROCESSED_DIR, "scoring_dataset.csv.gz"), compression="gzip",
                  parse_dates=["created_date", "scoring_date", "actual_close_date"])

closed = df[df["is_closed"]].copy()
closed = closed.sort_values("scoring_date").reset_index(drop=True)

NUM_FEATURES = [
    "log_deal_value", "discount_requested_pct", "num_stakeholders", "employees",
    "engagement_score", "days_since_created_at_scoring", "stage_number_at_scoring",
    "days_in_current_stage_at_scoring", "num_activities_to_date", "num_calls",
    "num_emails", "num_meetings", "num_demos", "pct_positive_outcome",
    "pct_no_response", "days_since_last_activity", "activity_span_days",
]
CAT_FEATURES = ["source", "industry", "company_size_band", "rep_seniority", "stage_at_scoring"]
TARGET = "label_won"

# ----------------------------------------------------------------------------
# Strict temporal split: earliest 70% of scoring dates -> train,
# next 15% -> validation (for calibration), most recent 15% -> held-out test
# ----------------------------------------------------------------------------
n = len(closed)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = closed.iloc[:train_end]
val = closed.iloc[train_end:val_end]
test = closed.iloc[val_end:]

print(f"Temporal split — train: {len(train):,} ({train['scoring_date'].min().date()} to "
      f"{train['scoring_date'].max().date()}) | val: {len(val):,} | test: {len(test):,} "
      f"({test['scoring_date'].min().date()} to {test['scoring_date'].max().date()})")

X_train, y_train = train[NUM_FEATURES + CAT_FEATURES], train[TARGET]
X_val, y_val = val[NUM_FEATURES + CAT_FEATURES], val[TARGET]
X_test, y_test = test[NUM_FEATURES + CAT_FEATURES], test[TARGET]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
])

results = {}
fitted_models = {}

# ---- Logistic Regression -----------------------------------------------
lr_pipe = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=1000, C=1.0))])
lr_pipe.fit(X_train, y_train)
fitted_models["logistic_regression"] = lr_pipe

# ---- Random Forest (calibrated) -----------------------------------------
rf_pipe = Pipeline([("prep", preprocess),
                     ("clf", RandomForestClassifier(n_estimators=300, max_depth=10,
                                                     min_samples_leaf=25, n_jobs=-1,
                                                     random_state=42))])
rf_pipe.fit(X_train, y_train)
rf_calibrated = CalibratedClassifierCV(FrozenEstimator(rf_pipe), method="isotonic")
rf_calibrated.fit(X_val, y_val)
fitted_models["random_forest_calibrated"] = rf_calibrated

# ---- XGBoost (calibrated) -----------------------------------------------
xgb_pipe = Pipeline([("prep", preprocess),
                      ("clf", XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                                             subsample=0.85, colsample_bytree=0.8,
                                             eval_metric="logloss", random_state=42))])
xgb_pipe.fit(X_train, y_train)
xgb_calibrated = CalibratedClassifierCV(FrozenEstimator(xgb_pipe), method="isotonic")
xgb_calibrated.fit(X_val, y_val)
fitted_models["xgboost_calibrated"] = xgb_calibrated

# ----------------------------------------------------------------------------
# Evaluate all three on the held-out (most recent) test set
# ----------------------------------------------------------------------------
metrics_rows = []
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

for name, model in fitted_models.items():
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    ll = log_loss(y_test, proba)
    ap = average_precision_score(y_test, proba)
    metrics_rows.append({"model": name, "roc_auc": auc, "brier_score": brier,
                          "log_loss": ll, "avg_precision": ap})
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    plt.plot(mean_pred, frac_pos, marker="o", label=name)
    print(f"{name:28s} AUC={auc:.4f}  Brier={brier:.4f}  LogLoss={ll:.4f}  AvgPrec={ap:.4f}")

plt.xlabel("Mean predicted P(Win)")
plt.ylabel("Observed win rate")
plt.title("Calibration curve — held-out test set (most recent deals)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "calibration_curve.png"), dpi=140)
plt.close()

metrics_df = pd.DataFrame(metrics_rows).sort_values("roc_auc", ascending=False)
metrics_df.to_csv(os.path.join(ROOT, "reports", "lead_scoring_metrics.csv"), index=False)
print("\n" + metrics_df.to_string(index=False))

best_model_name = metrics_df.iloc[0]["model"]
best_model = fitted_models[best_model_name]
print(f"\nBest model by ROC-AUC on held-out test: {best_model_name}")

# ----------------------------------------------------------------------------
# Feature importance (from the underlying tree model of the best pipeline,
# or coefficients if logistic regression wins)
# ----------------------------------------------------------------------------
def get_feature_names(ct):
    return list(NUM_FEATURES) + list(ct.named_transformers_["cat"].get_feature_names_out(CAT_FEATURES))

try:
    base_pipe = best_model.calibrated_classifiers_[0].estimator if hasattr(best_model, "calibrated_classifiers_") else best_model
    if isinstance(base_pipe, FrozenEstimator) or hasattr(base_pipe, "estimator"):
        base_pipe = base_pipe.estimator
    inner_clf = base_pipe.named_steps["clf"]
    feat_names = get_feature_names(base_pipe.named_steps["prep"])
    if hasattr(inner_clf, "feature_importances_"):
        importances = inner_clf.feature_importances_
    else:
        importances = np.abs(inner_clf.coef_[0])
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}) \
        .sort_values("importance", ascending=False).head(20)
    imp_df.to_csv(os.path.join(ROOT, "reports", "feature_importance.csv"), index=False)

    plt.figure(figsize=(8, 7))
    plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
    plt.title(f"Top 20 feature importances — {best_model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=140)
    plt.close()
    print("\nTop 10 features:\n" + imp_df.head(10).to_string(index=False))
except Exception as e:
    print(f"(feature importance extraction skipped: {e})")

# ----------------------------------------------------------------------------
# Score the CURRENT open pipeline with the best model
# ----------------------------------------------------------------------------
open_opps = df[~df["is_closed"]].copy()
X_open = open_opps[NUM_FEATURES + CAT_FEATURES]
open_opps["calibrated_win_probability"] = best_model.predict_proba(X_open)[:, 1]
open_opps["model_used"] = best_model_name

open_scored_path = os.path.join(PROCESSED_DIR, "open_pipeline_scored.csv.gz")
open_opps.to_csv(open_scored_path, index=False, compression="gzip")
print(f"\nScored {len(open_opps):,} open opportunities -> {open_scored_path}")
print(f"Mean scored win probability on open pipeline: {open_opps['calibrated_win_probability'].mean():.3f}")

joblib.dump(best_model, os.path.join(MODEL_DIR, "best_lead_scoring_model.joblib"))
with open(os.path.join(MODEL_DIR, "model_config.json"), "w") as f:
    json.dump({"best_model": best_model_name, "num_features": NUM_FEATURES,
               "cat_features": CAT_FEATURES, "target": TARGET}, f, indent=2)
print("Saved best model artifact.")
