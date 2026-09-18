"""Runs the entire project pipeline end-to-end, in order. Takes a few
minutes on a laptop; the synthetic data generation step is the slowest
(vectorized, but still ~300K+ opportunities and ~2M activities).
"""
import subprocess
import sys
import os

ROOT = os.path.dirname(__file__)
STEPS = [
    ("Generating synthetic CRM dataset", "scripts/generate_data.py"),
    ("Building SQLite DB for the SQL analytics layer", "sql/build_db.py"),
    ("Running SQL analytics queries", "sql/run_queries.py"),
    ("Building point-in-time scoring features", "features/build_features.py"),
    ("Training & evaluating lead scoring models", "models/lead_scoring.py"),
    ("Backtesting revenue forecasting", "models/revenue_forecasting.py"),
    ("Running sales-cycle survival analysis", "models/survival_analysis.py"),
    ("Optimizing sales capacity allocation", "models/capacity_optimization.py"),
    ("Generating Next Best Action worklist", "models/next_best_action.py"),
    ("Building Power BI export layer", "power_bi/build_exports.py"),
]

for label, script in STEPS:
    print(f"\n{'='*80}\n{label}  ({script})\n{'='*80}")
    result = subprocess.run([sys.executable, os.path.join(ROOT, script)])
    if result.returncode != 0:
        print(f"\nStep failed: {script}. Stopping.")
        sys.exit(1)

print("\nAll steps completed successfully.")
