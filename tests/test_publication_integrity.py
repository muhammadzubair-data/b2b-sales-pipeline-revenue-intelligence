from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def test_model_metrics_match_readme():
    m = pd.read_csv(ROOT/'reports/lead_scoring_metrics.csv').set_index('model')
    assert round(m.loc['xgboost_calibrated','roc_auc'],3) == 0.738
    assert round(m.loc['xgboost_calibrated','avg_precision'],3) == 0.617
    assert m.loc['xgboost_calibrated','roc_auc'] > m.loc['random_forest_calibrated','roc_auc']

def test_capacity_math_and_labeling():
    c = pd.read_csv(ROOT/'reports/capacity_strategy_comparison.csv').set_index('strategy')
    naive = c.loc['naive_win_probability','total_expected_incremental_revenue_captured']
    opt = c.loc['expected_incremental_revenue','total_expected_incremental_revenue_captured']
    ratio = opt/naive
    assert 8.9 < ratio < 9.1
    text = (ROOT/'README.md').read_text().lower()
    assert '9.0x' in text
    assert 'not a causal uplift estimate' in text or 'not observed or causally identified' in text

def test_forecast_code_reconstructs_point_in_time_features():
    code = (ROOT/'models/revenue_forecasting.py').read_text()
    assert 'build_snapshot' in code
    assert 'activity_date"] < snapshot_date' in code
    assert 'entered_date"] < snapshot_date' in code
    readme = (ROOT/'README.md').read_text().lower()
    assert 'legacy forecast numbers' in readme

def test_model_config_has_no_outcome_fields():
    cfg=json.loads((ROOT/'models/artifacts/model_config.json').read_text())
    features=cfg['num_features']+cfg['cat_features']
    forbidden={'close_status','actual_close_date','label_won','won','lost'}
    assert not (set(features)&forbidden)

def test_required_portfolio_assets_exist():
    required=['README.md','DATA_DICTIONARY.md','reports/lead_scoring_metrics.csv',
              'reports/capacity_strategy_comparison.csv','power_bi/docs/dax_measures.md',
              'power_bi/docs/report_structure.md','models/lead_scoring.py',
              'models/capacity_optimization.py','models/revenue_forecasting.py']
    for f in required: assert (ROOT/f).exists(), f

def test_readme_dataset_counts():
    text=(ROOT/'README.md').read_text()
    for x in ['120,000','1,000,000','313,197','1,931,436','119,024','175,626']:
        assert x in text
