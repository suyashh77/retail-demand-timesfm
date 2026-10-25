"""Recompute published metrics and expose variation across forecast starts.

These are descriptive checks, not hypothesis tests. Overlapping windows and a
single store do not support treating every target-horizon as independent evidence.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import evaluate, load_panel, simulate_inventory

ROOT=Path(__file__).resolve().parents[1]

def main():
    results=ROOT/'results'
    p=pd.read_csv(results/'forecasts.csv')
    observed,*_=load_panel(ROOT/'data/retail_panel.csv')
    scores,families=evaluate(p,observed)
    saved=pd.read_csv(results/'forecast_metrics.csv').sort_values('model').reset_index(drop=True)
    pd.testing.assert_frame_equal(scores.sort_values('model').reset_index(drop=True),saved,atol=1e-10,rtol=1e-10)
    inventory,daily,buffers=simulate_inventory(p,observed)
    pd.testing.assert_frame_equal(inventory.sort_values('model').reset_index(drop=True),
        pd.read_csv(results/'inventory_metrics.csv').sort_values('model').reset_index(drop=True),atol=1e-7,rtol=1e-10)
    test=p[p.split=='test'].copy()
    test['absolute_error']=abs(test.actual-test.prediction)
    per_start=test.groupby(['model','origin'])[['absolute_error','actual']].sum().reset_index()
    per_start['wape']=per_start.absolute_error/per_start.actual
    per_start.to_csv(results/'metrics_by_forecast_start.csv',index=False)
    paired=per_start.pivot(index='origin',columns='model',values='wape')
    paired['calendar_minus_boosting_percentage_points']=(paired['TimesFM + calendar']-paired['Gradient boosting'])*100
    paired.to_csv(results/'paired_comparison_by_start.csv')
    sample=test[(test.model=='TimesFM + calendar')&(test.origin=='2017-08-02')&(test.family=='BEVERAGES')]
    biggest=sample.iloc[abs(sample.actual-sample.prediction).argmax()]
    shares=test[test.model=='Seasonal naive'].groupby('family').actual.sum()
    shares=(shares/shares.sum()).to_dict()
    summary={
        'recomputed_forecast_scores_match':True,
        'recomputed_inventory_scores_match':True,
        'calendar_beats_boosting_starts':int((paired.calendar_minus_boosting_percentage_points<0).sum()),
        'total_test_starts':len(paired),
        'calendar_minus_boosting_gap_percentage_points_min':float(paired.calendar_minus_boosting_percentage_points.min()),
        'calendar_minus_boosting_gap_percentage_points_max':float(paired.calendar_minus_boosting_percentage_points.max()),
        'sales_share_in_primary_score':shares,
        'example_large_miss':{'model':biggest.model,'forecast_start':biggest.origin,'date':biggest.date,
            'family':biggest.family,'recorded_sales':float(biggest.actual),'forecast':float(biggest.prediction),
            'interval_lower':float(biggest.q10),'interval_upper':float(biggest.q90)},
        'interpretation':'Exploratory single-store comparison. No significance test, independence claim, or statistical equivalence claim.',
    }
    (results/'academic_audit.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
