"""Build a plain-language, offline report using the saved experiment outputs."""
import json
import argparse
from pathlib import Path
import pandas as pd
from experiment import evaluate, FAMILIES, MODEL_NAMES, TEST_ORIGINS, load_panel
from charts import make_figures

ROOT=Path(__file__).resolve().parents[1]

def build(initial=False):
    results=ROOT/'results' if initial else ROOT/'results/matched_inputs'
    predictions=pd.read_csv(results/'forecasts.csv')
    metrics=pd.read_csv(results/'forecast_metrics.csv').sort_values('wape')
    inventory=pd.read_csv(results/'inventory_metrics.csv').sort_values('total_cost')
    families=pd.read_csv(results/'metrics_by_family.csv')
    observed,*_=load_panel(ROOT/'data/retail_panel.csv')
    nextweek,_=evaluate(predictions[predictions.horizon<=7],observed)
    nextweek.to_csv(results/'next_week_metrics.csv',index=False)
    best=metrics[metrics.model.str.startswith('TimesFM')].iloc[0]
    if initial:
        make_figures(metrics,inventory,predictions,families,observed,best.model)
    models = MODEL_NAMES if initial else ['Seasonal naive', 'Gradient boosting: sales',
        'TimesFM multivariate', 'Gradient boosting: calendar', 'TimesFM + calendar',
        'Gradient boosting: all inputs', 'TimesFM + all inputs']
    history=[]
    for family in FAMILIES:
        for date,value in observed.loc['2017-05-01':,family].items():
            if pd.notna(value):
                history.append(dict(f=family,d=str(date.date()),y=round(float(value),3)))
    forecasts=[dict(m=r.model,f=r.family,o=r.origin,d=r.date,h=int(r.horizon),a=round(r.actual,3),
                    p=round(r.prediction,3),lo=round(r.q10,3),hi=round(r.q90,3))
               for r in predictions[predictions.split=='test'].itertuples()]
    data=dict(metrics=metrics.to_dict('records'),nextweek=nextweek.to_dict('records'),
        inventory=inventory.to_dict('records'),byFamily=families.to_dict('records'),
        bestTimesfm=best.model,families=FAMILIES,models=models,
        origins=[str(d.date()) for d in TEST_ORIGINS],history=history,forecasts=forecasts)
    template=(ROOT/('src/initial_dashboard_template.html' if initial else 'src/dashboard_template.html')).read_text(encoding='utf-8')
    if not initial:
        data['comparisonBoosting'] = 'Gradient boosting: calendar'
        cal_tf = metrics[metrics.model=='TimesFM + calendar'].iloc[0]
        cal_gb = metrics[metrics.model==data['comparisonBoosting']].iloc[0]
        naive = metrics[metrics.model=='Seasonal naive'].iloc[0]
        longer = pd.read_csv(results/'history_sensitivity/forecast_metrics.csv').iloc[0]
        tokens = {'CAL_TF_WAPE':f'{100*cal_tf.wape:.2f}%', 'CAL_GB_WAPE':f'{100*cal_gb.wape:.2f}%',
            'NAIVE_WAPE':f'{100*naive.wape:.2f}%', 'CAL_TF_COVERAGE':f'{100*cal_tf.coverage80:.1f}%',
            'CAL_GB_COVERAGE':f'{100*cal_gb.coverage80:.1f}%',
            'CAL_TF_PINBALL':f'{cal_tf.pinball:.2f}', 'CAL_GB_PINBALL':f'{cal_gb.pinball:.2f}',
            'CAL_TF_WIDTH':f'{cal_tf.mean_interval_width:.0f}', 'CAL_GB_WIDTH':f'{cal_gb.mean_interval_width:.0f}',
            'CAL_GB_365_WAPE':f'{100*longer.wape:.2f}%'}
        for name, model in [('NEXT_CAL_TF','TimesFM + calendar'), ('NEXT_CAL_GB',data['comparisonBoosting'])]:
            tokens[name] = f'{100*nextweek[nextweek.model==model].wape.iloc[0]:.2f}%'
        for name, value in tokens.items():
            template = template.replace('__'+name+'__', value)
    rendered=template.replace('__DATA__',json.dumps(data,separators=(',',':')).replace('</',r'<\/'))
    destination = ROOT/'archive/initial-report.html' if initial else ROOT/'report.html'
    destination.parent.mkdir(exist_ok=True)
    if initial:
        rendered=rendered.replace('href="matched_report.html"','href="../matched_report.html"')
    destination.write_text(rendered,encoding='utf-8')
    print(f'Built {destination.relative_to(ROOT)} from {results.relative_to(ROOT)}.')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial',action='store_true',help='Rebuild the archived unequal-input report')
    build(parser.parse_args().initial)
