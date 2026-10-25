"""Build a plain-language, offline report using the saved experiment outputs."""
import json
from pathlib import Path
import pandas as pd
from experiment import evaluate, FAMILIES, MODEL_NAMES, TEST_ORIGINS, load_panel
from charts import make_figures

ROOT=Path(__file__).resolve().parents[1]

def build():
    results=ROOT/'results'
    predictions=pd.read_csv(results/'forecasts.csv')
    metrics=pd.read_csv(results/'forecast_metrics.csv').sort_values('wape')
    inventory=pd.read_csv(results/'inventory_metrics.csv').sort_values('total_cost')
    families=pd.read_csv(results/'metrics_by_family.csv')
    observed,*_=load_panel(ROOT/'data/retail_panel.csv')
    nextweek,_=evaluate(predictions[predictions.horizon<=7],observed)
    nextweek.to_csv(results/'next_week_metrics.csv',index=False)
    best=metrics[metrics.model.str.startswith('TimesFM')].iloc[0]
    make_figures(metrics,inventory,predictions,families,observed,best.model)
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
        bestTimesfm=best.model,families=FAMILIES,models=MODEL_NAMES,
        origins=[str(d.date()) for d in TEST_ORIGINS],history=history,forecasts=forecasts)
    template=(ROOT/'src/dashboard_template.html').read_text(encoding='utf-8')
    rendered=template.replace('__DATA__',json.dumps(data,separators=(',',':')).replace('</',r'<\/'))
    (ROOT/'portfolio_report.html').write_text(rendered,encoding='utf-8')
    (ROOT/'CASE_STUDY.md').write_text((ROOT/'docs/PROJECT_STORY.md').read_text(encoding='utf-8'),encoding='utf-8')
    print('Built the plain-language dashboard and refreshed the result figures.')

if __name__=='__main__':build()
