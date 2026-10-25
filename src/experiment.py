"""Leakage-aware rolling forecasting experiment and periodic-review inventory replay.

Run with official TimesFM 3.0 installed, or pass --timesfm-source / --extra-packages.
All quantitative results are computed from downloaded observations and model outputs.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ['BEVERAGES', 'BREAD/BAKERY', 'DAIRY', 'GROCERY I']
CALENDAR = ['dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'holiday', 'payday']
Q = np.arange(1, 10) / 10
HORIZON, CONTEXT, REVIEW, LEAD = 14, 256, 7, 7
CAL_ORIGINS = pd.date_range('2017-02-15', periods=12, freq='7D')
TEST_ORIGINS = pd.date_range('2017-05-17', periods=12, freq='7D')
MODEL_NAMES = ['Seasonal naive', 'Gradient boosting', 'TimesFM univariate',
               'TimesFM multivariate', 'TimesFM + calendar', 'TimesFM + all inputs']


def causal_week_fill(frame):
    """Missing context observations use ONLY a previous same-weekday observation."""
    out = frame.copy()
    for i in range(7, len(out)):
        out.iloc[i] = out.iloc[i].fillna(out.iloc[i - 7])
    return out


def load_panel(path):
    panel = pd.read_csv(path, parse_dates=['date'])
    observed = panel.pivot(index='date', columns='family', values='sales')[FAMILIES]
    y = causal_week_fill(observed)
    promo = causal_week_fill(panel.pivot(index='date', columns='family', values='onpromotion')[FAMILIES])
    daily = panel.drop_duplicates('date').set_index('date').sort_index()
    traffic = daily.transactions.ffill()  # historical inputs only; no backfilling
    calendar = daily[CALENDAR]
    return observed, y, promo, traffic, calendar


def make_inputs(origin, y, promo, traffic, calendar):
    i = y.index.get_loc(origin)
    context = y.iloc[i-CONTEXT:i].to_numpy(dtype=np.float32).T
    past_only = traffic.iloc[i-CONTEXT:i].to_numpy(dtype=np.float32)[None, :]
    cal = calendar.iloc[i-CONTEXT:i+HORIZON].to_numpy(dtype=np.float32).T
    promotions = promo.iloc[i-CONTEXT:i+HORIZON].to_numpy(dtype=np.float32).T
    assert context.shape == (len(FAMILIES), CONTEXT)
    assert past_only.shape == (1, CONTEXT)
    assert cal.shape[1] == CONTEXT + HORIZON
    assert np.isfinite(context).all() and np.isfinite(past_only).all()
    assert np.isfinite(cal).all() and np.isfinite(promotions).all()
    return context, past_only, cal, np.concatenate([cal, promotions])


def direct_features(i, h, y, promo, traffic, calendar):
    """Origin i is the first unknown date; history always ends at i-1."""
    history = y[i-28:i]
    seasonal = history[-7 + h % 7]
    weekday_mean = np.mean(history[-7 + h % 7::-7])
    return [seasonal, weekday_mean, history[-1], history[-7:].mean(), history.mean(),
            history[-7:].std(), history[-7:].mean() - history[-14:-7].mean(),
            traffic[i-7:i].mean(), traffic[i-1], h + 1,
            promo[i+h], *calendar[i+h]]


def boosting_predictions(origins, observed, y, promo, traffic, calendar):
    """Train per-family direct models on the preceding year at every origin.

    Every training label is dated before the current origin. Future traffic is
    absent from features. Future promo counts carry the stated planned-input assumption.
    """
    predictions = {}
    dates = y.index
    ta = traffic.to_numpy()
    ca = calendar.to_numpy()
    first_i = dates.get_loc(origins.min())
    last_i = dates.get_loc(origins.max())
    for f, family in enumerate(FAMILIES):
        ya, pa, truth = y[family].to_numpy(), promo[family].to_numpy(), observed[family].to_numpy()
        features, labels, target_indices = [], [], []
        # Two years of data ensure a full year of labeled examples at first calibration.
        for i in range(max(28, first_i-380), last_i):
            for h in range(HORIZON):
                j = i+h
                if j < len(ya) and np.isfinite(truth[j]):
                    x = direct_features(i, h, ya, pa, ta, ca)
                    if np.isfinite(x).all():
                        features.append(x)
                        labels.append(truth[j])
                        target_indices.append(j)
        features, labels, target_indices = np.array(features), np.array(labels), np.array(target_indices)
        for origin in origins:
            i = dates.get_loc(origin)
            valid = (target_indices < i) & (target_indices >= i-365)
            model = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=15,
                learning_rate=.07, l2_regularization=10, random_state=42)
            model.fit(features[valid], labels[valid])
            pred = np.maximum(0, model.predict(np.array([direct_features(i,h,ya,pa,ta,ca) for h in range(HORIZON)])))
            predictions[(origin, family)] = pred
        print(f'Gradient boosting complete: {family}', flush=True)
    return predictions


def forecast_rows(model, origin, split, points, quantiles, observed):
    dates = pd.date_range(origin, periods=HORIZON)
    rows = []
    for f, family in enumerate(FAMILIES):
        actual = observed.loc[dates, family].to_numpy()
        assert np.isfinite(actual).all()
        for h, date in enumerate(dates):
            row = dict(model=model, origin=str(origin.date()), date=str(date.date()),
                       split=split, family=family, horizon=h+1,
                       actual=float(actual[h]), prediction=float(points[f,h]))
            row.update({f'q{int(q*100):02d}': float(quantiles[f,h,k]) for k,q in enumerate(Q)})
            rows.append(row)
    return rows


def evaluate(predictions, observed):
    test = predictions[predictions.split == 'test']
    history = observed.loc[observed.index < TEST_ORIGINS.min()]
    scale = (history - history.shift(7)).abs().mean()
    rows = []
    for (model, family), group in test.groupby(['model', 'family']):
        error = group.prediction.to_numpy() - group.actual.to_numpy()
        actual = group.actual.to_numpy()
        quantiles = group[[f'q{int(q*100):02d}' for q in Q]].to_numpy()
        residual = actual[:,None] - quantiles
        rows.append(dict(model=model, family=family, n=len(group), mae=float(abs(error).mean()),
                         wape=float(abs(error).sum()/actual.sum()), mase=float(abs(error).mean()/scale[family]),
                         bias=float(error.sum()/actual.sum()),
                         coverage80=float(((actual >= quantiles[:,0]) & (actual <= quantiles[:,-1])).mean()),
                         pinball=float(np.maximum(Q*residual, (Q-1)*residual).mean()),
                         mean_interval_width=float((quantiles[:,-1]-quantiles[:,0]).mean())))
    per_family = pd.DataFrame(rows)
    overall = []
    for model, group in test.groupby('model'):
        fam = per_family[per_family.model == model]
        actual = group.actual.to_numpy()
        error = group.prediction.to_numpy() - actual
        quantiles = group[[f'q{int(q*100):02d}' for q in Q]].to_numpy()
        residual = actual[:,None]-quantiles
        overall.append(dict(model=model, n=len(group), mae=float(abs(error).mean()),
            wape=float(abs(error).sum()/actual.sum()), macro_mase=float(fam.mase.mean()),
            bias=float(error.sum()/actual.sum()),
            coverage80=float(((actual >= quantiles[:,0]) & (actual <= quantiles[:,-1])).mean()),
            pinball=float(np.maximum(Q*residual,(Q-1)*residual).mean()),
            mean_interval_width=float((quantiles[:,-1]-quantiles[:,0]).mean())))
    return pd.DataFrame(overall).sort_values('wape'), per_family


def aggregate_buffers(predictions):
    """Estimate signed aggregate errors directly; never add marginal quantiles."""
    cal = predictions[predictions.split == 'calibration']
    totals = cal.groupby(['model','family','origin'])[['actual','prediction']].sum()
    totals['aggregate_error'] = totals.actual - totals.prediction
    records = []
    for (model, family), group in totals.groupby(['model','family']):
        records.append(dict(model=model, family=family, calibration_origins=len(group),
            buffer=float(max(0,np.quantile(group.aggregate_error,.9,method='higher')))))
    return pd.DataFrame(records)


def simulate_inventory(predictions, observed, holding=.01, shortage=1.):
    """Daily lost-sales replay; receive, review/order, demand, then holding cost.

    Weekly order-up-to policy, deterministic 7-day lead time, no capacity limit.
    Initial on-hand is shared across models and computed from past seasonal sales.
    Observed sales are used as a demand proxy, not claimed recovered latent demand.
    """
    buffers = aggregate_buffers(predictions)
    start, end = TEST_ORIGINS.min(), TEST_ORIGINS.max()+pd.Timedelta(days=6)
    days = pd.date_range(start, end)
    rows = []
    for model in MODEL_NAMES:
        for family in FAMILIES:
            initial = float(observed.loc[start-pd.Timedelta(days=7):start-pd.Timedelta(days=1),family].sum()*2)
            stock, pipeline = initial, []
            buffer = float(buffers[(buffers.model==model)&(buffers.family==family)].buffer.iloc[0])
            for day in days:
                arrival = sum(qty for due,qty in pipeline if due==day)
                stock += arrival
                pipeline = [(due,qty) for due,qty in pipeline if due>day]
                ordered = 0.
                if day in TEST_ORIGINS:
                    forecast = predictions[(predictions.model==model)&(predictions.family==family)&(predictions.origin==str(day.date()))]
                    assert len(forecast)==HORIZON
                    target = float(forecast.prediction.sum()+buffer)
                    ordered = max(0.,target-stock-sum(qty for _,qty in pipeline))
                    pipeline.append((day+pd.Timedelta(days=LEAD), ordered))
                demand = float(observed.loc[day,family])
                served = min(stock,demand)
                lost = demand-served
                stock -= served
                assert stock>=-1e-8
                rows.append(dict(model=model,family=family,date=str(day.date()),
                    demand=demand,served=served,lost=lost,ending_inventory=stock,
                    arrival=arrival,ordered=ordered,holding_cost=holding*stock,
                    shortage_cost=shortage*lost,total_cost=holding*stock+shortage*lost))
    daily = pd.DataFrame(rows)
    summary = daily.groupby('model').agg(demand=('demand','sum'),lost_sales=('lost','sum'),
        average_inventory=('ending_inventory','mean'),holding_cost=('holding_cost','sum'),
        shortage_cost=('shortage_cost','sum'),total_cost=('total_cost','sum')).reset_index()
    summary['fill_rate']=1-summary.lost_sales/summary.demand
    summary['stockout_family_day_rate']=summary.model.map(daily.assign(stockout=daily.lost>1e-8).groupby('model').stockout.mean())
    return summary.sort_values('total_cost'), daily, buffers


def run(args):
    for extra in (args.timesfm_source,args.extra_packages):
        if extra:
            sys.path.insert(0,str(extra))
    import torch
    from timesfm3 import TimesFM3Evaluator, ModelConfig
    torch.set_num_threads(args.threads)
    np.random.seed(42)
    torch.manual_seed(42)
    out = ROOT/'results'
    out.mkdir(exist_ok=True)
    observed,y,promo,traffic,calendar = load_panel(ROOT/'data/retail_panel.csv')
    origins = CAL_ORIGINS.append(TEST_ORIGINS)
    # Latest calibration label precedes every test forecast.
    assert CAL_ORIGINS.max()+pd.Timedelta(days=HORIZON-1)<TEST_ORIGINS.min()
    if args.fresh or not (out/'baseline_points.csv').exists():
        boosted = boosting_predictions(origins,observed,y,promo,traffic,calendar)
        baseline_rows=[]
        for origin in origins:
            i=y.index.get_loc(origin)
            split='calibration' if origin in CAL_ORIGINS else 'test'
            for model in MODEL_NAMES[:2]:
                point = (np.tile(y.iloc[i-7:i].to_numpy().T,(1,2)) if model=='Seasonal naive'
                         else np.stack([boosted[(origin,f)] for f in FAMILIES]))
                # Placeholder quantiles are replaced using held-out calibration residuals below.
                baseline_rows.extend(forecast_rows(model,origin,split,point,np.repeat(point[:,:,None],9,axis=2),observed))
        pd.DataFrame(baseline_rows).drop(columns=[f'q{int(q*100):02d}' for q in Q]).to_csv(out/'baseline_points.csv',index=False)
    baseline=pd.read_csv(out/'baseline_points.csv')
    for model in MODEL_NAMES[:2]:
        for family in FAMILIES:
            mask=(baseline.model==model)&(baseline.family==family)
            cal=baseline[mask&(baseline.split=='calibration')]
            offsets=np.quantile(cal.actual-cal.prediction,Q)
            for k,q in enumerate(Q):
                baseline.loc[mask,f'q{int(q*100):02d}']=np.maximum(0,baseline.loc[mask,'prediction']+offsets[k])
    print('Loading official TimesFM 3.0 checkpoint',flush=True)
    forecaster=TimesFM3Evaluator(ModelConfig(checkpoint_path=str(args.checkpoint),device=args.device,
        per_core_batch_size=1,local_files_only=args.checkpoint.exists()))
    all_frames=[baseline]
    timings=[]
    for model in MODEL_NAMES[2:]:
        cache=out/(model.lower().replace(' ','_').replace('+','with')+'.csv')
        if cache.exists() and not args.fresh:
            all_frames.append(pd.read_csv(cache))
            print(f'Reusing completed {model}',flush=True)
            continue
        rows=[]
        for origin in origins:
            context,po,cal,pf=make_inputs(origin,y,promo,traffic,calendar)
            kwargs=dict(contexts=[context],horizon=HORIZON,return_quantiles=True,
                        use_symmetric_averaging=False,make_positive=True,sort_quantiles=True)
            if model=='TimesFM univariate':
                kwargs['univariate']=True
            elif model=='TimesFM + calendar':
                kwargs['past_future_covariates']=[cal]
            elif model=='TimesFM + all inputs':
                kwargs['past_only_covariates']=[po]
                kwargs['past_future_covariates']=[pf]
            before=time.monotonic()
            prediction=list(forecaster.predict_batch(**kwargs))[0]
            elapsed=time.monotonic()-before
            point=np.maximum(0,np.asarray(prediction.forecast))
            quantiles=np.maximum(0,np.asarray(prediction.quantiles))
            assert point.shape==(4,14) and quantiles.shape==(4,14,9)
            assert np.isfinite(quantiles).all() and np.isfinite(point).all()
            assert (np.diff(quantiles,axis=-1)>=-1e-5).all()
            rows.extend(forecast_rows(model,origin,'calibration' if origin in CAL_ORIGINS else 'test',point,quantiles,observed))
            timings.append(dict(model=model,origin=str(origin.date()),seconds=elapsed))
            print(f'{model} | {origin.date()} | {elapsed:.2f}s',flush=True)
        frame=pd.DataFrame(rows)
        frame.to_csv(cache,index=False)
        all_frames.append(frame)
    predictions=pd.concat(all_frames,ignore_index=True)
    predictions.to_csv(out/'forecasts.csv',index=False)
    scores,by_family=evaluate(predictions,observed)
    scores.to_csv(out/'forecast_metrics.csv',index=False)
    by_family.to_csv(out/'metrics_by_family.csv',index=False)
    inventory,daily,buffers=simulate_inventory(predictions,observed)
    inventory.to_csv(out/'inventory_metrics.csv',index=False)
    daily.to_csv(out/'inventory_daily.csv',index=False)
    buffers.to_csv(out/'aggregate_error_buffers.csv',index=False)
    if timings:
        pd.DataFrame(timings).to_csv(out/'inference_timings.csv',index=False)
    sensitivity=[]
    for h,s in [(.005,1),(.01,1),(.05,1),(.01,2)]:
        frame,_,_=simulate_inventory(predictions,observed,h,s)
        frame['holding_rate'],frame['shortage_rate']=h,s
        sensitivity.append(frame)
    pd.concat(sensitivity).to_csv(out/'inventory_cost_sensitivity.csv',index=False)
    config=dict(model='google/timesfm-3.0-pytorch',context=CONTEXT,horizon=HORIZON,
        calibration_origins=[str(d.date()) for d in CAL_ORIGINS],test_origins=[str(d.date()) for d in TEST_ORIGINS],
        seed=42,device=args.device,threads=args.threads,python=platform.python_version(),torch=torch.__version__,
        numpy=np.__version__,pandas=pd.__version__,sklearn=__import__('sklearn').__version__,
        future_input_assumption='Calendar known; promotion counts assumed planned; no future traffic',
        missing_context_handling='Previous same weekday, causal, for missing source Christmas observations',
        forecast_evaluation='All 14 horizons at 12 weekly origins; overlapping targets; not independent samples',
        inventory=dict(review_days=REVIEW,lead_days=LEAD,protection_days=HORIZON,days=84,
            holding_per_unit_day=.01,penalty_per_unmet_unit=1,units='illustrative cost units',
            buffer='90th empirical quantile of 14-day aggregate signed errors at 12 calibration origins; fixed during test',
            initialization='Same on-hand across models: twice previous 7-day observed sales; empty pipeline',
            ending='No terminal salvage or penalty on pipeline stock; open-system operational replay'))
    (out/'run_config.json').write_text(json.dumps(config,indent=2))
    print('\nFORECAST RESULTS\n'+scores.to_string(index=False),flush=True)
    print('\nINVENTORY RESULTS\n'+inventory.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--timesfm-source',type=Path)
    parser.add_argument('--extra-packages',type=Path)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--fresh',action='store_true',help='Recompute all models instead of reusing saved per-model forecasts')
    run(parser.parse_args())
