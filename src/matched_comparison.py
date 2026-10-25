"""Matched-input, matched-local-window correction to the initial comparison.

Fit three fixed gradient boosting workflows. Each sees the same raw input types,
four sales groups and 256-day local history as its saved TimesFM counterpart.
The representation, upstream pretraining and fitting procedure remain different.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from experiment import (CALENDAR, CAL_ORIGINS, CONTEXT, FAMILIES, HORIZON, Q,
                        ROOT, TEST_ORIGINS, evaluate, forecast_rows, load_panel,
                        simulate_inventory)

MODES = ('sales', 'calendar', 'all')
GB_NAMES = {'sales': 'Gradient boosting: sales', 'calendar': 'Gradient boosting: calendar',
            'all': 'Gradient boosting: all inputs'}
TF_NAMES = {'sales': 'TimesFM multivariate', 'calendar': 'TimesFM + calendar',
            'all': 'TimesFM + all inputs'}
PARAMS = dict(max_iter=80, max_leaf_nodes=15, learning_rate=.07,
              l2_regularization=10, random_state=42, early_stopping=False)
OUT = ROOT / 'results/matched_inputs'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def features(i, h, sales, promotions, transactions, calendar, mode):
    """Summarize only history before i, plus allowed signals at i+h.

    Every family receives the histories of all four target groups. Excluded
    channels are never inspected. The same representation is fixed for all runs.
    """
    if mode not in MODES:
        raise ValueError(mode)
    history = sales[i-28:i]
    if history.shape != (28, 4):
        raise ValueError('Each example requires 28 preceding days for four groups')
    values, names = [h + 1], ['horizon']
    for f, family in enumerate(FAMILIES):
        y = history[:, f]
        values.extend([y[-7 + h % 7], np.mean(y[-7 + h % 7::-7]), y[-1],
                       y[-7:].mean(), y.mean(), y[-7:].std(),
                       y[-7:].mean() - y[-14:-7].mean()])
        names.extend(f'{family}:{stat}' for stat in
                     ('seasonal', 'weekday_mean', 'last', 'mean7', 'mean28', 'std7', 'trend7'))
    if mode in ('calendar', 'all'):
        values.extend(calendar[i+h])
        names.extend(CALENDAR)
    if mode == 'all':
        values.extend([transactions[i-7:i].mean(), transactions[i-1]])
        names.extend(['transactions:mean7', 'transactions:last'])
        for f, family in enumerate(FAMILIES):
            values.extend([promotions[i+h, f], promotions[i-7:i, f].mean(),
                           promotions[i-28:i, f].mean()])
            names.extend(f'{family}:promotion:{stat}' for stat in ('planned', 'mean7', 'mean28'))
    return np.asarray(values, dtype=float), names


def design(origin, observed, y, promo, traffic, calendar, mode, history_days=CONTEXT):
    """Fit only inside the declared window (256 days in the matched comparison).

    Earlier examples use 28 days of warm-up within that window. No training
    label may reach the current forecast start. Actual future sales/traffic
    and all rows before the permitted window are sliced out before construction.
    """
    stop = y.index.get_loc(origin)
    start = stop - history_days
    if start < 0:
        raise ValueError('Insufficient shared history')
    ys = y.iloc[start:stop].to_numpy()
    truth = observed.iloc[start:stop].to_numpy()
    ps = promo.iloc[start:stop+HORIZON].to_numpy()
    ts = traffic.iloc[start:stop].to_numpy()
    cs = calendar.iloc[start:stop+HORIZON].to_numpy()
    x_rows, labels, starts, targets = [], [], [], []
    names = None
    for i in range(28, history_days):
        for h in range(HORIZON):
            target = i + h
            if target >= history_days:
                continue
            x, names = features(i, h, ys, ps, ts, cs, mode)
            if np.isfinite(x).all() and np.isfinite(truth[target]).all():
                x_rows.append(x)
                labels.append(truth[target])
                starts.append(i)
                targets.append(target)
    future = np.stack([features(history_days, h, ys, ps, ts, cs, mode)[0]
                       for h in range(HORIZON)])
    x, labels = np.stack(x_rows), np.stack(labels)
    assert np.isfinite(x).all() and np.isfinite(future).all()
    target_dates = y.index[start + np.asarray(targets)]
    assert target_dates.max() < origin
    return dict(x=x, labels=labels, future=future, feature_names=names,
                example_origins=np.asarray(starts), target_indices=np.asarray(targets),
                history_start=y.index[start], history_end=y.index[stop-1],
                first_label=target_dates.min(), last_label=target_dates.max())


def attach_quantiles(points):
    result = points.copy()
    for (_, _), index in result.groupby(['model', 'family']).groups.items():
        group = result.loc[index]
        cal = group[group.split == 'calibration']
        offsets = np.quantile(cal.actual - cal.prediction, Q)
        for q, offset in zip(Q, offsets):
            result.loc[index, f'q{int(q*100):02d}'] = np.maximum(0, group.prediction + offset)
    return result


def summarize(predictions, observed):
    scores, families = evaluate(predictions, observed)
    scores.to_csv(OUT / 'forecast_metrics.csv', index=False)
    families.to_csv(OUT / 'metrics_by_family.csv', index=False)
    first_week, _ = evaluate(predictions[predictions.horizon <= 7], observed)
    first_week.to_csv(OUT / 'next_week_metrics.csv', index=False)
    pairs, family_pairs = [], []
    for mode in MODES:
        a = scores[scores.model == TF_NAMES[mode]].iloc[0]
        b = scores[scores.model == GB_NAMES[mode]].iloc[0]
        pairs.append(dict(inputs=mode, timesfm_wape=a.wape, boosting_wape=b.wape,
                          timesfm_minus_boosting_pp=100*(a.wape-b.wape),
                          timesfm_coverage80=a.coverage80, boosting_coverage80=b.coverage80,
                          timesfm_pinball=a.pinball, boosting_pinball=b.pinball,
                          timesfm_mean_width=a.mean_interval_width, boosting_mean_width=b.mean_interval_width))
        for family in FAMILIES:
            fa = families[(families.model == TF_NAMES[mode]) & (families.family == family)].iloc[0]
            fb = families[(families.model == GB_NAMES[mode]) & (families.family == family)].iloc[0]
            family_pairs.append(dict(inputs=mode, family=family, timesfm_wape=fa.wape,
                                     boosting_wape=fb.wape, timesfm_minus_boosting_pp=100*(fa.wape-fb.wape)))
    pd.DataFrame(pairs).to_csv(OUT / 'paired_comparison.csv', index=False)
    pd.DataFrame(family_pairs).to_csv(OUT / 'paired_comparison_by_family.csv', index=False)
    test = predictions[predictions.split == 'test'].copy()
    test['absolute_error'] = abs(test.actual-test.prediction)
    per_start = test.groupby(['model', 'origin'])[['actual', 'absolute_error']].sum().reset_index()
    per_start['wape'] = per_start.absolute_error/per_start.actual
    per_start.to_csv(OUT / 'metrics_by_forecast_start.csv', index=False)
    inventory, daily, buffers = simulate_inventory(predictions, observed,
                                                   model_names=sorted(predictions.model.unique()))
    inventory.to_csv(OUT / 'inventory_metrics.csv', index=False)
    daily.to_csv(OUT / 'inventory_daily.csv', index=False)
    buffers.to_csv(OUT / 'aggregate_error_buffers.csv', index=False)
    print(pd.DataFrame(pairs).to_string(index=False), flush=True)


def run(threads=8):
    OUT.mkdir(parents=True, exist_ok=True)
    observed, y, promo, traffic, calendar = load_panel(ROOT / 'data/retail_panel.csv')
    source_path = ROOT / 'results/forecasts.csv'
    source = pd.read_csv(source_path)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Correct unequal covariates, local history budgets and related-series access in the initial comparison',
        status='Post-hoc correction on the previously examined test period; not a new untouched validation',
        modes=list(MODES), families=FAMILIES, context=CONTEXT, horizon=HORIZON,
        calibration_origins=[str(x.date()) for x in CAL_ORIGINS],
        test_origins=[str(x.date()) for x in TEST_ORIGINS],
        boosting_parameters=PARAMS, boosting_features='28-day summaries for all four groups, horizon, plus only allowed covariates',
        training='Per-family direct regressors; all features and labels inside 256-day window; 28-day warm-up inside window',
        no_tuning='Parameters inherited from initial experiment; no search or adjustment after inspecting new scores',
        known_future='Calendar; promotion counts assumed planned in all-input mode; no future realized transactions',
        upstream_pretraining='TimesFM has external pretraining; gradient boosting fits only the local window',
        uncertainty='Native TimesFM quantiles versus family-specific signed calibration-residual quantiles for boosting',
        reused_timesfm_predictions=True, source_forecasts_sha256=digest(source_path),
        prepared_panel_sha256=digest(ROOT / 'data/retail_panel.csv'),
        sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__, threads=threads)
    # Save the declared settings before any new model fits.
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    points, audit = [], []
    with threadpool_limits(limits=threads):
        for mode in MODES:
            for origin in CAL_ORIGINS.append(TEST_ORIGINS):
                before = time.monotonic()
                d = design(origin, observed, y, promo, traffic, calendar, mode)
                output = []
                for f, _ in enumerate(FAMILIES):
                    model = HistGradientBoostingRegressor(**PARAMS)
                    model.fit(d['x'], d['labels'][:, f])
                    output.append(np.maximum(0, model.predict(d['future'])))
                prediction = np.stack(output)
                split = 'calibration' if origin in CAL_ORIGINS else 'test'
                points.extend(forecast_rows(GB_NAMES[mode], origin, split, prediction,
                              np.repeat(prediction[:, :, None], 9, axis=2), observed))
                audit.append(dict(mode=mode, origin=str(origin.date()),
                    history_start=str(d['history_start'].date()), history_end=str(d['history_end'].date()),
                    first_label=str(d['first_label'].date()), last_label=str(d['last_label'].date()),
                    training_examples=len(d['x']), feature_count=len(d['feature_names']),
                    seconds=time.monotonic()-before))
                print(f"{mode}: {origin.date()} | {audit[-1]['seconds']:.2f}s", flush=True)
    boosted = attach_quantiles(pd.DataFrame(points))
    boosted.to_csv(OUT / 'boosting_forecasts.csv', index=False)
    reused = source[source.model.isin([*TF_NAMES.values(), 'Seasonal naive'])].copy()
    combined = pd.concat([reused, boosted], ignore_index=True)
    assert len(combined) == 7*24*4*14
    assert not combined.duplicated(['model', 'origin', 'family', 'horizon']).any()
    combined.to_csv(OUT / 'forecasts.csv', index=False)
    pd.DataFrame(audit).to_csv(OUT / 'training_audit.csv', index=False)
    (OUT / 'feature_names.json').write_text(json.dumps({mode: design(TEST_ORIGINS[0], observed, y, promo, traffic,
        calendar, mode)['feature_names'] for mode in MODES}, indent=2), encoding='utf-8')
    summarize(combined, observed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=8)
    run(parser.parse_args().threads)
