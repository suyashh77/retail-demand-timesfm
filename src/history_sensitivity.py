"""Check the matched calendar baseline with 365 rather than 256 local days.

This sensitivity changes only the local window in the corrected boosting
workflow. It is not a matched-window pair and is reported separately. It was
added as a design check after seeing the shared-256-day correction.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from matched_comparison import (CAL_ORIGINS, FAMILIES, HORIZON, OUT, PARAMS, Q,
                                ROOT, TEST_ORIGINS, attach_quantiles, design,
                                evaluate, forecast_rows, load_panel)


def run():
    destination = OUT / 'history_sensitivity'
    destination.mkdir(exist_ok=True)
    protocol = dict(reason='Measure the effect of a longer local window in the corrected calendar boosting workflow',
                    history_days=365, unchanged='Four-group sales summaries, calendar inputs, parameters, dates and scoring',
                    caveat='Longer boosting history than TimesFM local context; not a matched-window comparison',
                    design_timing='Specified after inspecting the 256-day correction; no parameter tuning',
                    parameters=PARAMS)
    (destination / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    panel = load_panel(ROOT / 'data/retail_panel.csv')
    rows, audit = [], []
    name = 'Gradient boosting: calendar, 365 days'
    with threadpool_limits(limits=8):
        for origin in CAL_ORIGINS.append(TEST_ORIGINS):
            before = time.monotonic()
            d = design(origin, *panel, 'calendar', history_days=365)
            points = []
            for f, _ in enumerate(FAMILIES):
                model = HistGradientBoostingRegressor(**PARAMS)
                model.fit(d['x'], d['labels'][:, f])
                points.append(np.maximum(0, model.predict(d['future'])))
            points = np.stack(points)
            rows.extend(forecast_rows(name, origin, 'calibration' if origin in CAL_ORIGINS else 'test',
                        points, np.repeat(points[:, :, None], 9, axis=2), panel[0]))
            audit.append(dict(origin=str(origin.date()), history_start=str(d['history_start'].date()),
                              history_end=str(d['history_end'].date()), last_label=str(d['last_label'].date()),
                              training_examples=len(d['x']), seconds=time.monotonic()-before))
            print(f"calendar 365 days: {origin.date()} | {audit[-1]['seconds']:.2f}s", flush=True)
    forecasts = attach_quantiles(pd.DataFrame(rows))
    forecasts.to_csv(destination / 'forecasts.csv', index=False)
    pd.DataFrame(audit).to_csv(destination / 'training_audit.csv', index=False)
    scores, families = evaluate(forecasts, panel[0])
    scores.to_csv(destination / 'forecast_metrics.csv', index=False)
    families.to_csv(destination / 'metrics_by_family.csv', index=False)
    # Separate elementary check of the main score and local time boundaries.
    test = forecasts[forecasts.split == 'test']
    wape = sum(abs(float(r.actual)-float(r.prediction)) for r in test.itertuples()) / sum(float(v) for v in test.actual)
    assert abs(wape-float(scores.wape.iloc[0])) < 1e-12
    assert len(test) == 672
    for record in audit:
        origin = pd.Timestamp(record['origin'])
        assert pd.Timestamp(record['history_start']) == origin-pd.Timedelta(days=365)
        assert pd.Timestamp(record['history_end']) == origin-pd.Timedelta(days=1)
        assert pd.Timestamp(record['last_label']) < origin
    print(scores.to_string(index=False), flush=True)


if __name__ == '__main__':
    run()
