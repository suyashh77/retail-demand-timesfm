"""Check saved evidence with standard-library arithmetic, without experiment.py.

This verifies the recorded artifacts, not the model's training lineage or every
implementation choice. Run after preparing data and computing the results.
"""
import csv
from collections import defaultdict
from datetime import date, timedelta
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / name).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def close(actual, expected):
    assert math.isclose(actual, float(expected), rel_tol=1e-9, abs_tol=1e-7), (actual, expected)


def linear_quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    left = math.floor(position)
    right = math.ceil(position)
    return ordered[left] + (position - left) * (ordered[right] - ordered[left])


def main():
    forecasts = read('results/forecasts.csv')
    source = {(r['date'], r['family']): float(r['sales'])
              for r in read('data/retail_panel.csv') if r['sales']}
    assert len(forecasts) == 8064
    assert len({(r['model'], r['origin'], r['family'], r['horizon']) for r in forecasts}) == 8064
    grouped, calibration, windows = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in forecasts:
        origin = date.fromisoformat(row['origin'])
        day = date.fromisoformat(row['date'])
        assert day == origin + timedelta(days=int(row['horizon']) - 1)
        close(source[row['date'], row['family']], row['actual'])
        values = [float(row[k]) for k in ['actual', 'prediction'] + [f'q{q}' for q in range(10, 100, 10)]]
        assert all(math.isfinite(v) and v >= 0 for v in values)
        assert values[2:] == sorted(values[2:])
        key = row['model'], row['family']
        windows[key + (row['origin'],)].append(row)
        if row['split'] == 'calibration':
            assert day < date(2017, 5, 17)
            calibration[key].append(float(row['actual']) - float(row['prediction']))
        else:
            assert row['split'] == 'test'
            grouped[row['model']].append(row)
        if row['model'].startswith('TimesFM'):
            close(float(row['q50']), row['prediction'])
        if row['model'] == 'Seasonal naive':
            reference = origin - timedelta(days=7) + timedelta(days=(int(row['horizon']) - 1) % 7)
            close(source[reference.isoformat(), row['family']], row['prediction'])

    # The reported summaries are checked using a separate implementation.
    calculated = {}
    for model, rows in grouped.items():
        assert len(rows) == 672
        actual = sum(float(r['actual']) for r in rows)
        absolute = sum(abs(float(r['prediction']) - float(r['actual'])) for r in rows)
        covered = sum(float(r['q10']) <= float(r['actual']) <= float(r['q90']) for r in rows)
        losses = []
        for r in rows:
            for percentile in range(10, 100, 10):
                q = percentile / 100
                residual = float(r['actual']) - float(r[f'q{percentile}'])
                losses.append(max(q * residual, (q - 1) * residual))
        calculated[model] = dict(n=len(rows), wape=absolute / actual,
            mae=absolute / len(rows), bias=sum(float(r['prediction']) - float(r['actual']) for r in rows) / actual,
            coverage80=covered / len(rows), pinball=sum(losses) / len(losses),
            mean_interval_width=sum(float(r['q90']) - float(r['q10']) for r in rows) / len(rows))
    for saved in read('results/forecast_metrics.csv'):
        for field, value in calculated[saved['model']].items():
            close(value, saved[field])
    for saved in read('results/metrics_by_family.csv'):
        rows = [r for r in grouped[saved['model']] if r['family'] == saved['family']]
        close(sum(abs(float(r['actual']) - float(r['prediction'])) for r in rows)
              / sum(float(r['actual']) for r in rows), saved['wape'])

    for row in forecasts:
        if row['model'] in ('Seasonal naive', 'Gradient boosting'):
            errors = calibration[row['model'], row['family']]
            for percentile in range(10, 100, 10):
                close(max(0, float(row['prediction']) + linear_quantile(errors, percentile / 100)),
                      row[f'q{percentile}'])

    buffers = {}
    for saved in read('results/aggregate_error_buffers.csv'):
        key = saved['model'], saved['family']
        totals = [sum(float(r['actual']) - float(r['prediction']) for r in rows)
                  for window_key, rows in windows.items()
                  if window_key[:2] == key and rows[0]['split'] == 'calibration']
        assert len(totals) == 12
        buffer = max(0, sorted(totals)[math.ceil((len(totals) - 1) * .9)])
        close(buffer, saved['buffer'])
        buffers[key] = buffer

    daily = defaultdict(list)
    for r in read('results/inventory_daily.csv'):
        daily[r['model'], r['family']].append(r)
    inventory_totals = defaultdict(lambda: defaultdict(float))
    for key, rows in daily.items():
        rows.sort(key=lambda r: r['date'])
        assert len(rows) == 84
        first = date.fromisoformat(rows[0]['date'])
        stock = 2 * sum(source[(first - timedelta(days=d)).isoformat(), key[1]] for d in range(1, 8))
        orders = {}
        for r in rows:
            day = date.fromisoformat(r['date'])
            arrival = orders.get(day - timedelta(days=7), 0)
            close(arrival, r['arrival'])
            stock += arrival
            forecast = windows.get(key + (r['date'],))
            outstanding = sum(qty for ordered, qty in orders.items() if ordered + timedelta(days=7) > day)
            expected_order = max(0, sum(float(f['prediction']) for f in forecast) + buffers[key]
                                 - stock - outstanding) if forecast else 0
            close(expected_order, r['ordered'])
            orders[day] = float(r['ordered'])
            demand = source[r['date'], key[1]]
            served = min(stock, demand)
            stock -= served
            close(demand, r['demand'])
            close(served, r['served'])
            close(demand - served, r['lost'])
            close(stock, r['ending_inventory'])
            close(.01 * stock, r['holding_cost'])
            close(demand - served, r['shortage_cost'])
            close(.01 * stock + demand - served, r['total_cost'])
            totals = inventory_totals[key[0]]
            for field in ('demand', 'holding_cost', 'shortage_cost', 'total_cost'):
                totals[field] += float(r[field])
            totals['lost_sales'] += float(r['lost'])
            totals['average_inventory'] += stock / (84 * 4)
    for saved in read('results/inventory_metrics.csv'):
        totals = inventory_totals[saved['model']]
        totals['fill_rate'] = 1 - totals['lost_sales'] / totals['demand']
        for field, value in totals.items():
            close(value, saved[field])

    summary = dict(status='passed', forecast_rows_checked=len(forecasts),
        test_rows_per_model=672, inventory_rows_checked=sum(map(len, daily.values())),
        checks=['Target dates and actual values match prepared source data',
                'Seasonal baseline uses only the previous observed week',
                'WAPE, MAE, bias, coverage, width and nine-quantile pinball match saved overall scores',
                'Family WAPE matches saved family scores',
                'Baseline quantiles use only calibration residuals',
                'Aggregate buffers use only twelve calibration windows',
                'Orders, seven-day arrivals, served quantity, stock balances and default costs match',
                'Inventory totals and fill rates match'],
        forecast_metrics=calculated,
        implementation='Python standard library; does not import experiment.py or audit_results.py',
        limits='Checks saved artifacts. Does not establish pretraining independence, statistical superiority, '
               'real-time promotion availability, or production suitability.')
    (ROOT / 'results/review_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
