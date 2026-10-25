"""Create an executed nbformat-4 walkthrough without a Jupyter dependency.

Each analysis code cell is executed in a shared Python namespace and its actual
stdout is captured into a standard notebook output. Model rerun commands are
documented separately rather than falsely marked executed in the notebook.
"""
import base64
import contextlib
import io
import json
from pathlib import Path
import traceback

ROOT=Path(__file__).resolve().parents[1]

def main():
    cells=[]
    namespace={'__name__':'__main__','PROJECT_ROOT':ROOT}
    count=0
    def markdown(text):
        cells.append(dict(cell_type='markdown',id=f'cell-{len(cells):03d}',metadata={},source=text.splitlines(keepends=True)))
    def code(source):
        nonlocal count
        count+=1
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(source,f'<portfolio-cell-{count}>','exec'),namespace)
        outputs=[dict(output_type='stream',name='stdout',text=stream.getvalue().splitlines(keepends=True))] if stream.getvalue() else []
        cells.append(dict(cell_type='code',id=f'cell-{len(cells):03d}',metadata={},execution_count=count,
            source=source.splitlines(keepends=True),outputs=outputs))
    markdown('''# Retail Demand Lab: TimesFM 3.0

An executed analysis of real Favorita observations and saved official TimesFM 3.0 predictions. The forecasting run is separate from these lightweight analysis cells. See README for the full inference commands.

**Business question:** Do multivariate sales and extra inputs improve forecasts and inventory decisions?

**Scope:** Store 1, Quito; four families; twelve held-out weekly origins in 2017; 14-day forecasts. Future promotions are an assumed-known scenario, and prices are absent.''')
    code('''from pathlib import Path
import json
import pandas as pd
import numpy as np

# Portable whether opened in the project folder or its parent.
root = Path.cwd()
if not (root / 'data' / 'retail_panel.csv').exists():
    if (root / 'retail-demand-lab' / 'data').exists():
        root = root / 'retail-demand-lab'
    elif 'PROJECT_ROOT' in globals():
        root = PROJECT_ROOT
    else:
        raise FileNotFoundError('Open this notebook from the project directory.')
panel = pd.read_csv(root / 'data' / 'retail_panel.csv', parse_dates=['date'])
forecasts = pd.read_csv(root / 'results' / 'forecasts.csv')
scores = pd.read_csv(root / 'results' / 'forecast_metrics.csv')
inventory = pd.read_csv(root / 'results' / 'inventory_metrics.csv')
print('Panel rows:', len(panel), '| Observed sales:', panel.sales.notna().sum())
print('Saved forecast rows:', len(forecasts))
print(panel.groupby('family').sales.agg(['count', 'mean', 'std']).round(2).to_string())''')
    markdown('''## 1. Input and split audit

Targets are observed product-family sales, which can be fractional. Past-only traffic is measured by transaction counts. Only calendar and assumed-planned promotion inputs extend into the forecast horizon. Future sales and realized traffic never enter input features. Missing Christmas history is filled causally from the previous same weekday, while missing source targets remain identifiable in the exported panel.''')
    code('''config = json.loads((root / 'results' / 'run_config.json').read_text())
cal_last = pd.Timestamp(config['calibration_origins'][-1]) + pd.Timedelta(days=13)
test_first = pd.Timestamp(config['test_origins'][0])
assert cal_last < test_first
print('Last calibration target:', cal_last.date())
print('First test forecast:', test_first.date())
print('Context:', config['context'], '| Horizon:', config['horizon'])
print('Missing raw sales observations:', panel.sales.isna().sum())
print(config['future_input_assumption'])''')
    markdown('''## 2. Forecast comparison

WAPE = total absolute error / total observed sales. Macro MASE averages scaled errors across families. Forecast windows overlap; a separate next-week score avoids repeated dates. TimesFM uses pretrained weights without task-specific fitting; gradient boosting is a fitted conventional comparator.''')
    code('''print(scores[['model', 'wape', 'macro_mase', 'coverage80', 'pinball']].round(4).to_string(index=False))
best_timesfm = scores[scores.model.str.startswith('TimesFM')].sort_values('wape').iloc[0]
naive = scores[scores.model == 'Seasonal naive'].iloc[0]
print(f"\\nBest TimesFM relative WAPE reduction vs naive: {1-best_timesfm.wape/naive.wape:.1%}")
print('\\nNon-overlapping next-week scores:')
print(pd.read_csv(root / 'results' / 'next_week_metrics.csv')[['model', 'wape']].round(4).to_string(index=False))''')
    markdown('''![Forecast and inventory comparison](results/figures/results_overview.png)

Calendar-aware TimesFM and gradient boosting are essentially tied. Additional promotion and historical transaction channels did not improve the calendar-only model here. These fixed ablations cannot identify the separate causal effect of promotions or traffic.''')
    markdown('''## 3. Inspect a saved held-out forecast

These values are actual saved outputs from the official checkpoint, not generated example numbers. The offline report adds selectors for family, origin, and configuration.''')
    code('''sample = forecasts[(forecasts.model == 'TimesFM + calendar') &
                   (forecasts.origin == '2017-08-02') &
                   (forecasts.family == 'BEVERAGES')]
print(sample[['date','actual','prediction','q10','q50','q90']].round(2).to_string(index=False))
qcols = [f'q{q:02d}' for q in range(10,100,10)]
assert (np.diff(forecasts[qcols].to_numpy(), axis=1) >= -1e-5).all()
print('\\nAll saved quantiles are ordered.')''')
    markdown('''![Four-family held-out forecast](results/figures/held_out_forecast.png)

## 4. Inventory policy and aggregate uncertainty

Weekly review, seven-day delivery lead time, 14-day protection period. Safety buffers use the 90th empirical percentile of signed aggregate 14-day point forecast errors from pre-test calibration. TimesFM point forecasts are medians. The procedure never sums daily 90th percentiles. Calibration windows overlap and the sample is small, so no formal 90% service guarantee is claimed.''')
    code('''buffers = pd.read_csv(root / 'results' / 'aggregate_error_buffers.csv')
print(buffers.round(2).to_string(index=False))
print('\\nInventory replay:')
print(inventory[['model','fill_rate','lost_sales','average_inventory','total_cost']].round(3).to_string(index=False))
inv_best = inventory.sort_values('total_cost').iloc[0]
inv_naive = inventory[inventory.model == 'Seasonal naive'].iloc[0]
print(f"\\nLowest replay cost: {inv_best.model}; reduction vs naive: {1-inv_best.total_cost/inv_naive.total_cost:.1%}")''')
    markdown('''## 5. Verify an inventory balance

For each family-day: previous on-hand + received stock − served sales = ending stock. All models use identical initial stock, calculated from the week before testing. Costs are illustrative units and observed sales are an imperfect demand proxy.''')
    code('''daily = pd.read_csv(root / 'results' / 'inventory_daily.csv')
truth = panel.pivot(index='date', columns='family', values='sales')
for (_, family), group in daily.groupby(['model','family']):
    initial = 2 * truth.loc['2017-05-10':'2017-05-16', family].sum()
    previous = np.r_[initial, group.ending_inventory.to_numpy()[:-1]]
    np.testing.assert_allclose(previous + group.arrival - group.served, group.ending_inventory, atol=1e-7)
    np.testing.assert_allclose(group.served + group.lost, group.demand, atol=1e-7)
print('Daily stock and demand balances verified for all 6 models and 4 families.')
print('\\nCost sensitivity (same orders, different unit costs):')
sensitivity = pd.read_csv(root / 'results' / 'inventory_cost_sensitivity.csv')
print(sensitivity.sort_values('total_cost').groupby(['holding_rate','shortage_rate']).first()[['model','total_cost']].round(2).to_string())''')
    markdown('''## Interpretation and limits

- The model with the lowest forecast error was not the model with the lowest simulated inventory cost.
- This is a single-store historical benchmark. No statistical significance, universal superiority, or actual business savings claim is justified.
- Promotion availability is assumed; target sales may be censored by stockouts; product-family units differ; no prices are included.
- Public data may overlap foundation-model pretraining, whose full lineage cannot be independently audited here.
- The inventory model omits capacity, expiry, buying costs, and terminal valuation. It supplies a reproducible experiment, not a production policy.
- Downloadable TimesFM 3.0 weights have a non-commercial, non-production license. Original data terms also apply.

For exact sources, settings, full experiment rerun commands, and six integrity tests, see the README and CASE_STUDY.''')
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python',version='3.12.0')),nbformat=4,nbformat_minor=5)
    (ROOT/'project_walkthrough.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    print(f'Executed {count} analysis cells and saved project_walkthrough.ipynb')

if __name__=='__main__':main()
