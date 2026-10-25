"""Publish the matched comparison as documentation and a static results page."""
import json
from pathlib import Path

import pandas as pd

from experiment import ROOT
from matched_comparison import GB_NAMES, TF_NAMES

OUT = ROOT / 'results/matched_inputs'
LABELS = {'sales': 'Sales histories only', 'calendar': 'Sales + calendar',
          'all': 'Sales + calendar + promotions + past transactions'}
REPO = 'https://github.com/suyashh77/retail-demand-timesfm'
BLOB = REPO + '/blob/codex/retail-demand-study/'


def comparison_tables():
    pairs = pd.read_csv(OUT / 'paired_comparison.csv')
    families = pd.read_csv(OUT / 'paired_comparison_by_family.csv')
    headline = pairs[['inputs', 'timesfm_wape', 'boosting_wape']].copy()
    headline['inputs'] = headline.inputs.map(LABELS)
    for col in ['timesfm_wape', 'boosting_wape']:
        headline[col] = headline[col].map(lambda x: f'{100*x:.2f}%')
    headline.columns = ['Inputs available to both models', 'TimesFM WAPE', 'Gradient boosting WAPE']
    family = families[families.inputs == 'calendar'][['family', 'timesfm_wape', 'boosting_wape']].copy()
    for col in ['timesfm_wape', 'boosting_wape']:
        family[col] = family[col].map(lambda x: f'{100*x:.2f}%')
    family.columns = ['Product group, calendar pair', 'TimesFM WAPE', 'Gradient boosting WAPE']
    full_family = families[families.inputs == 'all'][['family', 'timesfm_wape', 'boosting_wape']].copy()
    for col in ['timesfm_wape', 'boosting_wape']:
        full_family[col] = full_family[col].map(lambda x: f'{100*x:.2f}%')
    full_family.columns = ['Product group, full-input pair', 'TimesFM WAPE', 'Gradient boosting WAPE']
    uncertainty = pairs[['inputs', 'timesfm_coverage80', 'boosting_coverage80',
                         'timesfm_pinball', 'boosting_pinball']].copy()
    uncertainty['inputs'] = uncertainty.inputs.map(LABELS)
    for col in ['timesfm_coverage80', 'boosting_coverage80']:
        uncertainty[col] = uncertainty[col].map(lambda x: f'{100*x:.1f}%')
    for col in ['timesfm_pinball', 'boosting_pinball']:
        uncertainty[col] = uncertainty[col].map(lambda x: f'{x:.2f}')
    uncertainty.columns = ['Shared inputs', 'TimesFM 80% coverage', 'Boosting 80% coverage',
                           'TimesFM quantile loss', 'Boosting quantile loss']
    return pairs, headline, family, full_family, uncertainty


def markdown_table(frame):
    rows = ['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---']*len(frame.columns)) + ' |']
    rows.extend('| ' + ' | '.join(str(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None))
    return '\n'.join(rows)


def build():
    pairs, headline, family, full_family, uncertainty = comparison_tables()
    cal = pairs[pairs.inputs == 'calendar'].iloc[0]
    inventory = pd.read_csv(OUT / 'inventory_metrics.csv')
    ti = inventory[inventory.model == TF_NAMES['calendar']].iloc[0]
    gi = inventory[inventory.model == GB_NAMES['calendar']].iloc[0]
    protocol = json.loads((OUT / 'protocol.json').read_text())
    result_sentence = (f'With sales and calendar inputs, TimesFM scored {100*cal.timesfm_wape:.2f}% WAPE '
                       f'and gradient boosting scored {100*cal.boosting_wape:.2f}%. '
                       'This replaces the earlier headline comparison with unequal inputs.')
    method = ('Both workflows have access to the same four product groups and the same latest 256 daily observations. '
              'They predict the same fourteen days at the same twelve weekly test starts. '
              'The sales-only pair excludes calendar, promotions and transactions. The calendar pair adds only calendar signals. '
              'The full-input pair adds the four promotion series and historical store transactions to both workflows. '
              'Future realized transactions and sales are excluded.')
    differences = ('TimesFM uses the saved official checkpoint forecasts, whose inputs have not changed. '
                   'Its weights remain unchanged. Each gradient boosting model is fitted separately per target and forecast start, '
                   'using supervised examples drawn entirely from that 256-day window. Each example includes sales summaries '
                   'for all four groups, a horizon indicator and only the allowed covariates. '
                   'The 28-day feature warm-up is inside the shared window, leaving fewer labeled training days than 256. '
                   'Matching available local data does not equate the feature representation, external pretraining, model capacity or fitting procedure.')
    limitation = ('This correction was designed after the original results had been examined. '
                  'It reuses the same historical test period and does not provide a new untouched validation. '
                  'There is one store, overlapping forecast windows, assumed-known promotions and possible pretraining overlap. '
                  'No significance or statistical equivalence claim is made. '
                  'The fixed boosting settings were inherited from the original run, without a new parameter search.')
    sensitivity_path = OUT / 'history_sensitivity/forecast_metrics.csv'
    sensitivity = ''
    if sensitivity_path.exists():
        longer = pd.read_csv(sensitivity_path).iloc[0]
        sensitivity = (f'With the corrected calendar boosting representation kept fixed, extending its local window from 256 to 365 days '
                       f'changed WAPE from {100*cal.boosting_wape:.2f}% to {100*longer.wape:.2f}%. '
                       'The latter has more local history than the TimesFM context and is a separate sensitivity check. '
                       'The original 9.69% boosting score used a different, own-group feature representation, a year of labels and extra covariates. '
                       'Changes from that old score cannot be attributed solely to removing covariates. '
                       'See results/matched_inputs/history_sensitivity/ for the forecasts and declared settings.')
    inventory_text = (f'Under the same illustrative weekly-order rule, calendar TimesFM filled {100*ti.fill_rate:.2f}% '
                      f'of the recorded sales quantity and calendar gradient boosting filled {100*gi.fill_rate:.2f}%. '
                      f'Default replay costs were {ti.total_cost:,.0f} and {gi.total_cost:,.0f} illustrative units respectively. '
                      'Buffers differ by workflow because each is estimated from that workflow\'s earlier aggregate errors. '
                      'This is a model-plus-buffer policy comparison, not measured retailer savings. '
                      'Native daily prediction intervals do not set these buffers.')
    doc = f'''# Matched-input correction

## Situation

The initial headline compared calendar-only TimesFM with gradient boosting that also received promotions and transactions. The workflows also differed in local history and related-series access. That comparison could not isolate differences between forecasting methods.

## Task

Compare three pairs with matching raw input types, product-group access, local history windows, forecast dates and horizon. Preserve the initial outputs as a separate exploratory stage.

## Analysis

{method}

{differences}

Gradient boosting uses `max_iter=80`, `max_leaf_nodes=15`, `learning_rate=0.07`, `l2_regularization=10`, `random_state=42`, and `early_stopping=False`. Historical training origins begin after the in-window 28-day warm-up. Labels end before the current forecast start. Missing Christmas labels are excluded; context filling follows the original causal procedure.

Calendar features describe weekday, annual position, applicable holidays and payday indicators. Promotions remain an assumed-known scenario because dated plan versions are unavailable. The full-input comparison changes promotions and transactions together and cannot isolate them.

{limitation}

The declared settings were saved in `results/matched_inputs/protocol.json` before the new fits. This file is a run record, not a formal preregistration. The original forecast file and prepared panel are identified by SHA-256 checksums. The matched run adds 288 gradient boosting fits: three configurations, 24 calibration/test starts and four targets. Previously computed TimesFM outputs are reused exactly.

## Result

{result_sentence}

WAPE is total absolute forecast error divided by total recorded sales. Lower is better. Each method has 672 scored group/start/horizon combinations.

{markdown_table(headline)}

### Product groups in the calendar pair

{markdown_table(family)}

The original product-group ranking against the earlier boosting baseline should not be carried over to this corrected comparison. The table above is the relevant comparison when both workflows receive sales and calendar inputs.

### Product groups with all inputs

{markdown_table(full_family)}

With full inputs, boosting has lower WAPE on beverages, bread/bakery and dairy, while TimesFM has lower WAPE on grocery I. TimesFM still has lower pooled WAPE. Grocery I contributes about 44.6% of the scored quantity. This shows why the pooled ranking should be read alongside product-group results.

### History sensitivity

{sensitivity}

### Prediction intervals

{markdown_table(uncertainty)}

Coverage measures how often observations fall within the estimated middle-80% range. Higher coverage alone does not establish better uncertainty estimates. Quantile loss is mean pinball loss over all nine predicted quantiles, with lower better. Widths are saved in `paired_comparison.csv`. These pooled quantities also reflect differences in family scale. TimesFM intervals are native; boosting intervals use signed residual quantiles from the earlier calibration period.

### Inventory replay

{inventory_text}

The same initialization, seven-day review and delivery intervals, costs and limitations as the original simulation apply. The matched replay includes all six paired workflows plus the seasonal baseline. It omits capacity, expiry, purchase costs and terminal valuation.

## Reproduce and check

After preparing `data/retail_panel.csv` and obtaining the original saved forecasts:

```shell
python src/matched_comparison.py --threads 8
python src/history_sensitivity.py
python src/review_evidence.py --matched
python src/build_matched_report.py
python -m unittest discover -s tests -v
```

The correction needs the analysis dependencies and the saved original forecasts. It does not download or rerun the TimesFM checkpoint. `threadpoolctl` is a scikit-learn dependency.

`training_audit.csv` records the allowed window, label dates, example count and timing at every fit. `feature_names.json` exposes included channels. Four additional integrity tests check excluded covariates, the shared history boundary, future-target/transaction exclusion, related-series access and calibration offsets. The separate arithmetic checker covers saved scores, source targets, buffers, orders, deliveries and costs.

The next validation should use new stores or a later untouched period, consider stronger or tuned baselines under a declared budget, and test promotions and transactions separately. Matching inputs here improves the comparison; it does not establish a general retail winner.
'''
    (ROOT / 'docs/MATCHED_COMPARISON.md').write_text(doc, encoding='utf-8')
    table = lambda f: '<div class="table-wrap">' + f.to_html(index=False, border=0, escape=True) + '</div>'
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TimesFM grocery forecasting: matched-input comparison</title>
<meta name="description" content="A corrected comparison of TimesFM 3.0 and gradient boosting using the same local sales histories and input types.">
<style>:root{{--ink:#203330;--muted:#52635e;--line:#dce3dd;--green:#14745c}}*{{box-sizing:border-box}}body{{margin:0;background:#f6f7f4;color:var(--ink);font:17px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}}header,main,footer{{max-width:1000px;margin:auto;padding:28px 28px}}nav{{display:flex;gap:12px 26px;flex-wrap:wrap;font-size:14px}}a{{color:var(--green);text-underline-offset:3px}}h1{{font-size:clamp(32px,5vw,48px);line-height:1.15;letter-spacing:-.03em;max-width:830px;margin:34px 0 24px}}h2{{font-size:26px;line-height:1.25}}h3{{font-size:20px}}p{{max-width:850px}}section{{border-top:1px solid var(--line);padding:26px 0}}.summary{{background:#e5f0e8;padding:20px 24px;border-radius:6px}}.table-wrap{{overflow-x:auto;margin:22px 0}}table{{border-collapse:collapse;width:100%;font-size:15px;font-variant-numeric:tabular-nums}}th,td{{text-align:right;padding:13px 11px;border-bottom:1px solid var(--line);vertical-align:top}}th:first-child,td:first-child{{text-align:left}}th{{border-bottom-color:var(--ink)}}.note{{font-size:14px;color:var(--muted)}}.label{{font-size:12px;letter-spacing:.09em;color:var(--green);font-weight:650}}footer{{border-top:1px solid var(--line);font-size:14px}}@media(max-width:600px){{header,main,footer{{padding:22px 18px}}table{{font-size:13px}}th,td{{padding:10px 8px}}}}</style></head><body>
<header><nav><a href="https://suyashh77.github.io/suyashsapre_portfolio/">Portfolio</a><a href="{REPO}">Code and saved results</a><a href="{BLOB}docs/MATCHED_COMPARISON.md">Exact comparison methods</a></nav>
<h1>TimesFM and gradient boosting, given the same local inputs</h1>
<p class="summary"><strong>Corrected comparison:</strong> {result_sentence}</p>
<p class="note">Historical experiment · Favorita store 1, Quito · Four product groups · 14-day forecasts · Correction on previously examined test dates</p></header>
<main><section><div class="label">SITUATION AND TASK</div><h2>Does a pretrained model forecast grocery sales well?</h2>
<p>TimesFM learns patterns from many numerical sequences before it is used on a new forecasting task. Version 3.0 can forecast related series together and use known future inputs. This study checks how those forecasts compare with a conventional model fitted to earlier store sales.</p>
<p>The initial comparison gave the fitted model promotions and transactions that calendar-only TimesFM did not receive. This follow-up matches the available input types and local history windows before comparing the scores.</p></section>
<section><div class="label">ANALYSIS</div><h2>Same product groups, local history and forecast dates</h2><p>{method}</p><p>{differences}</p><p>Sales can be fractional and may understate demand during stockouts. Future promotion availability is assumed. No future realized sales or transactions are supplied.</p></section>
<section><div class="label">RESULT</div><h2>Forecast errors for the three matched pairs</h2><p>WAPE adds the size of all forecast mistakes and divides by total recorded sales. Lower is better.</p>{table(headline)}<h3>Where the calendar pair did well and poorly</h3>{table(family)}<h3>Full inputs tell a different product-level story</h3>{table(full_family)}<p>With full inputs, gradient boosting did better on three of the four groups. TimesFM did better on grocery I and had lower pooled error. Grocery I accounts for about 44.6% of the scored quantity, so the overall ranking gives that group considerable weight.</p><p class="note">These are the updated product-group results. The earlier ranking against the original boosting baseline is a different comparison.</p></section>
<section><h2>Does the available history change the comparison?</h2><p>{sensitivity}</p></section>
<section><h2>How well did the uncertainty ranges work?</h2>{table(uncertainty)}<p>Coverage describes how often actual sales fell inside the estimated middle-80% range. A wider range can cover more outcomes. Quantile loss checks all nine predicted quantiles, with lower better. Coverage, width and quantile loss should be considered together.</p><p class="note">TimesFM uses native quantiles. Gradient boosting uses offsets estimated from earlier calibration errors. The procedures differ, and the pooled scores reflect differences in product-group size.</p></section>
<section><h2>What happened in the ordering simulation?</h2><p>{inventory_text}</p><p class="note">The replay omits expiry, capacity, buying costs and terminal valuation. Costs are illustrative units.</p></section>
<section><h2>What the comparison supports</h2><p>{limitation}</p><p>The next study should test new stores and periods, compare additional inputs separately, and evaluate shortages and stock held under realistic operating constraints.</p><p><a href="{BLOB}results/matched_inputs/paired_comparison.csv">Saved paired results</a> · <a href="{BLOB}results/matched_inputs/review_audit.json">Arithmetic audit</a> · <a href="{BLOB}docs/MATCHED_COMPARISON.md">Methods and reproduction</a></p></section></main>
<footer><p>TimesFM forecasts were reused from the recorded official-checkpoint run. All matched gradient boosting models were fitted for this correction. <a href="https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/">Official TimesFM explanation</a> · <a href="https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data">Original data</a></p><p>Downloaded TimesFM 3.0 weights have a non-commercial, non-production license. This page presents saved research outputs.</p></footer></body></html>'''
    (ROOT / 'matched_report.html').write_text(html, encoding='utf-8')
    print(result_sentence)


if __name__ == '__main__':
    build()
