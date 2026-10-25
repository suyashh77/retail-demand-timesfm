# Matched-input correction

## Situation

The initial headline compared calendar-only TimesFM with gradient boosting that also received promotions and transactions. The workflows also differed in local history and related-series access. That comparison could not isolate differences between forecasting methods.

## Task

Compare three pairs with matching raw input types, product-group access, local history windows, forecast dates and horizon. Preserve the initial outputs as a separate exploratory stage.

## Analysis

Both workflows have access to the same four product groups and the same latest 256 daily observations. They predict the same fourteen days at the same twelve weekly test starts. The sales-only pair excludes calendar, promotions and transactions. The calendar pair adds only calendar signals. The full-input pair adds the four promotion series and historical store transactions to both workflows. Future realized transactions and sales are excluded.

TimesFM uses the saved official checkpoint forecasts, whose inputs have not changed. Its weights remain unchanged. Each gradient boosting model is fitted separately per target and forecast start, using supervised examples drawn entirely from that 256-day window. Each example includes sales summaries for all four groups, a horizon indicator and only the allowed covariates. The 28-day feature warm-up is inside the shared window, leaving fewer labeled training days than 256. Matching available local data does not equate the feature representation, external pretraining, model capacity or fitting procedure.

Gradient boosting uses `max_iter=80`, `max_leaf_nodes=15`, `learning_rate=0.07`, `l2_regularization=10`, `random_state=42`, and `early_stopping=False`. Historical training origins begin after the in-window 28-day warm-up. Labels end before the current forecast start. Missing Christmas labels are excluded; context filling follows the original causal procedure.

Calendar features describe weekday, annual position, applicable holidays and payday indicators. Promotions remain an assumed-known scenario because dated plan versions are unavailable. The full-input comparison changes promotions and transactions together and cannot isolate them.

This correction was designed after the original results had been examined. It reuses the same historical test period and does not provide a new untouched validation. There is one store, overlapping forecast windows, assumed-known promotions and possible pretraining overlap. No significance or statistical equivalence claim is made. The fixed boosting settings were inherited from the original run, without a new parameter search.

The declared settings were saved in `results/matched_inputs/protocol.json` before the new fits. This file is a run record, not a formal preregistration. The original forecast file and prepared panel are identified by SHA-256 checksums. The matched run adds 288 gradient boosting fits: three configurations, 24 calibration/test starts and four targets. Previously computed TimesFM outputs are reused exactly.

## Result

With sales and calendar inputs, TimesFM scored 9.68% WAPE and gradient boosting scored 12.30%. This replaces the earlier headline comparison with unequal inputs.

WAPE is total absolute forecast error divided by total recorded sales. Lower is better. Each method has 672 scored group/start/horizon combinations.

| Inputs available to both models | TimesFM WAPE | Gradient boosting WAPE |
| --- | --- | --- |
| Sales histories only | 10.16% | 11.98% |
| Sales + calendar | 9.68% | 12.30% |
| Sales + calendar + promotions + past transactions | 9.99% | 11.14% |

### Product groups in the calendar pair

| Product group, calendar pair | TimesFM WAPE | Gradient boosting WAPE |
| --- | --- | --- |
| BEVERAGES | 8.53% | 9.43% |
| BREAD/BAKERY | 9.74% | 10.04% |
| DAIRY | 9.26% | 9.05% |
| GROCERY I | 10.74% | 15.89% |

The original product-group ranking against the earlier boosting baseline should not be carried over to this corrected comparison. The table above is the relevant comparison when both workflows receive sales and calendar inputs.

### Product groups with all inputs

| Product group, full-input pair | TimesFM WAPE | Gradient boosting WAPE |
| --- | --- | --- |
| BEVERAGES | 8.59% | 8.41% |
| BREAD/BAKERY | 10.31% | 9.34% |
| DAIRY | 9.90% | 8.74% |
| GROCERY I | 11.16% | 14.32% |

With full inputs, boosting has lower WAPE on beverages, bread/bakery and dairy, while TimesFM has lower WAPE on grocery I. TimesFM still has lower pooled WAPE. Grocery I contributes about 44.6% of the scored quantity. This shows why the pooled ranking should be read alongside product-group results.

### History sensitivity

With the corrected calendar boosting representation kept fixed, extending its local window from 256 to 365 days changed WAPE from 12.30% to 10.82%. The latter has more local history than the TimesFM context and is a separate sensitivity check. The original boosting workflow used a different, own-group feature representation, a year of labels and extra covariates. Changes from that old score cannot be attributed solely to removing covariates. See results/matched_inputs/history_sensitivity/ for the forecasts and declared settings.

### Prediction intervals

| Shared inputs | TimesFM 80% coverage | Boosting 80% coverage | TimesFM quantile loss | Boosting quantile loss |
| --- | --- | --- | --- | --- |
| Sales histories only | 88.7% | 87.2% | 60.10 | 70.13 |
| Sales + calendar | 87.8% | 83.3% | 56.78 | 72.82 |
| Sales + calendar + promotions + past transactions | 85.1% | 81.5% | 58.98 | 68.24 |

Coverage measures how often observations fall within the estimated middle-80% range. Higher coverage alone does not establish better uncertainty estimates. Quantile loss is mean pinball loss over all nine predicted quantiles, with lower better. Widths are saved in `paired_comparison.csv`. These pooled quantities also reflect differences in family scale. TimesFM intervals are native; boosting intervals use signed residual quantiles from the earlier calibration period.

### Inventory replay

Under the same illustrative weekly-order rule, calendar TimesFM filled 99.47% of the recorded sales quantity and calendar gradient boosting filled 99.65%. Default replay costs were 24,542 and 27,466 illustrative units respectively. Buffers differ by workflow because each is estimated from that workflow's earlier aggregate errors. This is a model-plus-buffer policy comparison, not measured retailer savings. Native daily prediction intervals do not set these buffers.

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
