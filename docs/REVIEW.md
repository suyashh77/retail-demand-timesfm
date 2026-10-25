# Current matched-input evidence review

## Situation

The initial comparison used unequal inputs. Current results use three pairs with matching raw input types, four sales histories and a 256-day local history window. The original narrative reviews are retained under `docs/archive/`.

## Task

Check the corrected forecast scores and inventory replay against the saved observations and predictions, and make the current results consistent across documentation and reports.

## Analysis

The separate standard-library arithmetic checker verifies 9,408 forecast rows and 2,352 inventory rows without importing the experiment's scoring functions. It checks source targets, calibration offsets, aggregate buffers, orders, deliveries and costs. Ten integrity tests pass, including excluded-input and history-boundary checks. These checks support the recorded computations, rather than every research assumption.

## Result

With sales and calendar inputs, TimesFM scored 9.68% WAPE and gradient boosting scored 12.30%. This replaces the earlier headline comparison with unequal inputs.

| Inputs available to both models | TimesFM WAPE | Gradient boosting WAPE |
| --- | --- | --- |
| Sales histories only | 10.16% | 11.98% |
| Sales + calendar | 9.68% | 12.30% |
| Sales + calendar + promotions + past transactions | 9.99% | 11.14% |

Calendar-pair middle-80% coverage is 87.8% for TimesFM and 83.3% for boosting. Pinball loss is 56.78 versus 72.82. TimesFM's average range is slightly wider. With full inputs, boosting has lower WAPE on beverages, bread/bakery and dairy; TimesFM has lower WAPE on grocery I and lower pooled WAPE.

With the corrected calendar boosting representation kept fixed, extending its local window from 256 to 365 days changed WAPE from 12.30% to 10.82%. The latter has more local history than the TimesFM context and is a separate sensitivity check. The original boosting workflow used a different, own-group feature representation, a year of labels and extra covariates. Changes from that old score cannot be attributed solely to removing covariates. See results/matched_inputs/history_sensitivity/ for the forecasts and declared settings.

This correction was designed after the original results had been examined. It reuses the same historical test period and does not provide a new untouched validation. There is one store, overlapping forecast windows, assumed-known promotions and possible pretraining overlap. No significance or statistical equivalence claim is made. The fixed boosting settings were inherited from the original run, without a new parameter search.

See [current methods](MATCHED_COMPARISON.md) and [saved audit](../results/matched_inputs/review_audit.json). The archived initial review is [here](archive/INITIAL_REVIEW.md).
