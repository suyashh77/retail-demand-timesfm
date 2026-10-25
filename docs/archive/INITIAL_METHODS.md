# Initial experiment: methods and claim boundaries

The current matched-input correction is documented in [MATCHED_COMPARISON.md](../MATCHED_COMPARISON.md). The methods below describe the original unequal-input exploration, whose outputs are preserved.

This is an exploratory, retrospective case study. It is not a randomized study, a registered benchmark, a test of all retail forecasting methods, or evidence of actual financial savings.

## What was run

One fixed store and four fixed product families were evaluated with six fixed configurations. No TimesFM fine-tuning or per-task optimization was performed. The primary metric and input configurations were encoded before the saved inference run. There was no formal preregistration. Test-set rankings are reported retrospectively, and a separate second holdout has not validated selection of a winning configuration.

| Element | Recorded setup |
| --- | --- |
| Store | Favorita store 1, Quito, Ecuador |
| Families | BEVERAGES, BREAD/BAKERY, DAIRY, GROCERY I |
| Prepared dates | 2015-01-01 through 2017-08-15 |
| Source observations | 3,824 nonmissing family/day sales values |
| Explicitly missing rows | Eight Christmas family/day rows, retained as missing in the export |
| TimesFM context | 256 daily observations per target |
| Forecast horizon | 14 days |
| Calibration starts | Twelve, weekly, 2017-02-15 through 2017-05-03 |
| Last calibration label | 2017-05-16 |
| Test starts | Twelve, weekly, 2017-05-17 through 2017-08-02 |
| Last scored target | 2017-08-15 |
| Primary scored combinations | 672 per configuration, with overlapping dates |
| Secondary scored combinations | 336 per configuration, using only forecast days 1 through 7 |
| TimesFM checkpoint | Official `google/timesfm-3.0-pytorch` |
| Runtime | Python 3.12.10, PyTorch 2.11.0+cpu, eight CPU threads, seed 42 |

Exact source-code and model revisions and SHA-256 checksums are in `data/source_manifest.json`. The code commit is `e51928e27119cb17bebc005be2696b75e0a9e688`. The model revision is `43046b85ec22d584a13f8098c2ed39c889e129c2`.

## Target and information available to the model

The target is recorded family-level sales quantity. It is not revenue, SKU-level count, recovered lost demand, or a direct measurement of all customer demand. Fractions are possible. Differences in units across families limit the economic interpretation of pooled quantity metrics.

At a forecast start, target and transaction histories end on the preceding day. Weekday, annual position, applicable holidays, and payday indicators extend into the horizon. Historical promotion counts also extend into the horizon under an explicit planned-input assumption. No timestamped plan archive exists in the source. This limits any claim of a fully point-in-time realistic promotion experiment.

There are no product prices. Traffic is proxied by transaction counts, which count purchases rather than visitors. Future realized transactions are excluded. Missing sales and promotion history are filled from the previous same weekday, sequentially. Traffic is forward-filled. Missing evaluation targets would raise an error instead of being silently filled.

The holiday feature includes national holidays and regional/local holidays applicable to Pichincha/Quito. Transferred original dates are excluded, and scheduled Transfer dates are included. Unscheduled Event rows, such as earthquake-related events, are excluded. Calendar publication timestamps were not independently audited either.

## Comparators and unequal information histories

Seasonal naive repeats the last seven days twice. Its point forecasts have no fitted parameters.

Gradient boosting uses one `HistGradientBoostingRegressor` per family, refitted at each start. It predicts all fourteen horizons using direct supervised examples with horizon as an input. Its labeled targets come from the preceding 365 days and are strictly earlier than the current start. Features use recent 28-day sales summaries, aligned weekday values, historical transaction summaries, calendar, and assumed-known promotions. Settings are `max_iter=80`, `max_leaf_nodes=15`, `learning_rate=0.07`, `l2_regularization=10`, and `random_state=42`. Other settings follow scikit-learn 1.8.0 defaults. With fewer than 10,000 training examples, default automatic early stopping is inactive.

TimesFM configurations are independent single-target forecasts, four-target forecasts, four-target forecasts plus calendar, and four-target forecasts plus calendar/promotions/past transactions. The last configuration changes promotions and past transactions together. It does not isolate their effects. All configurations use official evaluator options: `use_symmetric_averaging=False`, `make_positive=True`, `sort_quantiles=True`, `use_znorm=False`, and `padding_mode="none"`. Native targets are nonnegative. The point forecast is the sorted median quantile.

Calendar-only TimesFM omits promotions and transaction summaries that gradient boosting receives. The full-input TimesFM workflow receives the same raw input types, but represents them differently. These are comparisons of specified workflows. They are not comparisons at equal training-data volume, fitting budget, parameter count, or total computation. TimesFM has upstream pretraining, the fitted comparator uses a year of labels with recent summary features, and seasonal naive uses one week. No strong statement about training speed, energy, cost, or equal-budget efficiency follows. Recorded inference times cover only TimesFM prediction calls, excluding loading and preparation.

ARIMA, ETS, and other foundation models were not evaluated. Their absence limits broad claims about traditional forecasting or the best available model.

## Scores and dependence

For saved test predictions, WAPE is `sum(abs(actual - prediction)) / sum(actual)`. Errors are calculated separately for every family, start, and horizon before aggregation. This weights high-volume groups more heavily and counts overlapping target dates at different forecast distances. Beverages and grocery I contribute approximately 82.2% of the total scored quantity. `metrics_by_family.csv` exposes the family differences.

The secondary first-week score omits horizons 8 through 14. It avoids repeating target dates across starts but does not make adjacent sales observations independent. `metrics_by_forecast_start.csv` and `paired_comparison_by_start.csv` expose differences over time. No p-values, population confidence intervals, significance claims, or statistical equivalence claims are reported. Twelve starts and one store provide limited evidence about broader performance.

Macro MASE averages family-level MAE divided by each family's mean absolute seasonal lag-7 difference in pre-test observations. The scale excludes missing source values. Coverage is the proportion of targets within the 10th-to-90th percentile band, including endpoints. Average pinball loss is computed over all nine quantiles and saved rows. Interval width and pinball loss are in source quantity units, so they also reflect family scale.

TimesFM intervals are native outputs after the official sorting and nonnegative options. Conventional intervals add empirical signed calibration residual quantiles to the fitted point forecast, with negative lower values clamped to zero. Baseline q50 can therefore differ from its original fitted point forecast. Calibration-row baseline intervals are not independently evaluated. Test intervals use only pre-test residuals. The two interval-generation procedures differ and should not be described as equally calibrated methods.

## Inventory replay

The replay uses recorded sales as a demand proxy for 84 days, 2017-05-17 through 2017-08-08. Review occurs every seven days and delivery lead time is seven days. On each day, deliveries arrive, any scheduled order is placed, demand is served, and a holding charge applies to ending inventory. Unmet sales are lost, rather than backordered.

For each model/family, protection-period forecast is the sum of fourteen point forecasts. The buffer is `max(0, quantile(aggregate_signed_error, 0.9, method="higher"))` over twelve earlier calibration starts, where aggregate signed error is the fourteen-day actual total minus the fourteen-day point-forecast total. Buffers stay fixed during testing. Orders are `max(0, forecast_total + buffer - on_hand - outstanding_orders)`.

This empirically estimates total-period error. It does not sum daily upper quantiles, infer a joint distribution from marginal quantiles, use native TimesFM probability forecasts to set the buffer, or claim a conformal guarantee. Calibration horizons overlap and the buffer sample is small.

All models begin with the same on-hand quantity per family, equal to twice that family's preceding seven days of observed sales. The initial pipeline is empty. No warm-up is discarded. Initial stock can materially affect results and has not been varied in this experiment.

Default cost is `0.01 * total_ending_inventory + 1 * total_unmet_sales`. The values are illustrative cost units. Family quantities differ, so the pooled cost does not estimate actual money. There are no buying charges, order capacity, case-pack restrictions, expiry, or terminal stock/pipeline valuations. At a holding charge of 0.05, the preferred replay changes to the calendar configuration. The default 20.4% reduction is conditional on the specified policy, starting stock, test window, demand proxy, and cost assumptions.

Fill rate is total served quantity divided by total recorded demand-proxy quantity. It is distinct from the fraction of days without a shortage and from cycle service level. Simulated lost sales are not measured Favorita lost sales.

## Audit trail and provenance

`src/audit_results.py` recomputes the scored summaries and inventory replay from saved forecasts and checks them against published result files. `src/review_evidence.py` supplies a separate standard-library check of saved scores, source targets, calibration offsets, stock orders, arrivals and costs without importing the experiment functions. Its scope and remaining limits are in `docs/REVIEW.md`. Six integrity tests cover future-sales/traffic perturbations, causal history filling, calibration separation, unique nonnegative predictions and ordered quantiles, buffer independence from test labels and marginal quantiles, and daily stock balance.

The implementation and editorial drafting were developed with an AI coding assistant. Model predictions were produced by executing the official checkpoint, rather than invented as example values. The code, saved outputs, package versions, and checks are provided for independent review. Passing these tests does not validate every design assumption or prove all implementation details correct.

The original ICML paper concerns the original TimesFM model. This project does not present that paper as peer review of the specific TimesFM 3.0 release or of this experiment. Version-specific capability claims cite the official release documentation.

Public data could overlap upstream pretraining. The model card describes exclusions, but full training lineage was not independently inspected. The only warranted zero-shot claim is that no task-specific TimesFM fitting was performed here.

## Claim boundaries

| Supported by this run | Unsupported extension |
| --- | --- |
| Calendar TimesFM scored 9.68% versus seasonal naive's 12.56% | TimesFM beats every traditional method |
| Gradient boosting scored 9.69% in this setup | The methods are statistically equivalent |
| The final input bundle scored worse than calendar only | Promotions are useless or harmed sales |
| Default multivariate replay cost was 20.4% lower than seasonal naive | Favorita saved 20.4%, or a deployable policy is ready |
| Daily native intervals were evaluated | Those intervals directly generated the inventory improvement |
| Forecast windows follow time and hide later sales/traffic | Every promotion/calendar input is verified to have been available in real time |
