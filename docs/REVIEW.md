# Evidence review: October 25, 2026

## Situation

The project compares an already trained TimesFM 3.0 checkpoint with a weekly rule and a fitted gradient boosting model on grocery sales. The documentation, dashboard and saved results should agree. This review checks that agreement and the calculations behind it.

## Task

Check the reported numbers, the time order of the inputs, the stock simulation and the interpretation. Make the evidence available for someone else to inspect. This is an implementation review of an exploratory study, rather than independent peer review.

## Analysis

The existing audit reruns the experiment's scoring and inventory functions. A second checker, `src/review_evidence.py`, uses Python's standard library and does not import those functions. It checks 8,064 saved forecast rows against the prepared sales panel and verifies the recorded target dates. It recomputes overall WAPE, MAE, bias, interval coverage, mean width and nine-quantile pinball loss, plus product-group WAPE. It verifies the weekly baseline and confirms that baseline uncertainty offsets use only earlier calibration errors.

The second checker also reconstructs the twelve-window aggregate-error buffers and follows all 2,016 inventory rows. It verifies orders against the stated rule, deliveries against orders made seven days earlier, stock and demand balances, default costs, totals and fill rates. This arithmetic check is separate from the original implementation, but it shares the stated research design and prepared inputs. It cannot establish that those assumptions describe a particular retailer.

Six existing integrity tests check causal filling, future sales/transaction exclusion, calibration separation, saved output structure, buffer independence from later outcomes and daily uncertainty estimates, and inventory balances. The report-control harness checks 288 forecast selections and two cost-slider rankings. These are logic checks, rather than browser layout checks. The official model's inference was executed in the original recorded run; this review does not claim a new full checkpoint rerun.

Code inspection confirmed that gradient boosting's training-label mask ends before each forecast start. Its input history ends before the corresponding training example's forecast start. Neither workflow supplies future realized transaction counts. Promotions and calendars remain conditional on the documented availability assumptions.

## Result

The saved arithmetic checks and six integrity tests pass. The headline WAPE values remain 9.68% for calendar-aware TimesFM, 12.56% for the weekly rule and 9.69% for gradient boosting. The documented product-group ranking agrees with the saved scores. The supplied-input bundle scores 9.99%, so the claim that it did not improve the calendar configuration is supported for this run.

Calendar TimesFM's middle-80% interval covers 87.8% of outcomes, compared with 86.2% for gradient boosting. Its average width is 573.67 source quantity units, compared with 615.68. Mean nine-quantile pinball loss, a score penalizing quantile errors, is 56.78 versus 56.67, with lower better. These numbers provide no clear overall uncertainty winner. Pooled widths and losses also reflect the different product-group scales. Requesting prediction intervals does not itself improve a model's point forecasts.

The documentation defines WAPE and identifies the unequal inputs. The long account uses Situation, Task, Analysis and Result. The saved checker output is `results/review_audit.json`.

The main remaining limits are substantive:

- One selected store and four groups do not establish broad retail performance. Twelve overlapping test windows provide limited evidence, and no significance or equivalence test was conducted.
- Gradient boosting received promotions and transaction summaries absent from calendar-only TimesFM. The observed comparison cannot isolate the forecasting architecture. History lengths and feature representations also differ.
- Promotions and past transactions change together in the full TimesFM configuration. Their individual contributions and the reason for product-group differences remain untested.
- Sales can understate demand during stockouts. Future promotion plans have no dated archive, calendar publication dates were not audited, and public data may have appeared in pretraining.
- The inventory replay uses illustrative costs and a small earlier sample for extra stock. It omits expiry, capacity and terminal valuation. Its buffers use aggregate point-forecast errors, so its result does not demonstrate the value of native TimesFM intervals.
- The calendar configuration was the strongest TimesFM configuration on this test period. It has not been validated as a selected winner on a second untouched holdout.

The next study should match raw input sets, vary the added inputs separately, document the history budget and repeat on new stores and periods. Inventory evaluation should use the same ordering rule and realistic constraints, and compare extra-stock methods directly if the question is whether prediction intervals help.
