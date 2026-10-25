# Results index

The **current comparison** is in [`matched_inputs/`](matched_inputs/). Start with:

- [Paired forecast errors](matched_inputs/paired_comparison.csv): calendar TimesFM 9.68% WAPE versus calendar boosting 12.30%.
- [Product-group comparisons](matched_inputs/paired_comparison_by_family.csv).
- [Uncertainty scores](matched_inputs/forecast_metrics.csv): calendar-pair coverage 87.8% versus 83.3%, and pinball loss 56.78 versus 72.82.
- [Longer-history check](matched_inputs/history_sensitivity/forecast_metrics.csv): calendar boosting with 365 local days scores 10.82% WAPE.
- [Arithmetic checks](matched_inputs/review_audit.json).

CSV and JSON files directly in this directory preserve the **initial unequal-input experiment**. Those historical scores are not the current matched comparison. They remain unchanged so the original checkpoint outputs and correction can be audited.
