# TimesFM 3.0 retail forecasting study

[Live matched comparison](https://suyashh77.github.io/suyashsapre_portfolio/timesfm-retail/) · [Matched methods](docs/MATCHED_COMPARISON.md) · [Evidence review](docs/REVIEW.md)

> **Current results:** Sales + calendar gives **9.68% WAPE for TimesFM and 12.30% for gradient boosting**. The separate 365-day boosting history check scores **10.82%**. These figures supersede the original unequal-input headline.

## Situation

Retail forecasts inform stock orders. This study compares pretrained TimesFM 3.0 with a weekly-repeat rule and fitted gradient boosting on Favorita grocery sales. The initial headline used unequal inputs. The current comparison corrects that mismatch and also matches local history and related-series access.

## Task

Compare three pairs using the same raw input types, four sales groups, 256-day local history window, fourteen-day horizon and twelve weekly test starts. TimesFM weights remain unchanged; boosting is fitted using supervised examples entirely inside each local window.

## Analysis

The data covers store 1 in Quito, Ecuador, with beverages, bread/bakery, dairy and grocery I. Calibration uses twelve earlier starts, with its last target on May 16, 2017. Test starts run from May 17 through August 2. Each workflow has 672 scored group/start/horizon combinations, with overlapping dates.

Both workflows receive all four sales histories. The sales-only pair excludes other variables. The calendar pair adds weekday, annual position, applicable holidays and payday indicators. The full-input pair also receives promotion counts for all groups and historical store transactions. No future realized sales or transactions are supplied. Future promotions remain assumed known because the data lacks dated plan archives.

Boosting uses fixed 28-day summary features and per-family direct regressors, with a warm-up inside the 256-day window. Its feature representation, fitting procedure and external pretraining differ from TimesFM. Matching local inputs does not make those model properties equivalent. See [MATCHED_COMPARISON.md](docs/MATCHED_COMPARISON.md) and [the saved protocol](results/matched_inputs/protocol.json).

## Result

WAPE is total absolute forecast error divided by total recorded sales. Lower is better.

| Inputs available to both models | TimesFM WAPE | Gradient boosting WAPE |
| --- | --- | --- |
| Sales histories only | 10.16% | 11.98% |
| Sales + calendar | 9.68% | 12.30% |
| Sales + calendar + promotions + past transactions | 9.99% | 11.14% |

TimesFM has lower pooled WAPE in these three shared-256-day comparisons. Product-level results are mixed: with full inputs, boosting has lower WAPE on beverages, bread/bakery and dairy, while TimesFM scores better on grocery I. Grocery I contributes about 44.6% of the scored quantity, so that group strongly affects the pooled ranking.

For the calendar pair, middle-80% interval coverage is 87.8% for TimesFM and 83.3% for boosting. Mean nine-quantile pinball loss is 56.78 versus 72.82, with lower better. TimesFM's average range is slightly wider in this pair, and coverage alone does not establish a better uncertainty estimate. Full width and quantile metrics are saved alongside the paired results.

A separate history check keeps the corrected calendar boosting representation fixed and extends its local window to 365 days. WAPE changes from 12.30% to 10.82%. That longer-history workflow has more local observations than the TimesFM context and is reported separately. The initial workflow used a different representation, extra inputs and a year of labels. Its archived score describes a different comparison.

This is a post-hoc correction on the previously examined test dates, not a new untouched validation. It covers one store; windows overlap; sales can understate demand; future promotions are assumed known; and pretraining overlap cannot be ruled out. No significance, equivalence or general superiority claim is made. Fixed boosting settings were inherited without a new tuning search.

The inventory replay applies the same weekly ordering rule and earlier aggregate-error buffer method. Its costs are illustrative, and it does not use native daily prediction intervals to set buffers. Results are in `results/matched_inputs/inventory_metrics.csv`; they are model-plus-buffer policy outcomes rather than measured retailer savings.

## Reproduce the experiment

Use Python 3.12 in an isolated environment:

```shell
python -m pip install -r requirements.txt
python src/download_assets.py
python src/prepare_data.py --raw-dir raw
python src/experiment.py --checkpoint checkpoints/timesfm3 --fresh
python src/audit_results.py
python src/review_evidence.py
python src/matched_comparison.py --threads 8
python src/history_sensitivity.py
python src/review_evidence.py --matched
python src/build_report.py
python src/build_matched_report.py
python src/build_notebook.py
python -m unittest discover -s tests -v
```

The downloader retrieves approximately 125 MB of raw data and a 1.3 GB official checkpoint at pinned revisions and checks their hashes. The recorded run used CPU inference. `--device cuda` is optional on supported hardware. Minor numeric differences across platforms are possible.

Use `--fresh` after changing inputs or settings. Without it, the experiment reuses saved forecast caches.

### Check saved results without model inference

```shell
python -m pip install -r requirements-analysis.txt
python src/download_assets.py --skip-model
python src/prepare_data.py --raw-dir raw
python src/audit_results.py
python src/review_evidence.py
python src/review_evidence.py --matched
python -m unittest discover -s tests -v
```

Open `matched_report.html` for the current comparison. `report.html` provides interactive exploration of the current matched forecasts. The initial report is archived in `archive/initial-report.html`. Optional report-control check with Node.js:

```shell
node tests/verify_report.cjs report.html
```

## Repository contents

| Location | Purpose |
| --- | --- |
| `src/` | Data preparation, inference, scoring, inventory replay and report generation |
| `tests/` | Temporal input boundaries, calibration separation and inventory integrity |
| `docs/MATCHED_COMPARISON.md` | Current matched-input protocol, findings and remaining limits |
| `docs/METHODS.md` | Exact methods and interpretation limits |
| `docs/REVIEW.md` | Evidence checks and unresolved research questions |
| `data/` | Source manifest and dataset description |
| `results/matched_inputs/` | Current matched forecasts, scores, calibration buffers and simulation outputs |
| `results/` root files | Preserved initial unequal-input outputs; see `results/README.md` |
| `project_walkthrough.ipynb` | Executed analysis of saved results |
| `matched_report.html` | Current matched-input results |
| `report.html` | Interactive matched-input forecasts |
| `archive/initial-report.html` | Archived initial unequal-input report |

Ten integrity tests and 336 matched-report control selections pass on the recorded artifacts. A separate standard-library implementation checks 8,064 forecast rows and 2,016 inventory rows without importing the experiment's scoring functions. The matched correction checks another 9,408 forecast rows and 2,352 inventory rows; see [matched review_audit.json](results/matched_inputs/review_audit.json). Browser layout verification was unavailable. These checks validate saved calculations within their stated scope, rather than every research assumption.

The implementation was developed with an AI coding assistant. Saved model predictions came from executing the official checkpoint. The notebook executes analysis of those saved outputs; it does not rerun inference.

## Sources and usage terms

- [Official TimesFM source](https://github.com/google-research/timesfm), [3.0 release explanation](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/), and [checkpoint](https://huggingface.co/google/timesfm-3.0-pytorch).
- [Original Favorita data](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data), retrieved from a [pinned public mirror](https://huggingface.co/datasets/t4tiana/store-sales-time-series-forecasting).

Downloaded TimesFM 3.0 weights are restricted to non-commercial, non-production use. Original dataset terms also apply. Raw data, full processed panels and model weights are excluded from GitHub. Review [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before reproduction or reuse.
