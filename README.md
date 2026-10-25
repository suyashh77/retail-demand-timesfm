# TimesFM 3.0 retail forecasting study

[Live results](https://suyashh77.github.io/suyashsapre_portfolio/timesfm-retail/forecasts.html) · [Methods](docs/METHODS.md) · [Evidence review](docs/REVIEW.md)

## Situation

Retail forecasts inform stock orders. This repository evaluates whether the pretrained TimesFM 3.0 checkpoint provides useful grocery-sales forecasts without task-specific fitting. The comparison includes a weekly-repeat rule and gradient boosting fitted to earlier sales.

## Task

Compare forecast error, uncertainty estimates and a conditional inventory replay. Test whether forecasting related sales groups together, adding calendar inputs, and then adding promotions and past transactions improves the recorded forecasts.

## Analysis

| Element | Setup |
| --- | --- |
| Data | Favorita Store Sales; store 1, Quito, Ecuador |
| Targets | Beverages, bread/bakery, dairy and grocery I |
| Prepared period | January 2015 through August 2017 |
| TimesFM history and horizon | 256 daily observations to predict 14 days |
| Earlier calibration | 12 weekly starts; final target May 16, 2017 |
| Held-out evaluation | 12 weekly starts, May 17 through August 2, 2017 |
| Scored combinations | 672 per configuration, with overlapping target dates |
| Comparators | Previous week repeated twice; per-family gradient boosting |
| TimesFM configurations | Separate targets; joint targets; joint plus calendar; joint plus calendar, promotions and past transactions |
| Inventory replay | Weekly review, seven-day lead time, buffers from earlier 14-day aggregate errors |

TimesFM weights remain unchanged. Gradient boosting is refitted at each forecast start using labels dated before that start. Future realized sales and transaction counts are excluded from input features. Future promotions are assumed planned because the data lacks dated plan archives. Calendar publication dates were not audited.

Calendar inputs describe weekday, annual position, applicable holidays and payday indicators. Gradient boosting also receives promotion and transaction inputs absent from calendar-only TimesFM. The full-input TimesFM workflow receives the same raw input types, with a different representation. History budgets differ, so the comparison does not isolate model architecture.

Exact revisions and checksums are in [source_manifest.json](data/source_manifest.json). See [METHODS.md](docs/METHODS.md) for feature construction, calibration, metrics, simulator assumptions and claim boundaries.

## Result

WAPE is total absolute forecast error divided by total recorded sales. Lower is better.

| Configuration | WAPE |
| --- | ---: |
| Repeat the previous week | 12.56% |
| Gradient boosting | 9.69% |
| TimesFM, separate targets | 10.20% |
| TimesFM, joint targets | 10.16% |
| TimesFM, joint targets plus calendar | 9.68% |
| TimesFM, calendar plus promotions and past transactions | 9.99% |

Calendar-aware TimesFM and gradient boosting had very close overall scores. TimesFM scored better on beverages and bread/bakery, while gradient boosting scored better on dairy and grocery I. Adding promotions and transactions together did not improve the calendar-only configuration. Their individual effects remain untested.

The calendar TimesFM middle-80% range covered 87.8% of outcomes, versus 86.2% for gradient boosting. Its average range was narrower, while the nine-quantile pinball score was almost tied. Coverage alone does not identify the better uncertainty estimate.

The joint-target TimesFM workflow had the lowest default simulated inventory cost, 20.4% below the weekly baseline. This result uses illustrative costs and earlier aggregate point-forecast errors to set buffers. It is not measured retailer savings or evidence that native daily prediction intervals improved orders.

This is an exploratory, single-store historical experiment. Forecast windows overlap, no significance or equivalence test was conducted, and public data may overlap pretraining. Recorded sales can understate demand during stockouts. The inventory replay omits expiry, capacity, purchase costs and terminal valuation. The calendar configuration has not been validated as a selected winner on a second holdout.

## Reproduce the experiment

Use Python 3.12 in an isolated environment:

```shell
python -m pip install -r requirements.txt
python src/download_assets.py
python src/prepare_data.py --raw-dir raw
python src/experiment.py --checkpoint checkpoints/timesfm3 --fresh
python src/audit_results.py
python src/review_evidence.py
python src/build_report.py
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
python -m unittest discover -s tests -v
```

Open `report.html` to explore saved forecasts offline. Optional report-control check with Node.js:

```shell
node tests/verify_report.cjs report.html
```

## Repository contents

| Location | Purpose |
| --- | --- |
| `src/` | Data preparation, inference, scoring, inventory replay and report generation |
| `tests/` | Temporal input boundaries, calibration separation and inventory integrity |
| `docs/METHODS.md` | Exact methods and interpretation limits |
| `docs/REVIEW.md` | Evidence checks and unresolved research questions |
| `data/` | Source manifest and dataset description |
| `results/` | Saved forecasts, scores, calibration buffers, simulation outputs and figures |
| `project_walkthrough.ipynb` | Executed analysis of saved results |
| `report.html` | Self-contained exploration of saved results |

Six integrity tests and 288 report-control selections pass on the recorded artifacts. A separate standard-library implementation checks 8,064 forecast rows and 2,016 inventory rows without importing the experiment's scoring functions. See [review_audit.json](results/review_audit.json). Browser layout verification was unavailable. These checks validate saved calculations within their stated scope, rather than every research assumption.

The implementation was developed with an AI coding assistant. Saved model predictions came from executing the official checkpoint. The notebook executes analysis of those saved outputs; it does not rerun inference.

## Sources and usage terms

- [Official TimesFM source](https://github.com/google-research/timesfm), [3.0 release explanation](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/), and [checkpoint](https://huggingface.co/google/timesfm-3.0-pytorch).
- [Original Favorita data](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data), retrieved from a [pinned public mirror](https://huggingface.co/datasets/t4tiana/store-sales-time-series-forecasting).

Downloaded TimesFM 3.0 weights are restricted to non-commercial, non-production use. Original dataset terms also apply. Raw data, full processed panels and model weights are excluded from GitHub. Review [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before reproduction or reuse.
