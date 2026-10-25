# Testing TimesFM 3.0 on grocery sales

## Situation: I wanted to test the model's promise on a real problem

Google describes TimesFM as a model that can forecast a sequence of numbers using patterns learned from many other sequences. Give it daily sales, for example, and it predicts the sales that come next. The original TimesFM research appeared at ICML in 2024. This project uses the later TimesFM 3.0 release, which also accepts related sales histories and extra information about the past or future. [1, 2]

The attraction is practical. A retailer can try an already trained model without first training a new forecasting model for that retailer's data. Google calls this **zero-shot forecasting**. The model still needs useful input data, and its forecasts still need to be checked.

Retail demand was a useful place to investigate this. A store has to order stock before it knows how much customers will buy. Ordering too little can leave shelves empty. Ordering too much ties up stock and can increase waste. Sales also change with the day of the week, holidays, and promotions.

TimesFM can also be tested on other quantities recorded over time, such as electricity use, website traffic, or workload. Its usefulness has to be established for each application. Predicting a stable weekly sales pattern and predicting an unexpected disruption are different challenges.

I wanted to see how the model behaved on actual grocery sales and whether useful forecasts led to sensible inventory decisions.

## Task: compare it with understandable alternatives

The aim was to compare TimesFM with a simple weekly forecast and a conventional machine-learning model, then check whether adding information helped. I also wanted to follow the forecasts into a small inventory simulation.

There are several established ways to forecast retail sales. A store might begin by expecting next Wednesday to resemble last Wednesday. Exponential smoothing estimates a changing sales level and seasonal pattern. ARIMA uses relationships between earlier observations and their changes. Regression and tree-based models can learn from inputs such as recent sales, weekday, and promotion plans. These approaches fit their patterns to the available business data. [3]

This project tests two alternatives: the weekly-repeat rule and gradient boosting, a machine-learning method built from decision trees. ARIMA and exponential smoothing provide background context here. They were not run, so this experiment cannot establish that TimesFM beats them.

The main difference in this comparison is how the models obtain their patterns. The gradient boosting model learns from this retail dataset. TimesFM arrives with patterns learned during Google's earlier training. In this project, its model weights remain unchanged.

## Action: use real sales, hide the answers, and compare the predictions

### The data

I used the public Corporacion Favorita Store Sales dataset from Ecuador. The original competition data covers many stores and product families. I chose store 1 in Quito and four groups: beverages, bread/bakery, dairy, and grocery I. [4]

These are groups of products, rather than individual products. The sales field measures quantity and can contain fractions because some items are sold by weight. It does not measure revenue. Recorded sales may also understate what customers wanted if the store ran out of stock.

The prepared panel covers January 2015 to August 2017. It contains 3,824 recorded sales values. Eight missing Christmas entries remain marked as missing in the exported data. Where a missing value falls inside a model's history, the input preparation uses the previous same weekday. It does not borrow a later observation to fill the gap.

The extra information includes daily store transaction counts, a calendar, and the number of promoted items in each group. Transaction counts are a rough indicator of store activity, rather than a direct count of visitors. The data contains no product prices.

Promotion records need special care. The dataset says what was promoted, but it does not preserve dated versions of the promotion plans. The experiment therefore **assumes future promotions were already planned and known** when making the forecast. A real deployment would need to verify that assumption. Actual future transaction counts are never supplied.

### The methods

The weekly-repeat method copies the latest seven observed days into the next two weeks. It never uses the actual first forecast week to construct the second.

Gradient boosting is fitted separately for each product group at every forecast start. It uses earlier labeled examples from the preceding year. Its inputs describe recent sales, patterns over the last four weeks, historical transactions, calendar information, and the assumed-known promotions. Its settings are fixed throughout this experiment. [5]

TimesFM receives the latest 256 days of sales and predicts the next 14 days. I tested four versions of its inputs. One forecasts each group separately. Another forecasts the four groups together. The third adds calendar information. The fourth also adds promotion counts and historical transactions. "Together" means the model can consider the other groups' histories while predicting each group. It still produces a separate forecast for each one.

TimesFM also returns a range of possible outcomes for each future day. In the charts, the shaded band runs from its estimated 10th percentile to its estimated 90th percentile. That is intended to contain the middle 80% of outcomes for that day. It is not a promise that every observation will fall inside it.

### The test

The experiment follows time in order. On May 17, 2017, for example, sales and transaction inputs stop on May 16. The models predict May 17 through May 30. A week later, the next forecast can use the observations that have become available in the meantime.

There are twelve weekly test starts, from May 17 to August 2. Their forecasts extend through August 15. Because each prediction covers two weeks and starts one week apart, some target dates are scored twice at different forecast distances. The 672 scored group/day/forecast-distance combinations per method are therefore not 672 independent experiments. A separate score checks only the first forecast week, giving 336 observations without overlapping target dates.

Before testing, twelve earlier forecast starts were used to estimate the baselines' forecast ranges and the extra stock in the inventory policy. The last of those earlier target dates is May 16. This earlier calculation does not train or fine-tune TimesFM.

The primary score is **total absolute forecast error divided by total recorded sales**, usually called WAPE. For example, total absolute mistakes of 1,000 against total recorded sales of 10,000 would give a 10% score. That example explains the calculation. It does not mean every individual prediction is within 10% of the answer.

## Result: useful forecasts, with several limits

### Calendar information helped; the full set of inputs did not

| Method | Forecast error, lower is better |
| --- | ---: |
| Repeat the previous week | 12.56% |
| Gradient boosting, fitted to earlier retail data | 9.69% |
| TimesFM, each group separately | 10.20% |
| TimesFM, groups together | 10.16% |
| TimesFM, groups together with calendar | 9.68% |
| TimesFM, calendar, promotions, and past transactions | 9.99% |

The calendar version of TimesFM reduced error by 22.9% relative to repeating the previous week. The gradient boosting result was almost the same as TimesFM's result. Their overall scores differ by about 0.01 percentage points. That small observed difference does not establish a reliable advantage, and no statistical equivalence test was conducted.

TimesFM with calendar information had lower error than gradient boosting at seven of the twelve forecast starts. The difference changed direction across the test. It would be misleading to describe the two models as performing identically every week.

Forecasting the groups together produced only a small improvement over forecasting them separately. Adding calendar information helped further. The final version, with promotions and past transactions, had higher error than the calendar version. That final comparison changes two sets of inputs at once. It cannot tell us which input caused the difference or why.

### The strongest method depended on the product group

| Product group | TimesFM with calendar | Gradient boosting |
| --- | ---: | ---: |
| Beverages | 8.53% | 9.60% |
| Bread/bakery | 9.74% | 12.66% |
| Dairy | 9.26% | 7.94% |
| Grocery I | 10.74% | 9.83% |

TimesFM did better on beverages and bread/bakery. Gradient boosting did better on dairy and grocery I. The overall score is also affected by volume: beverages and grocery I account for about 82% of the recorded quantity in the scored forecasts. A reader should check the group-level results before treating the overall ranking as an answer for every product.

There were large individual misses. In the forecast made on August 2, calendar-aware TimesFM predicted about 2,120 beverage sales units for August 11. The recorded value was 1,006. Its estimated middle-80% range was about 1,595 to 2,601, so the observation fell below that range. This is a documented failure case. This analysis does not identify its cause.

Across all test forecasts, that model's middle-80% band contained about 87.8% of recorded observations. The range was somewhat conservative on this sample. Coverage needs to be considered alongside the width of the range and the probability-forecast scores saved in the results.

### Inventory results answered a different question

I then simulated a store that orders once a week and receives deliveries seven days later. Each policy uses its next fourteen days of point forecasts plus extra stock estimated from earlier fourteen-day forecast mistakes. All policies start with the same inventory.

The extra-stock calculation uses errors in the **total sales over fourteen days**. It does not add fourteen daily upper forecasts and call their sum a fourteen-day percentile. TimesFM's daily probability forecasts are evaluated separately. They are not the direct basis of the inventory buffer in this experiment.

Under a holding charge of 0.01 illustrative cost units per unit of stock per day and an unmet-sales penalty of 1 unit, TimesFM with the groups together had the lowest simulated cost. It was 20.4% below the weekly-repeat policy. It fulfilled 99.73% of the recorded sales quantity, compared with 98.61% for that baseline.

These are simulated outcomes. They are not measured savings for Favorita. The policy uses recorded sales as a substitute for demand and leaves out buying costs, delivery capacity, expiry, and the value of stock at the end. Product-group quantities are also not financially interchangeable. When the holding charge rises from 0.01 to 0.05, the calendar version becomes the lowest-cost policy among the tested methods. The cost assumption matters.

The model with the lowest forecast error was therefore different from the model with the lowest cost in the default simulation. The size of earlier total-sales mistakes affects the safety stock, which changes the inventory result.

### What I would claim from this experiment

On this store and test period, pretrained TimesFM 3.0 produced forecasts close to a fitted gradient boosting baseline, with useful improvement over the weekly-repeat rule. Calendar information helped. Supplying more inputs did not improve the best TimesFM result.

That is evidence for testing the model as a forecasting candidate. It does not establish superiority across retail, a causal effect of calendar or promotions, or readiness for production. The public dataset could overlap the model's earlier training data, whose full history cannot be independently audited here. "Zero-shot" describes the absence of task-specific fitting in this project.

I would next repeat the comparison on other stores and a new, untouched time period, add seasonal statistical models, and use timestamped promotion plans. For inventory, the next study should check the extra-stock calculation over a longer history and include the constraints of a particular retailer.

## Sources and reproducibility

The repository contains the experiment code, saved predictions and scores, a notebook, and tests for future-input handling and inventory balance. `docs/METHODS.md` records the exact setup. `results/academic_audit.json` confirms that the published scores were recomputed from the saved predictions and inventory replay.

1. Das, Kong, Sen, and Zhou. [A decoder-only foundation model for time-series forecasting](https://proceedings.mlr.press/v235/das24c.html). ICML, 2024. This is the original TimesFM paper, not a peer-reviewed report of the specific 3.0 release.
2. Google Research. [TimesFM 3 release explanation](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/), [official source](https://github.com/google-research/timesfm), and [3.0 checkpoint](https://huggingface.co/google/timesfm-3.0-pytorch). Release documentation supports the version-specific capabilities described here.
3. Hyndman and Athanasopoulos. [Forecasting: Principles and Practice](https://otexts.com/fpp3/). Background on seasonal forecasts, exponential smoothing, ARIMA, and evaluation over time.
4. Corporacion Favorita. [Store Sales competition data](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data), accessed through a [public mirror](https://huggingface.co/datasets/t4tiana/store-sales-time-series-forecasting). Immutable download revisions and checksums are recorded in the project.
5. Scikit-learn. [Histogram-based gradient boosting](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html). The recorded run used version 1.8.0.

The downloadable TimesFM 3.0 weights have a non-commercial, non-production license. The original data has separate terms. Raw data and weights are excluded from the GitHub repository. The code downloads them from their sources for reproduction.


---

## Read or reproduce the project

Open `portfolio_report.html` for the plain-language dashboard. It works offline and explores saved forecasts. The full explanation above follows Situation, Task, Action, and Result. The short interview answer and LinkedIn draft are in `docs/`.

### Files to start with

| File | What it explains |
| --- | --- |
| `docs/PROJECT_STORY.md` | The full project in plain language |
| `docs/METHODS.md` | Exact methods, assumptions, and limits of the claims |
| `docs/INTERVIEW_ANSWER.md` | A short STAR answer |
| `docs/LINKEDIN_POST.md` | A short post inviting feedback |
| `project_walkthrough.ipynb` | Executed checks and analysis of saved outputs |
| `results/forecast_metrics.csv` | Forecast scores for all six configurations |
| `results/metrics_by_family.csv` | Where each method did well or poorly |
| `results/paired_comparison_by_start.csv` | How the near-tie varied across forecast starts |
| `results/inventory_metrics.csv` | Simulated costs, shortages, and fulfilled quantity |
| `results/academic_audit.json` | Checks of the published figures and a documented failure case |
| `data/source_manifest.json` | Exact data/model versions and file checksums |
| `src/` and `tests/` | All project code and integrity tests |

### Reproduce the measured run

Use Python 3.12. From this repository, create and activate an isolated environment, then run:

```shell
python -m pip install -r requirements.txt
python src/download_assets.py
python src/prepare_data.py --raw-dir raw
python src/experiment.py --checkpoint checkpoints/timesfm3 --fresh
python src/audit_results.py
python src/build_report.py
python src/build_notebook.py
python -m unittest discover -s tests -v
```

The downloader retrieves approximately 125 MB of raw source data and a 1.3 GB official checkpoint, using fixed revisions and checking their hashes. Downloaded TimesFM 3.0 weights are licensed for non-commercial, non-production use. Review the original dataset's terms as well. Raw files, generated full data panels, and weights are excluded from GitHub. Saved forecasts and derived results document the experiment. See `THIRD_PARTY_NOTICES.md`.

The original run used CPU inference. `--device cuda` is optional on a supported GPU. Minor numeric differences across platforms are possible. `--fresh` recomputes the forecasts. Without it, the experiment reuses the saved model caches, so use `--fresh` after changing inputs or model settings.

### Check the saved results without downloading the model

```shell
python -m pip install -r requirements-analysis.txt
python src/download_assets.py --skip-model
python src/prepare_data.py --raw-dir raw
python src/audit_results.py
python -m unittest discover -s tests -v
```

Optional report-control check with Node.js:

```shell
node tests/verify_report.cjs portfolio_report.html
```

Six integrity tests passed on the recorded project. The report script was checked across 288 product/method/start-date selections and cost-slider scenarios using a local test harness. That is a logic check, rather than full browser layout verification. Static result figures were visually inspected. The local browser preview was unavailable in the execution environment.

This is an AI-assisted research implementation. Saved predictions came from executing the official TimesFM checkpoint. Scores and the inventory replay were recomputed from those outputs. The repository supports review and reproduction, while the single-store design and data-availability assumptions limit the conclusions.
