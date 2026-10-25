I heard about Google's TimesFM 3.0 and wanted to examine the idea of a "ChatGPT moment" for time-series forecasting: could a model trained on many sequences forecast a new task without being trained specifically for it?

Language models learn by predicting tokens, pieces of text. TimesFM learns patterns in numerical sequences. Google reports training it on over a trillion real and synthetic time points. It groups observations into patches and predicts future values from history. Version 3.0 can forecast the horizon in one pass, using related series and known future inputs. [Google explains the approach here](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/).

I wanted to test its accuracy and uncertainty estimates on grocery sales, where forecasts inform stock orders.

**Analysis:** I used four product groups at one Favorita store in Quito, Ecuador. Each TimesFM forecast used 256 days of history to predict fourteen days, tested at twelve weekly starts.

I compared separate and joint forecasts, then added calendar inputs: weekday, time of year, holidays and payday indicators. Another configuration added promotions and past transactions. Baselines repeated last week or used gradient boosting fitted to earlier sales. TimesFM's weights stayed unchanged.

**Results:** Calendar-aware TimesFM scored 9.68% WAPE, versus 12.56% for the weekly rule and 9.69% for gradient boosting. WAPE is total absolute error divided by total recorded sales. The overall scores were very close.

TimesFM did better on beverages and bread/bakery. Gradient boosting did better on dairy and grocery I. Adding promotions and past transactions together raised TimesFM's error to 9.99%. More inputs did not help here.

TimesFM's estimated middle-80% range covered 87.8% of observations, versus 86.2% for gradient boosting. Wider ranges can cover more outcomes, so coverage alone cannot establish better uncertainty estimates.

**Takeaway and next study:** TimesFM came close to a fitted baseline without task-specific training. But gradient boosting received inputs absent from calendar-only TimesFM, and product-level results varied.

I would match input sets, add promotions and transactions separately, then repeat across stores and untouched periods before comparing shortages and stock held. This is a single-store historical study with overlapping forecast windows. Future promotions were assumed known, and pretraining overlap cannot be ruled out.

[Full study and interactive forecasts](https://suyashh77.github.io/suyashsapre_portfolio/timesfm-retail/)

[Code, methods, saved results and limitations](https://github.com/suyashh77/retail-demand-timesfm)

Would the product-level differences persist when both models receive the same inputs?
