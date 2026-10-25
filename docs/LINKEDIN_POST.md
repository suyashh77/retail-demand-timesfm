I wanted to see how Google's TimesFM 3.0 would handle a retail forecasting problem.

TimesFM is already trained on other time-series data. It can forecast related sales histories and use inputs such as a known calendar without training a new model for the dataset.

I tested four product groups at one Favorita grocery store, predicting fourteen days at twelve weekly starting dates. The comparisons were a forecast that repeats last week and gradient boosting fitted to earlier sales.

TimesFM with calendar information had 9.68% total forecast error, versus 12.56% for the weekly rule and 9.69% for gradient boosting. The fitted model was almost tied. TimesFM did better on beverages and bread/bakery, while gradient boosting did better on dairy and grocery I. Adding promotions and past transactions did not improve the calendar version.

An inventory simulation also explored the ordering decisions. Its best TimesFM version had 20.4% lower simulated cost than the weekly rule under the chosen assumptions. These are simulated outcomes, and future promotions were assumed planned because dated plans were unavailable.

Code, methods, and saved results: https://github.com/suyashh77/retail-demand-timesfm

Which additional baseline would you include before extending this comparison across stores?
