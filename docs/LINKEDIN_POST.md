Grocery stores place orders before they know how much customers will buy. I wanted to test whether Google's TimesFM 3.0 could help with that problem using its existing training, without fitting the model specifically to the store.

The question was whether it could compete with familiar forecasting approaches and provide useful estimates of uncertainty.

I used four product groups from Favorita store 1 in Quito, Ecuador. Each TimesFM forecast used 256 days of history to predict fourteen days, repeated at twelve weekly starting dates. I compared several input configurations with a forecast that repeats last week and gradient boosting fitted to earlier sales.

TimesFM with calendar information had 9.68% total forecast error, compared with 12.56% for the weekly rule and 9.69% for gradient boosting. Their overall scores were very close. Adding promotions and historical transactions did not improve the calendar configuration.

I also checked prediction intervals, the model's estimated ranges for future sales. Its middle-80% range contained 87.8% of recorded observations. The ranges were conservative on this sample, while the overall quantile-error score was very close to gradient boosting's. This does not establish an advantage in uncertainty forecasting.

The study covers one store and period. Future promotions were assumed planned because dated plans were unavailable, and overlap with the model's earlier training data cannot be ruled out.

[Model documentation and official code](https://github.com/google-research/timesfm)

[Project code, methods, and saved results](https://github.com/suyashh77/retail-demand-timesfm)

How would you evaluate whether these prediction intervals are reliable enough to inform extra stock across stores?
