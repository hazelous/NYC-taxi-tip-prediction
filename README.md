# NYC Taxi Tip Prediction

Predicts the tip amount on NYC Yellow Taxi trips from trip and fare data, covering the full path from raw monthly files to a tuned model. Built on roughly 11 million trips from Q1 2026 which is reduced to **6.7 million** after cleaning.

The more interesting outcome is not the model score but what the feature importances turned out to mean, which is covered below.

## Data

Three monthly Parquet files of Yellow Taxi trip records from the NYC Taxi and Limousine Commission, January to March 2026, plus the taxi zone lookup table.

Tip amount was chosen as the target because it depends on more than the mechanics of the trip. Rider behaviour, time of day, location and the structure of the fare all plausibly play a part which makes it a more interesting question than predicting fare or duration.

Pandas was chosen over PySpark deliberately. Three months is large enough to capture real variation across time of day, day of week, location and fare structure while still fitting in memory on one machine and allowing fast iteration during model tuning. A larger span would have added scale without adding much to this question.

## Cleaning

The monthly files were checked for matching columns and data types before concatenation since a silent mismatch there produces a malformed dataset that is difficult to notice later. There were no duplicate rows.

The raw data had several problems:

- Timestamps reaching back to 2008 which is far outside the analysis window and is likely from meters with incorrect date settings
- Negative values in `fare_amount`, `tip_amount`, `tolls_amount` and `total_amount` which are most likely refunds or corrections
- Trip distances above **328,000 miles** which is further than the Earth is from the Moon so, too far to be realistic.
- Missing `passenger_count` values

Filtering removed records outside Q1 2026, trips with a dropoff earlier than the pickup and impossible distances, fares and tips. Missing passenger counts were filled with the median. Row counts were tracked after each filter so the effect of each one stayed visible.

The largest single filter was restricting to only credit card payments since the TLC data only records tips for card transactions. That removes about a third of the data and introduces selection bias which is discussed under Limitations.

## Feature engineering

Each feature was added for a specific reason rather than because it could be computed.

| Feature | Reasoning |
|---|---|
| `pickup_hour`, `pickup_dayofweek`, `pickup_month` | Tipping is behavioural and behaviour varies by time. Late night riders differ from commuters, and weekends serve different trips. |
| `trip_duration_minutes` | How long the rider actually spent in the cab, which distance alone misses. |
| `avg_speed_mph` | Whether the trip ran smoothly or sat in traffic, which may affect satisfaction. |
| `fare_per_mile` | What value the rider got for the fare. |
| `pickup_borough`, `dropoff_borough` | From joining trips to the zone lookup. Manhattan serves business and tourist traffic where other boroughs serve more local trips. |
| `is_airport_trip` | Airport trips are longer, often flat-rate, and carry different riders. |

The derived features exposed a second round of outliers. Maximum trip duration came out near 30 hours, maximum average speed above 37,000 mph, and maximum fare per mile above $45,000. These come from dividing by very small numbers, since a trip recorded as lasting one second produces an enormous speed. The original cleaning had capped the upper bounds of distance and duration but set no meaningful lower bounds. A second filter brought maximum speed down to 80 mph and duration to 2 hours while keeping most rows.

## Feature selection

Some columns were deliberately excluded:

- **`total_amount`**, because it includes the tip and would leak the target directly
- Raw timestamps because it was already represented by `pickup_hour` and `trip_duration_minutes`
- Location IDs because it was already one-hot encoded as boroughs
- `payment_type` because we only have 1 after the credit card filter and therefore uninformative
- Surcharges and congestion fees which are largely fixed amounts already implied by `is_airport_trip`
- `VendorID`, `RatecodeID`, `store_and_fwd_flag`which does not reflect rider behaviour

Using every available column would have added correlated features and dimensions without adding information.

## Models

An 80/20 split with a fixed random state, linear regression first as a baseline and then a tuned random forest.

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 2.56 | 1.43 | 0.59 |
| Random Forest (tuned) | 2.38 | 1.26 | 0.64 |

Hyperparameters were tuned by grid search over `max_depth` and `min_samples_leaf` on a 200k-row subset since searching the full set would have been prohibitively slow with the final model then trained on all the data. The best values were `max_depth=15` and `min_samples_leaf=20` with 100 trees.

The random forest cut typical error by about 12% and explained 5% more variance. That improvement is real but smaller than expected, which itself says something: most of what makes tips predictable is close to linear and roughly 36% of the variance stays unexplained.

## What the feature importances actually mean

`is_airport_trip` came out as by far the most influential feature, followed by `fare_amount`. That is surprising given airport trips are only about 9% of the data so it was worth investigating rather than accepting.

Comparing the two groups directly:

- Airport trips have tips roughly 4x larger in absolute terms with a median of $12.21 against $3.01
- But as a share of the fare airport riders tip slightly less with a median of 23.5% against 26.9%

So `is_airport_trip` is not detecting generous riders, it is detecting long and expensive trips. Tipping scales with fare, airport trips have high fares and the flag is a clean binary split that lets the trees capture that in one cut. It correlates strongly with `fare_amount`, and the model would likely perform similarly using either feature alone.

## Tipping across the day

Mean tip amount and mean tip-to-fare ratio have noticeably different shapes.

The amount sits high around midnight at roughly $4.36, dips to about $3.47 at 7am, then climbs through the day to peak near $4.50 in the late evening.

The **ratio** is roughly flat overnight at about 25%, dips, then climbs to peak at 27% around 5pm. Its lowest point is around 5am, which is when airport travel is heaviest, consistent with the finding above.

Much of the variation in tip amount across the day is simply fare driven, since longer overnight trips raise the amount without changing the percentage. But the ratio is not flat. It moves about 7 percentage points between its low and its peak and that part is behavioural rather than explained by fare.

## Limitations

**Selection bias**
Only credit card trips record tips so about a third of trips are excluded and the results generalise to card-paying Yellow Taxi riders rather than to all NYC taxi riders. Cash payers may tip differently.

**Unobservable behaviour** 
Tipping depends on mood, generosity, satisfaction, habit and which button appears on the payment screen. None of that is in trip metadata which puts a ceiling on how well any model can do here. Richer behavioural data would help but would raise real privacy concerns around re-identification.

**One quarter only** 
Seasonal patterns and year-over-year trends are out of reach without a longer span.

**Fixed outlier thresholds**
Distance under 100 miles, fares under 500, tips under 200, durations from 1 to 180 minutes, speed under 80 mph. These were set to remove obvious nonsense rather than to classify borderline cases carefully. For context, the 99.9th percentile of average speed in the cleaned data is 44 mph against an 80 mph cap, so the thresholds are generous.

## A natural next step

Predicting tip percentage rather than tip amount would isolate the behavioural component given that the absolute amount is largely a function of fare. Extending to a full year would open up seasonality.

## Running it

```
pip install pandas numpy scikit-learn pyarrow matplotlib
```

No data files are committed to this repository. The Parquet files are around 60MB each and everything used here is public data available directly from the source.

From the [NYC TLC trip record data page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), download:

- `yellow_tripdata_2026-01.parquet`
- `yellow_tripdata_2026-02.parquet`
- `yellow_tripdata_2026-03.parquet`
- `taxi_zone_lookup.csv`

Place all four beside the notebook keeping those filenames then run it top to bottom.

The notebook is committed with its outputs saved so all tables and plots can be read on GitHub without downloading anything or running it.
