29-Jan-2019_batch9_prediction.csv contains the live predictions from the validation batch (which started on 24-Jan-2019).
Two channels returned anomalous predictions - channel 46 (bad contact) and channel 12 (unknown error).

predictions.csv contains the same data in 29-Jan-2019_batch9_prediction.csv, but in a more convenient format.

predictions_debiased.csv is identical to predictions.csv, but with 145 cycles subtracted from each prediction. 145 cycles is the difference between the means of the predictions and of the validation experiments.

validation.csv contains the (shuffled) policies to be tested.

validation_analysis.py plots predicted vs estimated cycle life, as well as their rankings.
