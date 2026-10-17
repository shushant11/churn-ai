# Churn Model FAQ

## What does the churn probability mean
The probability is calibrated. A score of 0.30 means that among many customers with that score, about 30 out of 100 left the bank. It is not a certainty for any single customer.

## Why is the threshold not 0.5
The threshold is chosen on validation data to maximise the expected profit of the retention campaign. Missing a churner costs more than an unnecessary offer, so the best threshold is usually lower than 0.5.

## How are drivers calculated
For each field the assistant asks what the probability would be if this customer had a typical value instead. The difference is the driver's contribution. A positive contribution pushes the customer towards churn. Drivers explain the model, not the real cause, and they are not proof of causation.

## How accurate is the model
Accuracy is the wrong measure because only about 20 percent of customers churn. The model is judged by PR-AUC, recall at the campaign threshold, calibration and expected profit on a held-out test set that was used only once.

## When is the model retrained
The model is retrained when population stability index on any important feature exceeds 0.25, when the monthly PR-AUC on newly labelled data falls by more than 0.05, or every six months, whichever comes first.
