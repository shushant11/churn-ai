# Model Card: Churn AI v1.0.0

## Intended use
Rank existing retail-bank customers by their calibrated probability of closing their account, so that a relationship team can prioritise retention contact. It is decision support: a person approves any offer above standard authority. It is not intended for credit, pricing or eligibility decisions, and must not be used to deny service.

## Data
Public "Churn_Modelling" dataset: 10,000 customers of a European bank (France, Germany, Spain), 20.4% churned. Identifiers (RowNumber, CustomerId, Surname) are removed before training. There are no timestamps, so validation is a random stratified split, not a time split. Split: 60% train, 20% validation (model selection, calibration, threshold), 20% test (reported once), seed 42.

## Model
Scikit-learn Pipelines: engineered features (balance-to-salary ratio, zero-balance flag, 3+ products flag) and one-hot geography. The deployed model is an equal-weight average of three pipelines (random forest, XGBoost, histogram gradient boosting), chosen as the best three of 13 single models by validation PR-AUC (0.700 for the blend, against 0.696 for the best single model), then Platt-calibrated on validation. The model is **gender-blind**: gender is not an input (`--keep-gender` trains the gender-aware version for comparison). Decision threshold 0.30 maximises expected campaign profit on validation under the business assumptions in `config.py`.

## Performance (test set, n = 2,000)
PR-AUC 0.712 (95% CI 0.672-0.752), ROC-AUC 0.862 (0.840-0.881), Brier 0.101, ECE 0.020. At threshold 0.30: precision 0.60, recall 0.64, 21.9% of customers flagged. The riskiest decile has a 4.2x lift and the riskiest 20% contain 62% of churners. Expected profit under the stated assumptions: +17,300 (the same as at threshold 0.5), against -38,950 for contacting everyone.

## Fairness evaluation (test set, threshold 0.30)
| Group | Actual churn | Selection rate | Recall | False positive rate |
|---|---|---|---|---|
| Female | 25.1% | 24.5% | 0.64 | 0.11 |
| Male | 16.3% | 19.5% | 0.65 | 0.11 |
| France | 17.1% | 17.4% | 0.56 | 0.09 |
| Germany | 30.0% | 35.7% | 0.76 | 0.18 |
| Spain | 17.1% | 16.6% | 0.59 | 0.08 |

Disparity ratios (lowest group over highest): gender selection 0.80 (0.797), recall 0.98, false positive 0.93; geography selection 0.47, recall 0.74, false positive 0.43. The selection-rate ratio is below 0.8 for both attributes, which triggers review under the fair-treatment policy. Review outcome for gender: the model does not use gender, recall and false-positive rates are nearly equal, and the remaining selection gap follows the real difference in churn (25% vs 16%). The same blend trained with gender reaches PR-AUC 0.720 (a difference inside the confidence interval) with a gender selection ratio of 0.59 and a false-positive ratio of 0.53, which is why the deployed model leaves gender out. Removing a column does not remove proxies, so the audit still runs on every retrain. For geography, Germany's real churn is much higher and Germany's churners are found more often (recall 0.76 vs 0.56 for France and 0.59 for Spain); this is a known weakness to monitor. Mitigations: actions are chosen from behavioural drivers only, the assistant's output guardrail rejects answers that give gender or nationality as a reason, and the audit runs on every retrain.

## Explainability
Global: permutation importance (number of products, age, activity, balance, geography lead). Local: per-customer "typical value" substitution showing how each field moves the probability. Optional SHAP. These explain the model, not causation.

## Monitoring and retraining
PSI and KS per feature against the training reference. Retrain when any important feature's PSI is above 0.25, monthly PR-AUC on new labels falls by more than 0.05, or every six months.

## Limitations
Small, old, anonymised public sample; no time dimension; business values are assumptions; survival analysis treats tenure as time-to-event, which is approximate for a snapshot; what-if results are model estimates, not treatment effects.
