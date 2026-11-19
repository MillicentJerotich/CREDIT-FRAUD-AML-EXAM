# Model card: card-fraud decision function

**Model:** Random Forest with Platt calibration | **Decision policy:** Amount-aware Bayes rule (false-alarm cost EUR 5)
**Trained on:** oldest 60% of the de-duplicated transactions (170,235 rows, 342 frauds)
**Created:** 2026-11-19T22:39:19 | scikit-learn 1.9.1

## Intended use
Real-time scoring of card transactions to route suspicious ones to manual review. Not for automatic account closure.

## Test performance (newest 20% of transactions, 74 frauds; 95% bootstrap CI)
| Metric | Estimate | 95% CI |
|---|---|---|
| PR-AUC | 0.811 | [0.727, 0.888] |
| ROC-AUC | 0.958 | [0.920, 0.987] |
| saving % | 68.528 | [41.684, 93.777] |
| recall | 0.676 | [0.554, 0.784] |
| precision | 0.926 | [0.854, 0.981] |

## Limitations
* Two days of data from one issuer (September 2013); fraud patterns change, so retrain and re-validate before real use.
* Features V1-V28 are anonymised PCA components, so feature-level explanations have no business meaning.
* Costs are assumptions; re-derive the threshold if the false-alarm cost changes.
* Labels (chargebacks) arrive with a delay, so live precision and recall are only known weeks later.

## Monitoring
PSI on all inputs against a reference window from the same hours of day, plus mean score and alert rate.
Alert at PSI > 0.25, investigate at > 0.10 (`python -m fraud.monitoring <batch>`).
