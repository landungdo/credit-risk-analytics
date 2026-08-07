# Ablation Study — Leakage Check and Baselines

Two questions a reviewer will reasonably ask about a Lending Club PD model:

1. **Is the performance just leakage?** `int_rate`, `grade`, and `sub_grade` are
   assigned by Lending Club's *own* risk process, so a model using them can
   partly re-learn a score someone else already computed.
2. **Is the gradient boosting justified**, or would a simple linear model do?

`experiments/ablation.py` answers both by training three models on the same
out-of-time split.

## Results (out-of-time test, 2016)

| Model | AUC | KS |
|---|---|---|
| FULL — XGBoost, all features | 0.687 | 0.286 |
| NO_PRICING — XGBoost without int_rate/grade/sub_grade/installment | 0.640 | 0.218 |
| BASELINE — logistic regression, all features | 0.676 | 0.260 |

## Interpretation

**On leakage.** Removing all four pricing variables lowers AUC by only ~0.05
(0.687 -> 0.640). The model keeps most of its discrimination using borrower
attributes alone (income, DTI, credit history, etc.). So the pricing variables
add a modest lift but are not the whole story — this is not a case where the
model is merely echoing a pre-computed grade. The honest framing: pricing
variables are informative *and* partly endogenous, and the no-pricing model is
the more conservative estimate of what the borrower's own profile predicts.

**On model choice.** XGBoost beats the logistic baseline by only ~0.01 AUC on
this dataset. That is a genuinely useful finding, not a disappointing one: it
says the signal here is largely linear/monotonic, so a scorecard-style logistic
model would be a defensible production choice with far better interpretability.
The gradient boosting is retained for the SHAP-based explanation tooling, but
the project does not overclaim that complexity was necessary for accuracy.

## Why this matters

Reporting these two comparisons pre-empts the two most likely challenges to the
headline AUC, and shows the number is understood in context rather than quoted
in isolation.
