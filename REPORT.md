# Credit Risk & Portfolio Analytics — Technical Report

**Project:** Explainable probability-of-default (PD) modeling on Lending Club data
**Scope:** end-to-end pipeline — data sourcing, methodology, modeling, explainability, fairness, portfolio risk, drift monitoring, and a served API with CI.

---

## 1. Problem statement

The goal is to estimate the probability that a consumer loan defaults, in a way
that is defensible for real credit-risk use rather than a leaderboard score.
That means three things beyond raw accuracy:

1. The evaluation must reflect how a deployed model actually faces the future.
2. The predicted probability must be usable directly in loss and capital math.
3. Every individual decision must be explainable and auditable for fairness.

These requirements shape every decision below.

---

## 2. Data

**Source.** Lending Club accepted-loans dataset (2007–2018), publicly available
on Kaggle. The raw file contains ~2.2M loans and ~150 columns.

**Column selection.** 22 columns are retained: the outcome (`loan_status`), the
origination date (`issue_d`), and borrower/loan attributes available at
application time (loan amount, term, interest rate, grade, income, DTI,
delinquencies, credit-line counts, revolving utilization, employment length,
home ownership, purpose, state, etc.).

**Leakage avoidance at selection time.** Post-origination fields that only exist
*after* a loan runs (e.g. `total_pymnt`, `recoveries`) are deliberately excluded
— they would leak the outcome into the features.

**Working sample.** A 20,000-row random sample is used for development so the
pipeline is fast and shareable. The sample preserves the full time span, so all
time-based analysis remains valid.

---

## 3. Target definition

The label is derived from `loan_status`:

- **1 (default):** Charged Off (and its credit-policy-exception variant)
- **0 (repaid):** Fully Paid (and its variant)
- **Excluded:** Current, Late, In Grace Period — these loans have no final
  outcome yet, so labeling them would inject guesses into the target.

### Key finding #1 — right-censoring by vintage

The share of loans that have reached a final outcome drops sharply for recent
issue years:

| Issue year | Resolved % |
|---|---|
| 2014 | ~95% |
| 2015 | ~90% |
| 2016 | ~67% |
| 2017 | ~40% |
| 2018 | ~11% |

Recent vintages are dominated by still-open loans, and the ones that *have*
resolved are disproportionately fast-resolving (shorter-term) loans. Using them
as-is would bias the model. **Consequence:** 2017–2018 vintages are excluded
from modeling entirely.

---

## 4. Methodology

### 4.1 Out-of-time split (not random)

A random split lets the model see future and past loans together and hides the
degradation a real model suffers over time. Instead the split is chronological:

- **Train:** issue_d < 2015
- **Validation:** 2015 (used for early stopping and calibration)
- **Test:** 2016 (a genuinely held-out later vintage)

Default rates rise across the splits (≈16.9% → 20.9% → 23.7%), a real vintage
effect the out-of-time design surfaces rather than masks.

### 4.2 Features

Categorical variables are passed to the model using pandas `category` dtype
rather than one-hot encoding (avoids a ~50-column explosion from `addr_state`).
Missing values are left as NaN on purpose — tree models handle them natively and
often learn a useful split from "missing". One engineered feature,
`credit_history_months` (issue date minus earliest credit line), is added as a
standard high-signal credit variable.

### 4.3 Model

Gradient-boosted trees (XGBoost) with early stopping on the 2015 validation
vintage, modest depth and regularization to limit overfitting, and a fixed
random seed for reproducibility.

### 4.4 Calibration

A ranking model (high AUC) is not necessarily a *calibrated* one. Because the PD
feeds loss and capital math, an isotonic calibrator is fit on the validation
vintage and evaluated on the test vintage, measured by Brier score and a
reliability diagram.

---

## 5. Results

### 5.1 Discrimination (out-of-time test, 2016)

| Metric | Train | Validation (2015) | Test (2016, OOT) |
|---|---|---|---|
| AUC | 0.79 | 0.72 | 0.68 |
| KS  | 0.44 | 0.32 | 0.28 |

The train-to-test gap is visible and *measured* — the honest picture an
out-of-time split produces, versus the inflated one a random split would give.

### 5.2 Calibration

Isotonic calibration lowers the Brier score on the test vintage (≈0.173 → 0.170)
and pulls predicted probabilities toward the diagonal on the reliability diagram,
making the PD usable as a real probability:

![Reliability diagram](reports/reliability_diagram.png)

### 5.3 Ablation — leakage check and baselines

`int_rate`, `grade`, and `sub_grade` are assigned by Lending Club's own risk
process, so a natural question is whether the model just re-learns their grade.
Three models on the same split:

| Model | AUC | KS |
|---|---|---|
| FULL — XGBoost, all features | 0.687 | 0.286 |
| NO_PRICING — without int_rate/grade/sub_grade/installment | 0.640 | 0.218 |
| BASELINE — logistic regression, all features | 0.676 | 0.260 |

### Key finding #2 — the score is not mostly leakage, and the signal is largely linear

- Dropping all pricing variables costs only ~0.05 AUC (0.687 → 0.640): the model
  keeps most of its discrimination from borrower attributes alone, so it is not
  merely echoing a pre-computed grade.
- XGBoost beats a logistic baseline by only ~0.01 AUC: the relationship is
  largely linear/monotonic, so a scorecard-style logistic model would be a
  defensible, more interpretable production choice. The gradient boosting is
  retained mainly to support SHAP-based explanations.

---

## 6. Explainability

Each prediction is attributed with SHAP, surfacing the top drivers pushing it
toward or away from default. A natural-language adverse-action reason is then
generated from those drivers.

### Key finding #3 — grounding guard against invented reasons

An LLM asked to "explain a decline" can invent plausible but unsupported
reasons, which in lending is a compliance risk. A validation step checks the
generated text against the permitted SHAP drivers and rejects any explanation
that introduces a factor SHAP did not surface. This behavior is covered by unit
tests: a clean explanation passes, and one that invents "employment history" or
"home ownership" is caught.

---

## 7. Fairness

The data has no direct protected attributes, so a disparate-impact audit uses
income-bracket and region proxies. At an illustrative approval threshold, the
four-fifths rule flags the low-income group. Crucially, the approved-default
rate is also reported per group: the flagged group's higher decline rate tracks
a genuinely higher realized default rate, illustrating the core tension in fair
lending between statistical accuracy and disparate impact — a disparity is not
automatically evidence of an unjustified bias, but it still warrants review.

---

## 8. Portfolio risk

Calibrated PDs are aggregated into portfolio metrics a risk function reports:

- **Expected Loss** = PD × LGD × EAD (LGD 45%, EAD ≈ loan amount) — ≈10% of
  exposure on the test portfolio.
- **Capital** via a simplified Basel IRB formula — ≈18% of exposure.

This closes the loop from an ML score to the financial numbers the score is
ultimately for, and is the reason calibration (Section 5.2) matters.

---

## 8b. From prediction to decision

A PD score is an input, not a decision. Three components turn the model into a
credit-risk decisioning system:

**Decision policy simulator.** Sweeping the approval cutoff shows the
approval-rate / default-rate / expected-profit trade-off. The cutoff is selected
on the 2015 validation book, then frozen and evaluated once on the untouched 2016
test book, so the reported test profit carries no test-set optimism bias: the
cutoff never sees the test data. (It is selected on the same 2015 vintage used
for calibration and early stopping; splitting 2015 into separate calibration and
policy-selection halves would tighten this further and is noted as a refinement.) The profit-maximizing cutoff approves lower-risk
applicants and accepts a moderate default rate — deliberately not the
lowest-default option, because interest on
good loans outweighs the losses. Below that cutoff the book is profitable; loosen
too far and it turns loss-making. This reframes the model output as a business
decision with a quantified optimum.

**Champion / challenger.** Model selection is treated as governance, not a
leaderboard: the XGBoost champion (deployed and SHAP-explained) is compared against a logistic challenger
across discrimination (AUC/KS), calibration (Brier), latency, and
interpretability. The champion's edge over the challenger is small (~0.01 AUC),
which is stated openly: the extra complexity is justified not by accuracy but by
the need for per-decision SHAP attribution to power adverse-action reason codes.
XGBoost therefore both makes the decision and is explained, so the reason codes
describe the deployed model; the logistic challenger is retained as an
interpretable benchmark and fallback.

**SQL risk mart.** Eight standard portfolio queries (vintage curves, bad-rate by
grade, resolution rate, exposure concentration) run over the loan book in SQL —
the reporting layer a risk team lives in alongside the model.

A one-page executive memo translates all of this into business language for a
non-technical stakeholder.

## 9. Drift monitoring

Population Stability Index (PSI) is computed overall and segmented by subgroup.

### Key finding #4 — subgroup drift hides inside a stable aggregate

The overall PSI is very low (~0.02, "stable"), yet the Northeast region reaches
the "moderate" threshold (~0.10). An aggregate-only monitor would have missed
it. The proportionate response is to watchlist the segment, not to recalibrate
on a single borderline, small-sample reading — which is exactly the judgment a
drift monitor exists to support.

---

## 10. Engineering

- **Serving:** a FastAPI service exposes `/predict`, `/explain`, `/decision`,
  and `/portfolio/summary`, backed by persisted model artifacts and a versioned
  `policy.json` loaded at startup. `/decision` returns the PD, an
  approve/manual-review/decline decision from the frozen policy, model/policy
  versions, and SHAP reason codes from the same champion; invalid inputs are
  rejected with HTTP 422.
- **Containerization:** a Dockerfile trains and serves the model, generating a
  synthetic demo dataset at build time so it builds from a clean clone.
- **Testing & CI:** a pytest suite covers metrics, split integrity,
  portfolio math, PSI properties, and the explanation grounding guard; GitHub
  Actions runs it on every push.

---

## 11. Limitations

Stated plainly so the results are read in context:

- Pricing features (`int_rate`/`grade`/`sub_grade`) are partly endogenous; the
  NO_PRICING model is the conservative estimate of borrower-intrinsic risk.
- Class imbalance (~20%) is handled implicitly via threshold-independent metrics
  rather than resampling.
- The fairness approval threshold and the LGD/EAD assumptions are illustrative,
  not tuned to a business cost matrix or modeled.
- Fairness groups are proxies, not a compliance-grade fair-lending analysis.
- Metrics come from a representative sample; exact values vary with sample/seed.

---

## 12. Conclusion

The project delivers a PD model whose value is in its methodology, not a headline
number: out-of-time validation that measures rather than hides degradation, a
calibrated probability that feeds real loss and capital math, explanations that
are guarded against fabrication, a fairness audit that distinguishes disparity
from bias, and drift monitoring that catches what an aggregate metric misses.
The ablation study shows the result is neither pure leakage nor dependent on
model complexity — an honest, defensible position. The remaining limitations are
scoped deliberately and documented, which is itself part of the intended
demonstration: knowing what a model does *not* establish is as important as its
metrics.
