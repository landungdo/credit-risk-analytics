# Target Definition

## Label mapping

| `loan_status` value | Label | Rationale |
|---|---|---|
| Fully Paid | 0 | Loan resolved, no default |
| Does not meet the credit policy. Status:Fully Paid | 0 | Same outcome, different policy flag at origination |
| Charged Off | 1 | Loan resolved as default |
| Does not meet the credit policy. Status:Charged Off | 1 | Same outcome, different policy flag at origination |
| Current | excluded | Outcome not yet known |
| Late (16-30 days) | excluded | Outcome not yet known |
| Late (31-120 days) | excluded | Outcome not yet known |
| In Grace Period | excluded | Outcome not yet known |

Excluding in-progress loans is standard practice: labeling them would require guessing a future outcome, which leaks noise into the target.

## Key finding: right-censoring by vintage

Resolution rate (resolved loans / total loans issued) drops sharply for recent issue years, because those loans haven't had enough time to reach a final outcome:

| Issue year | Resolved % (sample) |
|---|---|
| 2014 | ~95% |
| 2015 | ~90% |
| 2016 | ~67% |
| 2017 | ~40% |
| 2018 | ~11% |

**Implication:** 2017 and 2018 vintages are too right-censored to use reliably — the loans that *have* resolved by now in those years are disproportionately short-term/fast-resolving loans, which biases any split that includes them as-is.

## Revised out-of-time split

Original plan (train < 2016 / validation 2016 / test 2017+) is **revised**:

- **Train:** issue_d < 2015-01-01
- **Validation:** issue_d in 2015
- **Test (out-of-time):** issue_d in 2016
- **Excluded entirely from modeling:** 2017–2018 vintages (too censored for reliable labels)

This still gives genuine chronological separation (train on older loans, test on newer ones) while avoiding the censoring bias. Document this decision in the final README — a reviewer familiar with credit risk modeling will recognize vintage censoring as exactly the kind of pitfall this project is designed to demonstrate awareness of.

## Next steps (Week 2)

Apply this split logic in `src/oot_split.py`, and confirm the resolution-rate pattern holds on the full dataset (not just this 20k sample) before finalizing cutoffs.
