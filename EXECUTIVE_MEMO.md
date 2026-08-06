# Executive Memo — Credit Risk Decisioning

**To:** Lending / Risk leadership
**Re:** What the PD model and decision policy tell us, and what to do next
**One-line:** A calibrated default-risk model plus a profit-based approval policy
identifies a cutoff that turns a loss-making book into a profitable one.

---

## The decision we are informing

For each loan application we must decide: approve, send to manual review, or
decline. Approving more grows volume but adds defaults; approving less avoids
losses but forgoes interest income. The question is where to set the cutoff.

## What we found

**1. Risk is predictable and the score is honest.** The model ranks defaults
well on loans it has never seen (a later time period than it was trained on),
and its probabilities are calibrated — when it says 15%, about 15% actually
default. That matters because the number feeds loss and capital math directly.

**2. The profit-maximizing cutoff is not the safest one.** Simulating the
approval policy across cutoffs shows the most profitable decision approves the
lowest-risk ~35% of applicants and accepts an ~11% default rate — *higher* than
the ultra-safe option, because interest from good loans more than covers the
losses. Tightening further sacrifices profit; loosening past this point turns
the book loss-making.

**3. A simple model is enough.** A transparent scorecard-style model performs
within ~0.01 AUC of a complex one. We can deploy the interpretable model and use
the complex one only to generate the per-decision reason codes regulation
requires.

## What this is worth

On the tested book, moving from a naive "approve most applicants" stance to the
model-driven cutoff is the difference between a **six-figure loss and a
six-figure profit** on the same set of loans — driven entirely by *which* loans
are approved, not by approving more of them. This is a **retrospective
simulation** on historical outcomes; a shadow/pilot run is needed to confirm it
forward before the number can be claimed as realized profit.

## Risks and honest caveats

- The analysis is on a representative sample; production numbers will differ.
- Loss and interest assumptions are simplified; a full P&L would refine the
  exact optimal cutoff, not the direction of the conclusion.
- The model only sees approved-and-funded loans, so it cannot fully speak to
  applicants historically rejected — a known limitation of any lending model.
- One region shows early signs of population drift; it is on a watchlist.

## Recommendation

1. Evaluate the profit-based cutoff as a **candidate policy in a shadow/pilot
   run** (score live applications without acting on them, compare to current
   policy) before adopting it, with a manual-review band around the boundary.
2. Deploy the XGBoost champion for decisions and explain that same model with
   SHAP; keep the logistic model as an interpretable challenger/fallback for
   reason codes and monitoring.
3. Review the drift watchlist next vintage before any re-tuning.
