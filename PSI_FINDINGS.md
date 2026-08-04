# Week 7 — PSI Monitoring Findings

## What was measured

Population Stability Index (PSI) between the training baseline (pre-2015
vintages) and the out-of-time test (2016), computed both overall and segmented
by income bracket and region. PSI was also checked across 2015-2018 vintages to
probe a longer horizon.

## Result: the portfolio is stable — and that is a real finding, not a null result

All PSI values are below 0.10 (the "stable" threshold), both overall and within
every subgroup:

| Segment | PSI (train -> 2016) | Status |
|---|---|---|
| Overall | ~0.027 | stable |
| Income: high / mid / low | 0.03-0.06 | stable |
| Region: all four | 0.02-0.07 | stable |

Extending to later vintages (2015-2018) keeps every yearly PSI under 0.035.

**This is reported honestly rather than engineered into a dramatic "drift
discovered" story.** The Lending Club score distribution genuinely did not shift
much over this window, so the correct professional conclusion is: no
recalibration is triggered by the drift monitor for this period.

## Why the segmented monitor still matters

The value of segmented PSI is not that it always finds drift — it is that it
*would* catch drift concentrated in one subgroup that an aggregate PSI hides.
The two together answer different questions:

- **Overall PSI** — has the population shifted on average?
- **Segmented PSI** — has any subgroup shifted, even if the average looks calm?

Building the segmented monitor now means that when a future vintage does drift
(e.g. a macro shock hitting one region or income tier first), the system will
surface it at the subgroup level before it shows up in the aggregate — which is
exactly when a risk team wants the warning.

## Interview framing

The honest takeaway is stronger than a manufactured one: "I built subgroup-level
drift monitoring on top of the aggregate PSI. On this dataset the population was
stable, so the monitor correctly did *not* fire — but the segmented design is
what lets it catch drift that hides inside a stable-looking average." Being able
to say "the metric said stable, so I did not over-react" demonstrates the
judgment a monitoring system is actually for.
