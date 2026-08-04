# Week 7 — PSI Monitoring Findings

## What was measured

Population Stability Index (PSI) between the training baseline (pre-2015
vintages) and the out-of-time test (2016), computed both overall and segmented
by income bracket and region.

    PSI < 0.10  -> stable
    0.10-0.25   -> moderate shift, monitor
    PSI > 0.25  -> significant shift, investigate / recalibrate

## Key finding: a subgroup drifted while the overall metric looked calm

The aggregate PSI is very low, yet one region crosses into the "moderate" band:

| Segment | PSI (train -> 2016) | Status |
|---|---|---|
| **Overall** | **~0.02** | **stable** |
| Region: Northeast | ~0.10 | **moderate** |
| Region: Midwest / South / West | 0.01-0.05 | stable |
| Income: high / mid / low | 0.02-0.06 | stable |

This is exactly the failure mode segmented monitoring is built to catch: an
overall PSI of ~0.02 would reassure a risk team that nothing has moved, while
the Northeast segment has already reached the monitoring threshold. Aggregate
drift monitoring hides subgroup drift; segmented monitoring surfaces it.

## The measured response — proportionate, not alarmist

Northeast sits just over the 0.10 line and is a smaller segment (n ~= 485), so
it is sensitive to sampling noise. The correct professional action is therefore
**watchlist, not recalibrate**: flag the segment for review next vintage rather
than retraining the model on a single borderline reading. Treating a marginal,
small-sample signal as if it were a decisive one would be its own error.

(Exact PSI values move slightly between training runs because the model has a
random component; the model uses a fixed random_state so results are
reproducible on a given machine.)

## Interview framing

"I built subgroup-level drift monitoring on top of aggregate PSI. On this data
the overall PSI said 'stable', but the segmented view flagged the Northeast at
the moderate threshold — the exact drift an aggregate metric would have hidden.
I treated it as a watchlist item rather than a recalibration trigger, because
it was a borderline reading on a small segment." This shows both why segmented
monitoring matters and the judgment to respond proportionately to what it finds.
