# Explainable Credit Risk & Portfolio Analytics Platform

A probability-of-default (PD) credit risk model built with methodology that
mirrors real-world credit risk practice: out-of-time validation, probability
calibration, SHAP-based explanations grounded in natural language, fairness
auditing, and portfolio-level risk aggregation.

> **Status:** 🚧 In progress — Weeks 1–2 complete (EDA, target definition,
> out-of-time split, feature engineering).

## Key finding (so far)

Loan outcomes in the Lending Club data are **right-censored by vintage**: the
share of loans that have reached a final outcome (Fully Paid / Charged Off)
drops sharply for recent issue years — roughly 95% for 2014, but only ~40% for
2017 and ~11% for 2018, because newer loans are mostly still "Current".

Naively training or testing on those recent vintages would bias the model
toward loans that happen to resolve quickly (typically shorter-term loans). The
out-of-time split is therefore restricted to vintages with high resolution
rates (train < 2015, validation = 2015, test = 2016), and 2017–2018 loans are
excluded from modeling entirely. See [`TARGET_DEFINITION.md`](TARGET_DEFINITION.md)
for the full analysis.

## Methodology highlights

- **Out-of-time validation** (not random split) — trains on older loans and
  tests on newer ones, the way a deployed credit model actually faces the future.
- **Probability calibration** — a PD must be a real probability (used directly
  for expected-loss and capital calculations), not just a well-ranked score.
- **Grounded explanations** — SHAP identifies the drivers of each decision; an
  LLM layer translates them into natural-language adverse-action reasons without
  inventing factors beyond what SHAP surfaced.
- **Fairness audit** — disparate-impact checks across proxy groups.
- **Portfolio layer** — aggregates PD into expected loss (PD × LGD × EAD) and a
  Basel-style capital estimate.
- **PSI monitoring** — detects population drift over time, including drift that
  is invisible in aggregate metrics.

## Project structure

```
credit-risk-analytics/
├── src/                 # Core modules (imported by tests, API)
│   ├── oot_split.py     # Target definition + out-of-time train/val/test split
│   └── features.py      # Feature engineering
├── notebooks/           # Exploratory analysis
│   └── 01_eda.py        # Week 1 EDA (VS Code cell format)
├── scripts/             # One-off utilities
│   └── make_sample.py   # Generates a lightweight sample from the full dataset
├── data/                # Local only — not tracked (see .gitignore)
├── TARGET_DEFINITION.md # Target logic + right-censoring analysis
└── requirements.txt
```

## Data

Lending Club accepted-loans dataset (2007–2018), publicly available on
[Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club). The raw
file is not included in this repo (licensing + size). To reproduce:

1. Download `accepted_2007_to_2018Q4.csv` from Kaggle into `data/`.
2. Run `python scripts/make_sample.py` to create a lightweight working sample.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproduce the analysis

```bash
python notebooks/01_eda.py       # EDA + target definition + censoring finding
python src/oot_split.py          # Out-of-time split summary
python src/features.py           # Feature matrix summary
```
