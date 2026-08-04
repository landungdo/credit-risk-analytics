"""
Generate a lightweight sample from the full Lending Club dataset.

Filters to the columns needed for the credit risk model (target, date,
and predictive features) and takes a random sample, so the resulting
file is small enough to share and prototype with.
"""

import pandas as pd

# Columns required for target definition, out-of-time splitting,
# and baseline feature engineering (Weeks 1-3 of the project)
COLUMNS_NEEDED = [
    'issue_d', 'loan_status', 'loan_amnt', 'term', 'int_rate', 'installment',
    'grade', 'sub_grade', 'emp_length', 'home_ownership', 'annual_inc',
    'verification_status', 'purpose', 'dti', 'delinq_2yrs', 'earliest_cr_line',
    'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'addr_state'
]

SOURCE_PATH = "data/accepted_2007_to_2018Q4.csv"
OUTPUT_PATH = "data/sample.csv"
SAMPLE_SIZE = 20_000
RANDOM_STATE = 42  # fixed seed for reproducibility

df = pd.read_csv(SOURCE_PATH, usecols=COLUMNS_NEEDED, low_memory=False)
sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
sample.to_csv(OUTPUT_PATH, index=False)

print(f"Done. Sample has {len(sample)} rows, {len(sample.columns)} columns.")