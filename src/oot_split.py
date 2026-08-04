"""
Out-of-time train/validation/test split for the credit risk model.

Loans issued 2017 onward are excluded entirely: EDA showed they are too
right-censored (most still "Current") to provide a reliable label — see
TARGET_DEFINITION.md for the resolution-rate-by-vintage analysis behind
this decision.
"""

import pandas as pd

BAD_STATUS = ["Charged Off", "Does not meet the credit policy. Status:Charged Off"]
GOOD_STATUS = ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]

TRAIN_END = "2015-01-01"        # train: issue_d < this
VALIDATION_YEAR = 2015          # validation: issue_d in this year
TEST_YEAR = 2016                # test: issue_d in this year
CENSORED_CUTOFF = "2017-01-01"  # loans on/after this date are excluded (right-censored)


def load_resolved_loans(path: str) -> pd.DataFrame:
    """Load raw data, parse dates, and keep only loans with a known outcome."""
    df = pd.read_csv(path, low_memory=False)
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")

    resolved = df[df["loan_status"].isin(BAD_STATUS + GOOD_STATUS)].copy()
    resolved["target"] = resolved["loan_status"].isin(BAD_STATUS).astype(int)

    # Drop right-censored vintages (2017+) — see TARGET_DEFINITION.md
    resolved = resolved[resolved["issue_d"] < CENSORED_CUTOFF]

    return resolved


def out_of_time_split(df: pd.DataFrame):
    """Split by issue_d: train < 2015, validation = 2015, test = 2016."""
    train = df[df["issue_d"] < TRAIN_END]
    validation = df[df["issue_d"].dt.year == VALIDATION_YEAR]
    test = df[df["issue_d"].dt.year == TEST_YEAR]

    return train, validation, test


if __name__ == "__main__":
    resolved = load_resolved_loans("data/sample.csv")
    train, validation, test = out_of_time_split(resolved)

    print(f"Train:      {len(train):>6} loans  (issue_d < {TRAIN_END})")
    print(f"Validation: {len(validation):>6} loans  (issue_d in {VALIDATION_YEAR})")
    print(f"Test:       {len(test):>6} loans  (issue_d in {TEST_YEAR})")
    print()
    for name, split in [("Train", train), ("Validation", validation), ("Test", test)]:
        print(f"{name} default rate: {split['target'].mean():.2%}")
