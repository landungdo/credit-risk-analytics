# %% [markdown]
# # Week 1 — EDA & Target Definition
# Explainable Credit Risk & Portfolio Analytics Platform
#
# Works either way: run via `python notebooks/01_eda.py` in the terminal,
# or cell-by-cell in VS Code with Ctrl+Enter. Every result is explicitly
# printed so nothing depends on Jupyter's auto-display behavior.

# %%
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

df = pd.read_csv("data/sample.csv", low_memory=False)
print("Shape:", df.shape)
print(df.head())

# %%
# Missing values check
print("\n--- Missing values ---")
missing = df.isnull().sum()
print(missing[missing > 0])

# %%
# loan_status is the raw field the target will be derived from
print("\n--- loan_status value counts ---")
print(df["loan_status"].value_counts())

# %%
# Parse issue_d ("Feb-2015" style) into an actual datetime for time-based splitting
df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
print("\n--- issue_d range ---")
print("issue_d range:", df["issue_d"].min(), "to", df["issue_d"].max())
print("\n--- Loans per year ---")
print(df["issue_d"].dt.year.value_counts().sort_index())

# %% [markdown]
# ## Target definition
#
# - `1` (bad / default): "Charged Off" and its credit-policy-exception counterpart
# - `0` (good): "Fully Paid" and its credit-policy-exception counterpart
# - Excluded: "Current", "Late (16-30/31-120 days)", "In Grace Period" —
#   these loans have not reached a final outcome yet.

# %%
BAD_STATUS = ["Charged Off", "Does not meet the credit policy. Status:Charged Off"]
GOOD_STATUS = ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]

resolved = df[df["loan_status"].isin(BAD_STATUS + GOOD_STATUS)].copy()
resolved["target"] = resolved["loan_status"].isin(BAD_STATUS).astype(int)

print(f"\n--- Target definition ---")
print(f"Resolved loans: {len(resolved)} / {len(df)} ({len(resolved)/len(df):.1%})")
print("\nTarget distribution:")
print(resolved["target"].value_counts())
print(f"Default rate: {resolved['target'].mean():.2%}")

# %% [markdown]
# ## ⚠️ Key finding: right-censoring in recent vintages
#
# Loans issued recently haven't had time to reach a final outcome yet —
# most are still "Current". Check the *resolution rate* by issue year,
# not just the raw target distribution above.

# %%
resolution_by_year = df.groupby(df["issue_d"].dt.year).apply(
    lambda g: pd.Series({
        "total_loans": len(g),
        "resolved_loans": g["loan_status"].isin(BAD_STATUS + GOOD_STATUS).sum(),
    })
)
resolution_by_year["resolved_pct"] = (
    resolution_by_year["resolved_loans"] / resolution_by_year["total_loans"]
)

print("\n--- Resolution rate by issue year (THE key table) ---")
print(resolution_by_year)

print("\nThis table drives the out-of-time split cutoffs in TARGET_DEFINITION.md —")
print("years with low resolved_pct are too right-censored to use reliably.")
