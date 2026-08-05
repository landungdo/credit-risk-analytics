"""
SQL risk mart.

Credit-risk analytics is done in SQL as much as in Python: a risk team lives in
vintage curves, bad-rate-by-grade tables, and stability reports. This module
loads the loan book into an in-memory SQLite database and runs eight standard
portfolio queries, demonstrating the SQL a risk DS is expected to write.

Using SQLite keeps it dependency-free and runnable anywhere; the SQL itself is
standard and would port to Postgres/Snowflake with minimal change.
"""

import sqlite3
from pathlib import Path

import pandas as pd

# Each query is named and documented so the output reads as a risk report.
QUERIES = {
    "01_vintage_volume": """
        -- Loan volume and average amount by issue year (vintage)
        SELECT
            substr(issue_d, -4) AS vintage_year,
            COUNT(*)                        AS n_loans,
            ROUND(AVG(loan_amnt), 0)        AS avg_loan_amnt,
            ROUND(SUM(loan_amnt), 0)        AS total_exposure
        FROM loans
        GROUP BY vintage_year
        ORDER BY vintage_year;
    """,

    "02_bad_rate_by_grade": """
        -- Default (bad) rate by assigned grade, resolved loans only
        SELECT
            grade,
            COUNT(*)                                                       AS n_resolved,
            SUM(CASE WHEN loan_status = 'Charged Off' THEN 1 ELSE 0 END)   AS n_bad,
            ROUND(AVG(CASE WHEN loan_status = 'Charged Off' THEN 1.0 ELSE 0 END), 4) AS bad_rate
        FROM loans
        WHERE loan_status IN ('Charged Off', 'Fully Paid')
        GROUP BY grade
        ORDER BY grade;
    """,

    "03_bad_rate_by_vintage": """
        -- Bad rate by vintage year: the vintage curve risk teams track
        SELECT
            substr(issue_d, -4) AS vintage_year,
            COUNT(*)                                                       AS n_resolved,
            ROUND(AVG(CASE WHEN loan_status = 'Charged Off' THEN 1.0 ELSE 0 END), 4) AS bad_rate
        FROM loans
        WHERE loan_status IN ('Charged Off', 'Fully Paid')
        GROUP BY vintage_year
        ORDER BY vintage_year;
    """,

    "04_resolution_rate_by_vintage": """
        -- Share of loans that have reached a final outcome, by vintage
        -- (surfaces the right-censoring of recent vintages)
        SELECT
            substr(issue_d, -4) AS vintage_year,
            COUNT(*)                                                                AS n_total,
            SUM(CASE WHEN loan_status IN ('Charged Off','Fully Paid') THEN 1 ELSE 0 END) AS n_resolved,
            ROUND(1.0 * SUM(CASE WHEN loan_status IN ('Charged Off','Fully Paid')
                                 THEN 1 ELSE 0 END) / COUNT(*), 3)                  AS resolved_pct
        FROM loans
        GROUP BY vintage_year
        ORDER BY vintage_year;
    """,

    "05_exposure_by_grade": """
        -- Exposure concentration by grade: how much money sits in each risk band
        SELECT
            grade,
            COUNT(*)                                        AS n_loans,
            ROUND(SUM(loan_amnt), 0)                        AS exposure,
            ROUND(100.0 * SUM(loan_amnt) / (SELECT SUM(loan_amnt) FROM loans), 1) AS exposure_pct
        FROM loans
        GROUP BY grade
        ORDER BY grade;
    """,

    "06_bad_rate_by_purpose": """
        -- Bad rate by loan purpose, for the highest-volume purposes
        SELECT
            purpose,
            COUNT(*)                                                       AS n_resolved,
            ROUND(AVG(CASE WHEN loan_status = 'Charged Off' THEN 1.0 ELSE 0 END), 4) AS bad_rate
        FROM loans
        WHERE loan_status IN ('Charged Off', 'Fully Paid')
        GROUP BY purpose
        HAVING COUNT(*) >= 100
        ORDER BY bad_rate DESC;
    """,

    "07_dti_band_bad_rate": """
        -- Bad rate across debt-to-income bands: monotonic risk gradient check
        SELECT
            CASE
                WHEN dti < 10 THEN '1. <10'
                WHEN dti < 20 THEN '2. 10-20'
                WHEN dti < 30 THEN '3. 20-30'
                ELSE '4. 30+'
            END AS dti_band,
            COUNT(*)                                                       AS n_resolved,
            ROUND(AVG(CASE WHEN loan_status = 'Charged Off' THEN 1.0 ELSE 0 END), 4) AS bad_rate
        FROM loans
        WHERE loan_status IN ('Charged Off', 'Fully Paid') AND dti IS NOT NULL
        GROUP BY dti_band
        ORDER BY dti_band;
    """,

    "08_state_concentration": """
        -- Top 10 states by exposure: geographic concentration risk
        SELECT
            addr_state,
            COUNT(*)                        AS n_loans,
            ROUND(SUM(loan_amnt), 0)        AS exposure,
            ROUND(AVG(CASE WHEN loan_status = 'Charged Off' THEN 1.0 ELSE 0 END), 4) AS bad_rate
        FROM loans
        WHERE loan_status IN ('Charged Off', 'Fully Paid')
        GROUP BY addr_state
        ORDER BY exposure DESC
        LIMIT 10;
    """,
}


def build_database(sample_path: str = "data/sample.csv") -> sqlite3.Connection:
    """Load the loan book into an in-memory SQLite database."""
    df = pd.read_csv(sample_path, low_memory=False)
    conn = sqlite3.connect(":memory:")
    df.to_sql("loans", conn, index=False)
    return conn


def run_query(conn: sqlite3.Connection, name: str) -> pd.DataFrame:
    """Run one named query and return its result."""
    return pd.read_sql(QUERIES[name], conn)


def run_all(sample_path: str = "data/sample.csv") -> dict:
    """Run every query and return {name: DataFrame}."""
    conn = build_database(sample_path)
    return {name: run_query(conn, name) for name in QUERIES}


if __name__ == "__main__":
    results = run_all()
    for name, df in results.items():
        print(f"\n===== {name} =====")
        print(df.to_string(index=False))
