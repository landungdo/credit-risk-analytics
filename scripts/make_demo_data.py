"""
Generate a small synthetic Lending-Club-shaped dataset for a self-contained
Docker build / demo. This lets `docker build` succeed from a clean clone
(where the real, licensed data is not present) by producing data/sample.csv
with the same schema and the relationships the model expects.

This is DEMO data only — it exists so the container can train and serve end to
end without shipping licensed data. For real results, replace data/sample.csv
with a sample of the actual Lending Club dataset (see scripts/make_sample.py).
"""

import sys
from pathlib import Path

# Reuse the exact synthetic generator used by the test fixture, so demo data and
# test data share one definition.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.conftest import _make_synthetic

OUTPUT = Path("data/sample.csv")


def main(n: int = 6000):
    OUTPUT.parent.mkdir(exist_ok=True)
    df = _make_synthetic(n=n, seed=7)
    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(df)} synthetic demo rows to {OUTPUT}")


if __name__ == "__main__":
    main()
