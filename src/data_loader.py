from pathlib import Path

import pandas as pd

# Resolve the data folder from this file's location, so it works from anywhere
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_csv(name: str) -> pd.DataFrame:
    """Load a single CSV from the data folder by its file stem (e.g. 'products')."""
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def load_all() -> dict[str, pd.DataFrame]:
    """Load every CSV in the data folder into a dict keyed by file stem."""
    paths = sorted(DATA_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    return {path.stem: pd.read_csv(path) for path in paths}


if __name__ == "__main__":
    # Quick check when running the file directly
    for name, df in load_all().items():
        print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")