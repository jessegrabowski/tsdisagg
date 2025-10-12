from pathlib import Path

import pandas as pd

__all__ = ["load_data"]

DATASETS = {
    "annual_sales": "tests/data/sales_a.csv",
    "quarterly_exports": "tests/data/exports_q.csv",
    "monthly_exports": "tests/data/exports_m.csv",
    "quarterly_imports": "tests/data/imports_q.csv",
}


def here(path: Path) -> Path:
    """Find the path to current project's root directory, defined as where the .git folder is located."""
    location = Path(__file__).absolute()
    for parent in location.parents:
        if (parent / ".git").exists():
            return parent / path

    raise RuntimeError("Could not find the project root directory -- no .git folder found in any parent directories.")


def load_data(dataset: str, backend: str = "pandas") -> pd.DataFrame:
    """
    Load example datasets used in Examples.ipynb.

    Parameters
    ----------
    dataset : str
        Name of the dataset to load. Must be one of "annual_sales", "quarterly_exports",
        "monthly_exports", or "quarterly_imports".

    backend: str, optional
        Backend to use for data loading. Currently only 'pandas' is supported.

    Returns
    -------
    df: DataFrame
        Dataframe of the requested example data, in the requested backend.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Dataset {dataset} not recognized. Available datasets: {DATASETS}")
    if backend != "pandas":
        raise NotImplementedError("Currently only 'pandas' backend is supported.")

    path = here(Path(DATASETS[dataset]))

    return pd.read_csv(path, index_col=0, parse_dates=True)
