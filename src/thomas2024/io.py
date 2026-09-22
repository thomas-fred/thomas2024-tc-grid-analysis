from pathlib import Path

import pandas as pd


def read_wb_income(path: Path, financial_year: int) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Country Analytical History", skiprows=4)
    df = (
        df
        .iloc[1:, :]
        .rename(columns={"Unnamed: 0": "iso_a3"})
        .set_index("iso_a3")
        .drop(columns="Bank's fiscal year:")
        .loc[:, f"FY{str(financial_year)[-2:]}"]
        .dropna()
    )
    return df

