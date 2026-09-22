from pathlib import Path

import numpy as np
import pandas as pd

from thomas2024.io import read_wb_income


def assemble_thresholds(wb_income_path: Path, income_thresholds_path: Path, country_thresholds_path: Path) -> pd.DataFrame:
    """
    Compose the WB income level wind speed failure thresholds with any country specific ones.
    """
    # Thresholds based on income
    thresholds = read_wb_income(wb_income_path, 2024).to_frame("income_level")
    income_threshold_map = pd.read_csv(income_thresholds_path).set_index("iso_a3").to_dict()["threshold_ms-1"]
    thresholds["threshold_ms-1"] = thresholds.income_level.map(income_threshold_map)
    # Per country-calibrated thresholds
    per_country_calibrated_thresholds = pd.read_csv(country_thresholds_path).set_index("iso_a3")
    thresholds.loc[per_country_calibrated_thresholds.index, "threshold_ms-1"] = per_country_calibrated_thresholds["threshold_ms-1"]
    return thresholds


def lookup_pop_at_risk(df: pd.DataFrame, thresholds_by_country: pd.DataFrame) -> pd.DataFrame:
    """
    Given `df` with columns "iso_a3" along with "pop_20.0", "pop_22.5", "pop_25.0", ...
    and `thresholds_by_country` with "income_level" and "threshold_ms-1", return a copy of `df`
    with only the desired threshold column (renamed to "pop_at_risk").
    """
    thresh = thresholds_by_country.reset_index()[['iso_a3', 'income_level', 'threshold_ms-1']]
    df = df.merge(thresh, on='iso_a3', how='left')
    df['threshold_col'] = 'pop_' + df['threshold_ms-1'].astype(str)
    
    pop_cols = [c for c in df.columns if c.startswith('pop_')]
    col_index = {col: i for i, col in enumerate(pop_cols)}
    target_col = 'pop_' + df['threshold_ms-1'].astype(str)
    idx = target_col.map(col_index)  # NaN where no matching column exists
    values = df[pop_cols].to_numpy()
    valid = idx.notna()
    
    pop_at_risk = np.full(len(df), np.nan)
    pop_at_risk[valid] = values[np.where(valid)[0], idx[valid].astype(int)]
    
    df = df.drop(columns=[c for c in df.columns if c.startswith("pop_")])
    df["pop_at_risk"] = pop_at_risk
    df = df.drop(columns='threshold_col')

    return df
