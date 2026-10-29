"""
Typing preprocessing module.
"""

import pandas as pd


def preprocess_typing(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate raw typing/keystroke benchmark data."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df["session_duration_sec"] = df["session_duration_sec"].clip(lower=1.0)
    df["total_keystrokes"] = df["total_keystrokes"].clip(lower=1)
    return df