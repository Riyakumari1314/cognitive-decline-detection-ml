"""
Speech preprocessing module.
"""

import pandas as pd


def preprocess_speech(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate raw speech benchmark data."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df["audio_duration_sec"] = df["audio_duration_sec"].clip(lower=1.0)
    df["word_count"] = df["word_count"].clip(lower=1)
    df["sentence_count"] = df["sentence_count"].clip(lower=1)
    return df