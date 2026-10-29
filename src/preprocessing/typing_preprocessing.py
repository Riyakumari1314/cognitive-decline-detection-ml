
import pandas as pd


def _validate_and_impute(df, required_cols, min_values):
    """Validate features and impute numeric missing values."""
    df = df.copy()

    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {sorted(missing_cols)}"
        )

    # Keep labels and identifiers out of numeric imputation.
    feature_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in {"label"}
    ]

    for col in feature_cols:
        df[col] = df[col].fillna(df[col].median())

    for col, minimum in min_values.items():
        df[col] = df[col].clip(lower=minimum)

    if "label" in df.columns:
        if df["label"].isna().any():
            raise ValueError("Missing target labels found.")

        if not set(df["label"].unique()).issubset({0, 1}):
            raise ValueError("Expected binary labels 0 and 1.")

    return df
    def preprocess_typing(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate typing benchmark data."""
    return _validate_and_impute(
        df,
        required_cols=[
            "session_duration_sec",
            "total_keystrokes",
            "label",
        ],
        min_values={
            "session_duration_sec": 1.0,
            "total_keystrokes": 1,
        },
    )
