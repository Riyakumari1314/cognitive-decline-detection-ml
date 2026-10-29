"""
Typing feature extraction module.
"""

import pandas as pd


def extract_typing_features(typing_data: pd.DataFrame) -> pd.DataFrame:
    """Extract the 8 keystroke and typing features."""
    features = pd.DataFrame(index=typing_data.index)

    features["typing_speed"] = (typing_data["total_keystrokes"] / typing_data["session_duration_sec"]) * 60.0
    features["mean_inter_key_interval"] = typing_data["mean_iki_ms"]
    features["pause_frequency"] = (typing_data["typing_pauses"] / typing_data["session_duration_sec"]) * 60.0
    features["mean_pause_duration"] = typing_data["total_typing_pause_sec"] / typing_data["typing_pauses"].clip(lower=1)
    features["backspace_frequency"] = typing_data["backspace_count"] / typing_data["total_keystrokes"]
    features["correction_rate"] = typing_data["corrected_errors"] / typing_data["total_keystrokes"]
    total_errors = typing_data["corrected_errors"] + typing_data["uncorrected_errors"]
    features["error_rate"] = total_errors / typing_data["total_keystrokes"]
    features["session_duration"] = typing_data["session_duration_sec"]

    return features
    