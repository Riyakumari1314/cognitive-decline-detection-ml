"""
Speech feature extraction module.
"""

import pandas as pd


def extract_speech_features(audio_data: pd.DataFrame) -> pd.DataFrame:
    """Extract the 8 speech and linguistic features."""
    features = pd.DataFrame(index=audio_data.index)

    features["speech_rate"] = (audio_data["word_count"] / audio_data["audio_duration_sec"]) * 60.0
    features["pause_frequency"] = (audio_data["pause_count"] / audio_data["audio_duration_sec"]) * 60.0
    features["mean_pause_duration"] = audio_data["total_pause_time_sec"] / audio_data["pause_count"].clip(lower=1)
    features["hesitation"] = (audio_data["hesitation_count"] / audio_data["word_count"]) * 100.0
    features["lexical_diversity"] = audio_data["unique_words"] / audio_data["word_count"]
    features["repetition"] = audio_data["repeated_words"] / audio_data["word_count"]
    features["sentence_length"] = audio_data["word_count"] / audio_data["sentence_count"]
    features["sentence_complexity"] = audio_data["subordinate_clauses"] / audio_data["sentence_count"]

    return features