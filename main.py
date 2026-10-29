"""
main.py
Runs the end-to-end preprocessing, feature extraction, 5-fold cross-validation,
hold-out testing, and confusion matrix visualization.
"""

import os
import pandas as pd
from download_benchmarks import build_benchmark_files
from src.preprocessing.speech_preprocessing import preprocess_speech
from src.preprocessing.typing_preprocessing import preprocess_typing
from src.features.speech_features import extract_speech_features
from src.features.typing_features import extract_typing_features
from src.models.speech_model import train_speech_model
from src.models.typing_model import train_typing_model
from src.models.multimodal_model import train_multimodal_model
from src.evaluation.confusion_matrix import plot_confusion_matrix


def run_pipeline():
    speech_path = "data/speech_benchmark.csv"
    typing_path = "data/typing_benchmark.csv"
    if not (os.path.exists(speech_path) and os.path.exists(typing_path)):
        build_benchmark_files(n_per_class=120)

    speech_raw = pd.read_csv(speech_path)
    typing_raw = pd.read_csv(typing_path)

    speech_clean = preprocess_speech(speech_raw)
    typing_clean = preprocess_typing(typing_raw)

    X_speech = extract_speech_features(speech_clean)
    X_typing = extract_typing_features(typing_clean)
    y = speech_clean["label"]

    _, sp_res = train_speech_model(X_speech, y)
    _, ty_res = train_typing_model(X_typing, y)
    _, mm_res = train_multimodal_model(X_speech, X_typing, y)

    summary_rows = []
    for name, res in [
        ("Speech Unimodal", sp_res),
        ("Typing Unimodal", ty_res),
        ("Multimodal Fusion", mm_res)
    ]:
        summary_rows.append({
            "Model": name,
            "5-Fold CV F1": f"{res['cv_f1_mean']:.4f} ± {res['cv_f1_std']:.4f}",
            "Test Accuracy": round(res["accuracy"], 4),
            "Test Precision": round(res["precision"], 4),
            "Test Recall": round(res["recall"], 4),
            "Test F1-Score": round(res["f1_score"], 4),
            "Test ROC-AUC": round(res["roc_auc"], 4),
        })

    print("\n================ BENCHMARK EVALUATION SUMMARY ================")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("==============================================================\n")

    for name, res in [
        ("Speech Model", sp_res),
        ("Typing Model", ty_res),
        ("Multimodal Model", mm_res)
    ]:
        plot_confusion_matrix(res["y_test"], res["y_pred"], model_name=name)


if __name__ == "__main__":
    run_pipeline()