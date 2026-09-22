"""
Confusion matrix module.

This module generates confusion matrices for
cognitive-decline classification models.

Class labels:
0 = Healthy Control
1 = Cognitive Decline
"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """
    Generate and display a confusion matrix.

    Parameters:
        y_true: Actual class labels.
        y_pred: Predicted class labels.
        model_name: Name of the model being evaluated.
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Healthy Control",
            "Cognitive Decline"
        ]
    )

    display.plot()

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.show()

    return cm
