# WHO= Amanda
#   container for metric functions

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Optional: show full confusion matrix in terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def acc(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def class_report(y_true, y_pred, class_names=None):
    if class_names is not None:
        return classification_report(y_true, y_pred, target_names=class_names)
    return classification_report(y_true, y_pred)


def conf_matrix(y_true, y_pred, class_names=None, model_name="Model", save_path=None):
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

    return cm


def evaluate_model(y_true, y_pred, class_names=None, model_name="Model", save_path=None):
    print(f"\n{model_name}")
    print("Accuracy:", acc(y_true, y_pred))
    print(class_report(y_true, y_pred, class_names=class_names))

    conf_matrix(
        y_true,
        y_pred,
        class_names=class_names,
        model_name=model_name,
        save_path=save_path
    )


def print_conf_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    print("\nConfusion Matrix:")
    print(df_cm)
