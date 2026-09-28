"""Evaluation (CO5): accuracy, macro precision/recall/F1, confusion matrices, and results/*.json.

Every notebook saves its numbers with save_results(); nb12 and the docs read them back with
load_results(), so no number is ever typed in by hand.
"""
import json
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, \
    precision_recall_fscore_support

from src.utils import FAST, RESULTS


def classification_metrics(y_true, y_pred, label_names=None) -> dict:
    """Accuracy plus macro-averaged precision, recall and F1, and per-class scores.

    Macro = compute the metric for each class separately, then take the plain average,
    so every class counts equally however many examples it has.
    """
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True,
                                   zero_division=0)
    per_class = {k: {m: round(v[m], 4) for m in ("precision", "recall", "f1-score")}
                 for k, v in report.items() if isinstance(v, dict) and "avg" not in k}
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(p, 4),
        "macro_recall": round(r, 4),
        "macro_f1": round(f1, 4),
        "per_class": per_class,
    }


def size_mb(obj) -> float:
    """Size of a Python object (e.g. a fitted sklearn model) when pickled, in MB."""
    return round(len(pickle.dumps(obj)) / 1e6, 2)


def plot_confusion(y_true, y_pred, label_names, ax=None, title=None):
    """Confusion matrix as a heatmap: rows = true class, columns = predicted class."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
    ax = ax or plt.subplots(figsize=(4.5, 4))[1]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=label_names, yticklabels=label_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    if title:
        ax.set_title(title)
    return cm


def _to_builtin(x):
    """json.dump helper: turn NumPy numbers/arrays into plain Python types."""
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON serialisable: {type(x)}")


def save_results(name: str, data: dict):
    """Write results/<name>.json. FAST runs go to results/fast/ so they never replace real numbers."""
    folder = RESULTS / "fast" if FAST else RESULTS
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    with open(path, "w") as f:
        json.dump({"name": name, "fast_mode": FAST, **data}, f, indent=2, default=_to_builtin)
    print(f"saved {path.relative_to(RESULTS.parent)}")
    return path


def load_results() -> dict:
    """Read every results/*.json into {name: dict}."""
    return {p.stem: json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))}
