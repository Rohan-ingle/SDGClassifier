"""
Train all supported models and produce a comparison visualization for the poster.

This script iterates over algorithms, trains each model using the existing
`SDGModelTrainer` class (saving artifacts to `models/<algorithm>/`), evaluates on
the test set and produces a visual comparison saved under `reports/model_comparison.png`.
"""

import copy
import json
import logging
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Import trainer from the local package
from src.models.train import SDGModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_ALGORITHMS = ["random_forest", "svm", "logistic_regression", "neural_network"]

REPORT_DIR = "reports"
METRICS_DIR = "metrics"


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def evaluate_model_on_test(model_path, processed_path):
    # Load model and test data and compute simple metrics
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(processed_path, "X_test.pkl"), "rb") as f:
        X_test = pickle.load(f)
    with open(os.path.join(processed_path, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")

    return {
        "test_accuracy": float(acc),
        "f1_macro": float(f1),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
    }


def plot_comparison(all_metrics, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sns.set(style="whitegrid")

    algorithms = [m["algorithm"] for m in all_metrics]
    train_acc = [m.get("train_accuracy", np.nan) for m in all_metrics]
    val_acc = [m.get("validation_accuracy", np.nan) for m in all_metrics]
    test_acc = [m.get("test_accuracy", np.nan) for m in all_metrics]
    f1 = [m.get("f1_macro", np.nan) for m in all_metrics]

    x = np.arange(len(algorithms))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width, train_acc, width, label="Train Acc", color="#4c78a8")
    ax.bar(x, val_acc, width, label="Val Acc", color="#f58518")
    ax.bar(x + width, test_acc, width, label="Test Acc", color="#e45756")

    # Add F1 as markers
    for i, v in enumerate(f1):
        ax.plot([i], [v], marker="D", color="#5c4d7d")
        ax.text(i + 0.03, v + 0.005, f"F1={v:.2f}", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", " ").title() for a in algorithms])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison — Accuracy and F1 (macro)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    params = load_params()
    processed_path = params["data"]["processed_data_path"]

    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    all_metrics = []

    for algo in SUPPORTED_ALGORITHMS:
        logger.info(f"Training algorithm: {algo}")
        params_copy = copy.deepcopy(params)
        params_copy["model"]["algorithm"] = algo

        model_dir = os.path.join("models", algo)
        trainer = SDGModelTrainer(params_copy, model_dir=model_dir)

        # Load data, create model and train
        trainer.load_processed_data()
        trainer.create_model()
        trainer.train_model()
        trainer.generate_predictions_and_reports()
        trainer.save_model_and_metrics()

        # Evaluate on test set
        model_path = os.path.join(model_dir, "model.pkl")
        test_metrics = evaluate_model_on_test(model_path, processed_path)

        # Merge metrics
        metrics_summary = {
            "algorithm": algo,
            "train_accuracy": trainer.training_history.get("train_accuracy"),
            "validation_accuracy": trainer.training_history.get("validation_accuracy"),
            "cv_mean_accuracy": trainer.training_history.get("cv_mean_accuracy"),
            **test_metrics,
        }

        # Save per-algorithm metrics to metrics/
        metrics_file = os.path.join(METRICS_DIR, f"{algo}_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics_summary, f, indent=2)

        all_metrics.append(metrics_summary)

    # Plot comparison
    out_path = os.path.join(REPORT_DIR, "model_comparison.png")
    plot_comparison(all_metrics, out_path)

    # Save aggregated metrics
    with open(os.path.join(METRICS_DIR, "model_comparison.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"Model comparison chart saved to {out_path}")
    logger.info("All trainings completed")


if __name__ == "__main__":
    main()
