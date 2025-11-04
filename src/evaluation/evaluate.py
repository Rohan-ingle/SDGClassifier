"""
Model evaluation pipeline for SDG Classification
"""

import pandas as pd
import numpy as np
import pickle
import json
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import cross_val_score
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use("default")
sns.set_palette("husl")


class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(self, params):
        self.params = params
        self.results = {}

    def load_model_and_data(self):
        """Load trained model and test data"""
        logger.info("Loading model and test data...")

        # Load model
        with open("models/model.pkl", "rb") as f:
            self.model = pickle.load(f)

        # Load test data
        with open("data/processed/X_test.pkl", "rb") as f:
            self.X_test = pickle.load(f)

        with open("data/processed/y_test.pkl", "rb") as f:
            self.y_test = pickle.load(f)

        # Load encoders
        with open("data/processed/vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

        with open("data/processed/label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        logger.info(f"Test data loaded: {self.X_test.shape}")
        logger.info(f"Number of test samples: {len(self.y_test)}")

    def evaluate_model_performance(self):
        """Evaluate model performance on test set"""
        logger.info("Evaluating model performance...")

        # Make predictions
        y_pred = self.model.predict(self.X_test)

        # Get prediction probabilities if available
        y_pred_proba = None
        if hasattr(self.model, "predict_proba"):
            y_pred_proba = self.model.predict_proba(self.X_test)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision_macro = precision_score(self.y_test, y_pred, average="macro")
        recall_macro = recall_score(self.y_test, y_pred, average="macro")
        f1_macro = f1_score(self.y_test, y_pred, average="macro")

        precision_weighted = precision_score(self.y_test, y_pred, average="weighted")
        recall_weighted = recall_score(self.y_test, y_pred, average="weighted")
        f1_weighted = f1_score(self.y_test, y_pred, average="weighted")

        # Store predictions and probabilities
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

        # Store results
        self.results.update(
            {
                "test_accuracy": float(accuracy),
                "precision_macro": float(precision_macro),
                "recall_macro": float(recall_macro),
                "f1_macro": float(f1_macro),
                "precision_weighted": float(precision_weighted),
                "recall_weighted": float(recall_weighted),
                "f1_weighted": float(f1_weighted),
                "num_test_samples": int(len(self.y_test)),
                "num_classes": int(len(self.label_encoder.classes_)),
            }
        )

        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score (Macro): {f1_macro:.4f}")
        logger.info(f"F1 Score (Weighted): {f1_weighted:.4f}")

    def generate_classification_report(self):
        """Generate detailed classification report"""
        logger.info("Generating classification report...")

        # Class names
        class_names = [f"SDG_{cls}" for cls in self.label_encoder.classes_]

        # Classification report as dictionary
        report_dict = classification_report(
            self.y_test, self.y_pred, target_names=class_names, output_dict=True
        )

        # Classification report as string
        report_str = classification_report(self.y_test, self.y_pred, target_names=class_names)

        # Store results
        self.results["classification_report"] = report_dict
        self.classification_report_str = report_str

        # Save text report
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/classification_report.txt", "w") as f:
            f.write("SDG Classification Model - Test Set Evaluation\n")
            f.write("=" * 50 + "\n\n")
            f.write(report_str)
            f.write("\n\nPer-Class Metrics:\n")
            f.write("-" * 30 + "\n")

            for i, class_name in enumerate(class_names):
                sdg_num = self.label_encoder.classes_[i]
                class_metrics = report_dict[class_name]
                f.write(f"\nSDG {sdg_num}:\n")
                f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {class_metrics['f1-score']:.4f}\n")
                f.write(f"  Support: {class_metrics['support']}\n")

        logger.info("Classification report saved to metrics/classification_report.txt")

    def create_confusion_matrix_plot(self):
        """Create and save confusion matrix visualization"""
        logger.info("Creating confusion matrix plot...")

        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)

        # Create figure
        plt.figure(figsize=(12, 10))

        # Plot confusion matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[f"SDG {cls}" for cls in self.label_encoder.classes_],
            yticklabels=[f"SDG {cls}" for cls in self.label_encoder.classes_],
        )

        plt.title("Confusion Matrix - SDG Classification", fontsize=16, fontweight="bold")
        plt.xlabel("Predicted SDG", fontsize=12)
        plt.ylabel("Actual SDG", fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save plot
        plt.savefig("metrics/confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Store confusion matrix in results
        self.results["confusion_matrix"] = cm.tolist()

        logger.info("Confusion matrix saved to metrics/confusion_matrix.png")

    def create_feature_importance_plot(self):
        """Create feature importance plot if available"""
        logger.info("Creating feature importance plot...")

        feature_importance = None

        # Get feature importance based on model type
        if hasattr(self.model, "feature_importances_"):
            feature_importance = self.model.feature_importances_
            importance_type = "Feature Importance"
        elif hasattr(self.model, "coef_") and self.model.coef_.ndim == 2:
            # For linear models, take mean absolute coefficients
            feature_importance = np.mean(np.abs(self.model.coef_), axis=0)
            importance_type = "Mean Absolute Coefficient"

        if feature_importance is not None:
            # Get feature names from vectorizer
            feature_names = self.vectorizer.get_feature_names_out()

            # Get top 20 most important features
            top_indices = np.argsort(feature_importance)[-20:]
            top_features = feature_names[top_indices]
            top_importance = feature_importance[top_indices]

            # Create plot
            plt.figure(figsize=(10, 8))
            bars = plt.barh(range(len(top_features)), top_importance)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel(importance_type)
            plt.title(f"Top 20 Features - {importance_type}", fontsize=14, fontweight="bold")
            plt.gca().invert_yaxis()

            # Color bars
            for bar in bars:
                bar.set_color(plt.cm.viridis(bar.get_width() / max(top_importance)))

            plt.tight_layout()
            plt.savefig("metrics/feature_importance.png", dpi=300, bbox_inches="tight")
            plt.close()

            # Store feature importance
            self.results["top_features"] = {
                "features": top_features.tolist(),
                "importance_scores": top_importance.tolist(),
                "importance_type": importance_type,
            }

            logger.info("Feature importance plot saved to metrics/feature_importance.png")
        else:
            logger.info("Feature importance not available for this model type")
            # Create empty plot to satisfy DVC pipeline
            plt.figure(figsize=(8, 6))
            plt.text(
                0.5,
                0.5,
                "Feature importance not available for this model type",
                ha="center",
                va="center",
                fontsize=12,
            )
            plt.axis("off")
            plt.title("Feature Importance", fontsize=14, fontweight="bold")
            plt.savefig("metrics/feature_importance.png", dpi=300, bbox_inches="tight")
            plt.close()

    def perform_cross_validation(self):
        """Perform cross-validation analysis"""
        logger.info("Performing cross-validation analysis...")

        # Load training data for cross-validation
        with open("data/processed/X_train.pkl", "rb") as f:
            X_train = pickle.load(f)

        with open("data/processed/y_train.pkl", "rb") as f:
            y_train = pickle.load(f)

        # Perform cross-validation with different metrics
        cv_folds = self.params["evaluation"]["cv_folds"]

        cv_results = {}
        for metric in self.params["evaluation"]["scoring_metrics"]:
            scores = cross_val_score(
                self.model, X_train, y_train, cv=cv_folds, scoring=metric, n_jobs=-1
            )
            cv_results[f"cv_{metric}"] = {
                "scores": scores.tolist(),
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "min": float(scores.min()),
                "max": float(scores.max()),
            }

        self.results["cross_validation"] = cv_results

        logger.info("Cross-validation completed")
        for metric, result in cv_results.items():
            logger.info(f"{metric}: {result['mean']:.4f} (+/- {result['std'] * 2:.4f})")

    def save_evaluation_results(self):
        """Save all evaluation results"""
        logger.info("Saving evaluation results...")

        # Add timestamp and model info
        self.results.update(
            {
                "evaluation_timestamp": pd.Timestamp.now().isoformat(),
                "model_type": type(self.model).__name__,
                "model_parameters": self.model.get_params(),
            }
        )

        # Save results to JSON
        with open("metrics/evaluation_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info("Evaluation results saved to metrics/evaluation_results.json")

        # Print evaluation summary
        logger.info("\n" + "=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Test Accuracy: {self.results['test_accuracy']:.4f}")
        logger.info(f"F1 Score (Macro): {self.results['f1_macro']:.4f}")
        logger.info(f"F1 Score (Weighted): {self.results['f1_weighted']:.4f}")
        logger.info(f"Precision (Macro): {self.results['precision_macro']:.4f}")
        logger.info(f"Recall (Macro): {self.results['recall_macro']:.4f}")
        logger.info(f"Number of Test Samples: {self.results['num_test_samples']}")
        logger.info(f"Number of Classes: {self.results['num_classes']}")
        logger.info("=" * 50)


def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def main():
    """Main evaluation pipeline"""
    logger.info("Starting model evaluation pipeline...")

    # Load parameters
    params = load_params()

    # Initialize evaluator
    evaluator = ModelEvaluator(params)

    # Load model and test data
    evaluator.load_model_and_data()

    # Evaluate model performance
    evaluator.evaluate_model_performance()

    # Generate classification report
    evaluator.generate_classification_report()

    # Create visualizations
    evaluator.create_confusion_matrix_plot()
    evaluator.create_feature_importance_plot()

    # Perform cross-validation
    evaluator.perform_cross_validation()

    # Save results
    evaluator.save_evaluation_results()

    logger.info("Model evaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()
