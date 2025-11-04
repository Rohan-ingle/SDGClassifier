"""
Model validation pipeline for SDG Classification
"""

import pandas as pd
import numpy as np
import pickle
import json
import yaml
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelValidator:
    """Model validation on validation set"""

    def __init__(self, params):
        self.params = params
        self.results = {}

    def load_model_and_data(self):
        """Load trained model and validation data"""
        logger.info("Loading model and validation data...")

        # Load model
        with open("models/model.pkl", "rb") as f:
            self.model = pickle.load(f)

        # Load validation data
        with open("data/processed/X_val.pkl", "rb") as f:
            self.X_val = pickle.load(f)

        with open("data/processed/y_val.pkl", "rb") as f:
            self.y_val = pickle.load(f)

        # Load encoders
        with open("data/processed/label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        logger.info(f"Validation data loaded: {self.X_val.shape}")

    def validate_model(self):
        """Validate model on validation set"""
        logger.info("Validating model...")

        # Make predictions
        y_pred = self.model.predict(self.X_val)

        # Calculate metrics
        accuracy = accuracy_score(self.y_val, y_pred)
        precision_macro = precision_score(self.y_val, y_pred, average="macro")
        recall_macro = recall_score(self.y_val, y_pred, average="macro")
        f1_macro = f1_score(self.y_val, y_pred, average="macro")

        # Generate classification report
        class_names = [f"SDG_{cls}" for cls in self.label_encoder.classes_]
        report = classification_report(
            self.y_val, y_pred, target_names=class_names, output_dict=True
        )

        # Store results
        self.results = {
            "validation_accuracy": float(accuracy),
            "validation_precision_macro": float(precision_macro),
            "validation_recall_macro": float(recall_macro),
            "validation_f1_macro": float(f1_macro),
            "validation_classification_report": report,
            "num_validation_samples": int(len(self.y_val)),
            "validation_timestamp": pd.Timestamp.now().isoformat(),
        }

        logger.info(f"Validation Accuracy: {accuracy:.4f}")
        logger.info(f"Validation F1 (Macro): {f1_macro:.4f}")

    def save_validation_results(self):
        """Save validation results"""
        logger.info("Saving validation results...")

        os.makedirs("metrics", exist_ok=True)

        # Save JSON results
        with open("metrics/validation_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

        # Save text report
        with open("metrics/validation_report.txt", "w") as f:
            f.write("SDG Classification Model - Validation Set Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Validation Accuracy: {self.results['validation_accuracy']:.4f}\n")
            f.write(f"Validation F1 (Macro): {self.results['validation_f1_macro']:.4f}\n")
            f.write(
                f"Validation Precision (Macro): {self.results['validation_precision_macro']:.4f}\n"
            )
            f.write(f"Validation Recall (Macro): {self.results['validation_recall_macro']:.4f}\n")
            f.write(f"Number of Validation Samples: {self.results['num_validation_samples']}\n")

        logger.info("Validation results saved")


def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def main():
    """Main validation pipeline"""
    logger.info("Starting model validation pipeline...")

    # Load parameters
    params = load_params()

    # Initialize validator
    validator = ModelValidator(params)

    # Load model and validation data
    validator.load_model_and_data()

    # Validate model
    validator.validate_model()

    # Save results
    validator.save_validation_results()

    logger.info("Model validation pipeline completed successfully!")


if __name__ == "__main__":
    main()
