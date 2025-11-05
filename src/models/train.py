"""
Model training pipeline for SDG Classification
"""

import json
import logging
import os
import pickle
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# Setup centralized logging
import sys
from pathlib import Path
try:
    from ..utils.logging_config import setup_logging
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.utils.logging_config import setup_logging

logger = setup_logging(log_dir="logs", module_name="train", log_level="INFO")


class SDGModelTrainer:
    """SDG Classification Model Trainer"""

    def __init__(self, params, model_dir: str = "models"):
        self.params = params
        self.model = None
        self.training_history = {}
        # directory where model and metrics will be saved for this trainer
        self.model_dir = model_dir

    def load_processed_data(self):
        """Load preprocessed data"""
        logger.info("Loading processed data...")

        processed_path = self.params["data"]["processed_data_path"]

        with open(os.path.join(processed_path, "X_train.pkl"), "rb") as f:
            self.X_train = pickle.load(f)

        with open(os.path.join(processed_path, "X_val.pkl"), "rb") as f:
            self.X_val = pickle.load(f)

        with open(os.path.join(processed_path, "y_train.pkl"), "rb") as f:
            self.y_train = pickle.load(f)

        with open(os.path.join(processed_path, "y_val.pkl"), "rb") as f:
            self.y_val = pickle.load(f)

        with open(os.path.join(processed_path, "vectorizer.pkl"), "rb") as f:
            self.vectorizer = pickle.load(f)

        with open(os.path.join(processed_path, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)

        # Load data statistics
        with open(os.path.join(processed_path, "data_stats.json"), "r") as f:
            self.data_stats = json.load(f)

        logger.info(f"Loaded training data: {self.X_train.shape}")
        logger.info(f"Loaded validation data: {self.X_val.shape}")
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")

    def create_model(self):
        """Create model based on algorithm specified in params"""
        algorithm = self.params["model"]["algorithm"]
        logger.info(f"Creating {algorithm} model...")

        if algorithm == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.params["model"]["n_estimators"],
                max_depth=self.params["model"]["max_depth"],
                min_samples_split=self.params["model"]["min_samples_split"],
                min_samples_leaf=self.params["model"]["min_samples_leaf"],
                class_weight=self.params["model"]["class_weight"],
                random_state=self.params["data"]["random_state"],
                n_jobs=-1,
            )

        elif algorithm == "svm":
            self.model = SVC(
                kernel="rbf",
                C=1.0,
                class_weight=self.params["model"]["class_weight"],
                random_state=self.params["data"]["random_state"],
                probability=True,  # Enable probability estimates
            )

        elif algorithm == "logistic_regression":
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight=self.params["model"]["class_weight"],
                random_state=self.params["data"]["random_state"],
                n_jobs=-1,
            )

        elif algorithm == "neural_network":
            self.model = MLPClassifier(
                hidden_layer_sizes=tuple(self.params["neural_network"]["hidden_layers"]),
                learning_rate_init=self.params["neural_network"]["learning_rate"],
                max_iter=self.params["neural_network"]["epochs"],
                random_state=self.params["data"]["random_state"],
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
            )

        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        logger.info(f"Model created: {type(self.model).__name__}")
        return self.model

    def train_model(self):
        """Train the model"""
        logger.info("Starting model training...")
        start_time = datetime.now()

        # Train the model
        if self.params["model"]["algorithm"] == "neural_network":
            # For neural networks, use validation set for early stopping
            self.model.fit(self.X_train, self.y_train)

            # Track training history for neural networks
            if hasattr(self.model, "loss_curve_"):
                self.training_history["loss_curve"] = self.model.loss_curve_.tolist()
            if hasattr(self.model, "validation_scores_"):
                self.training_history["validation_scores"] = self.model.validation_scores_.tolist()
        else:
            # For other algorithms, just fit on training data
            self.model.fit(self.X_train, self.y_train)

        training_time = (datetime.now() - start_time).total_seconds()

        logger.info(f"Model training completed in {training_time:.2f} seconds")

        # Evaluate on training and validation sets
        train_score = self.model.score(self.X_train, self.y_train)
        val_score = self.model.score(self.X_val, self.y_val)

        logger.info(f"Training accuracy: {train_score:.4f}")
        logger.info(f"Validation accuracy: {val_score:.4f}")

        # Cross-validation on training set
        cv_scores = cross_val_score(
            self.model,
            self.X_train,
            self.y_train,
            cv=self.params["evaluation"]["cv_folds"],
            scoring="accuracy",
            n_jobs=-1,
        )

        logger.info(
            f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )

        # Store training metrics
        self.training_history.update(
            {
                "training_time_seconds": training_time,
                "train_accuracy": float(train_score),
                "validation_accuracy": float(val_score),
                "cv_mean_accuracy": float(cv_scores.mean()),
                "cv_std_accuracy": float(cv_scores.std()),
                "cv_scores": cv_scores.tolist(),
                "model_params": self.model.get_params(),
                "training_timestamp": datetime.now().isoformat(),
            }
        )

    def generate_predictions_and_reports(self):
        """Generate predictions and detailed reports"""
        logger.info("Generating predictions and reports...")

        # Predictions on validation set
        y_val_pred = self.model.predict(self.X_val)
        y_val_proba = None

        if hasattr(self.model, "predict_proba"):
            y_val_proba = self.model.predict_proba(self.X_val)

        # Classification report
        val_report = classification_report(
            self.y_val,
            y_val_pred,
            target_names=[f"SDG_{cls}" for cls in self.label_encoder.classes_],
            output_dict=True,
        )

        # Confusion matrix
        val_confusion = confusion_matrix(self.y_val, y_val_pred)

        # Feature importance (if available)
        feature_importance = None
        if hasattr(self.model, "feature_importances_"):
            feature_importance = self.model.feature_importances_.tolist()
        elif hasattr(self.model, "coef_") and self.model.coef_.ndim == 2:
            # For linear models, take mean absolute coefficients across classes
            feature_importance = np.mean(np.abs(self.model.coef_), axis=0).tolist()

        # Update training history with validation results
        self.training_history.update(
            {
                "validation_classification_report": val_report,
                "validation_confusion_matrix": val_confusion.tolist(),
                "feature_importance": feature_importance,
                "class_names": self.label_encoder.classes_.tolist(),
                "prediction_probabilities_available": y_val_proba is not None,
            }
        )

        logger.info("Predictions and reports generated successfully")

    def save_model_and_metrics(self):
        """Save trained model and training metrics"""
        logger.info("Saving model and metrics...")
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

        # Save the trained model
        model_path = os.path.join(self.model_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        # Save training metrics
        metrics_path = os.path.join(self.model_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Training metrics saved to {metrics_path}")

        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Algorithm: {self.params['model']['algorithm']}")
        logger.info(f"Training Accuracy: {self.training_history['train_accuracy']:.4f}")
        logger.info(f"Validation Accuracy: {self.training_history['validation_accuracy']:.4f}")
        logger.info(f"CV Mean Accuracy: {self.training_history['cv_mean_accuracy']:.4f}")
        logger.info(f"Training Time: {self.training_history['training_time_seconds']:.2f} seconds")
        logger.info("=" * 50)


def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def main():
    """Main training pipeline"""
    logger.info("Starting SDG classification model training...")

    # Load parameters
    params = load_params()

    # Initialize trainer
    trainer = SDGModelTrainer(params)

    # Load processed data
    trainer.load_processed_data()

    # Create model
    trainer.create_model()

    # Train model
    trainer.train_model()

    # Generate predictions and reports
    trainer.generate_predictions_and_reports()

    # Save model and metrics
    trainer.save_model_and_metrics()

    logger.info("Model training pipeline completed successfully!")


if __name__ == "__main__":
    main()
