"""
Model export and packaging for deployment
"""

import pickle
import json
import yaml
import os
import joblib
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExporter:
    """Export trained models for deployment"""

    def __init__(self, params):
        self.params = params

    def load_artifacts(self):
        """Load trained model and associated artifacts"""
        logger.info("Loading model artifacts...")

        # Load trained model
        with open("models/model.pkl", "rb") as f:
            self.model = pickle.load(f)

        # Load vectorizer
        with open("data/processed/vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

        # Load label encoder
        with open("data/processed/label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        # Load training metrics
        with open("models/training_metrics.json", "r") as f:
            self.training_metrics = json.load(f)

        # Load evaluation results
        with open("metrics/evaluation_results.json", "r") as f:
            self.evaluation_results = json.load(f)

        logger.info("All artifacts loaded successfully")

    def create_inference_pipeline(self):
        """Create a complete inference pipeline"""
        logger.info("Creating inference pipeline...")

        # Create inference pipeline as a simple function instead of nested class
        def inference_pipeline(text_data):
            """Predict SDG classification for text data"""
            if isinstance(text_data, str):
                text_data = [text_data]

            # Vectorize text
            X_vec = self.vectorizer.transform(text_data)

            # Predict
            predictions = self.model.predict(X_vec)

            # Get probabilities if available
            probabilities = None
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(X_vec)

            # Decode labels
            predicted_sdgs = self.label_encoder.inverse_transform(predictions)

            results = []
            for i, sdg in enumerate(predicted_sdgs):
                result = {
                    "predicted_sdg": int(sdg),
                    "confidence": (
                        float(probabilities[i].max()) if probabilities is not None else None
                    ),
                }

                if probabilities is not None:
                    # Add top 3 predictions with probabilities
                    top_indices = probabilities[i].argsort()[-3:][::-1]
                    result["top_predictions"] = [
                        {
                            "sdg": int(self.label_encoder.classes_[idx]),
                            "probability": float(probabilities[i][idx]),
                        }
                        for idx in top_indices
                    ]

                results.append(result)

            return results[0] if len(results) == 1 else results

        # Store as a simple inference function with model components
        self.inference_pipeline = {
            "model": self.model,
            "vectorizer": self.vectorizer,
            "label_encoder": self.label_encoder,
            "predict_function": inference_pipeline,
        }

        logger.info("Inference pipeline created")

    def create_model_metadata(self):
        """Create comprehensive model metadata"""
        logger.info("Creating model metadata...")

        metadata = {
            "model_info": {
                "name": self.params["deployment"]["model_name"],
                "version": self.params["deployment"]["model_version"],
                "algorithm": self.params["model"]["algorithm"],
                "created_at": datetime.now().isoformat(),
                "description": "SDG Classification Model for Research Papers",
            },
            "data_info": {
                "training_samples": self.training_metrics.get("train_accuracy"),
                "num_features": self.vectorizer.max_features,
                "num_classes": len(self.label_encoder.classes_),
                "class_names": self.label_encoder.classes_.tolist(),
            },
            "performance_metrics": {
                "train_accuracy": self.training_metrics.get("train_accuracy"),
                "validation_accuracy": self.training_metrics.get("validation_accuracy"),
                "test_accuracy": self.evaluation_results.get("test_accuracy"),
                "cv_mean_accuracy": self.training_metrics.get("cv_mean_accuracy"),
                "f1_macro": self.evaluation_results.get("f1_macro"),
                "precision_macro": self.evaluation_results.get("precision_macro"),
                "recall_macro": self.evaluation_results.get("recall_macro"),
            },
            "model_parameters": self.training_metrics.get("model_params", {}),
            "preprocessing_info": {
                "vectorizer_type": "TfidfVectorizer",
                "max_features": self.params["preprocessing"]["max_features"],
                "ngram_range": self.params["preprocessing"]["ngram_range"],
                "min_df": self.params["preprocessing"]["min_df"],
                "max_df": self.params["preprocessing"]["max_df"],
            },
            "deployment_config": {
                "serve_port": self.params["deployment"]["serve_port"],
                "max_batch_size": self.params["deployment"]["max_prediction_batch_size"],
            },
        }

        self.metadata = metadata
        logger.info("Model metadata created")

    def export_final_model(self):
        """Export final model package"""
        logger.info("Exporting final model package...")

        # Save final model (using joblib for better performance)
        joblib.dump(self.model, "models/final_model.pkl")

        # Save inference pipeline
        inference_components = {
            "model": self.model,
            "vectorizer": self.vectorizer,
            "label_encoder": self.label_encoder,
        }
        with open("models/inference_pipeline.pkl", "wb") as f:
            pickle.dump(inference_components, f)

        # Save metadata
        with open("models/model_metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info("Final model package exported successfully")

        # Print export summary
        logger.info("\n" + "=" * 50)
        logger.info("MODEL EXPORT SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Model Name: {self.metadata['model_info']['name']}")
        logger.info(f"Version: {self.metadata['model_info']['version']}")
        logger.info(f"Algorithm: {self.metadata['model_info']['algorithm']}")
        logger.info(f"Test Accuracy: {self.metadata['performance_metrics']['test_accuracy']:.4f}")
        logger.info(f"F1 Macro: {self.metadata['performance_metrics']['f1_macro']:.4f}")
        logger.info("Files exported:")
        logger.info("  - models/final_model.pkl")
        logger.info("  - models/inference_pipeline.pkl")
        logger.info("  - models/model_metadata.json")
        logger.info("=" * 50)


def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def main():
    """Main export pipeline"""
    logger.info("Starting model export pipeline...")

    # Load parameters
    params = load_params()

    # Initialize exporter
    exporter = ModelExporter(params)

    # Load artifacts
    exporter.load_artifacts()

    # Create inference pipeline
    exporter.create_inference_pipeline()

    # Create metadata
    exporter.create_model_metadata()

    # Export final model
    exporter.export_final_model()

    logger.info("Model export pipeline completed successfully!")


if __name__ == "__main__":
    main()
