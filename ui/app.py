"""
Streamlit UI for SDG Classification Model Inference
"""

import streamlit as st
import pandas as pd
import pickle
import random
import os
import sys
from pathlib import Path

# Setup logging
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.logging_config import setup_logging
logger = setup_logging(log_dir="logs", module_name="ui_app", log_level="INFO")

# SDG Mappings
SDG_NAMES = {
    1: "No Poverty",
    2: "Zero Hunger",
    3: "Good Health and Well-being",
    4: "Quality Education",
    5: "Gender Equality",
    6: "Clean Water and Sanitation",
    7: "Affordable and Clean Energy",
    8: "Decent Work and Economic Growth",
    9: "Industry, Innovation and Infrastructure",
    10: "Reduced Inequality",
    11: "Sustainable Cities and Communities",
    12: "Responsible Consumption and Production",
    13: "Climate Action",
    14: "Life Below Water",
    15: "Life on Land",
    16: "Peace and Justice Strong Institutions"
}

class SDGClassifierApp:
    """Streamlit app for SDG classification"""

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.sample_texts = []
        self.load_model()
        self.load_sample_texts()

    def load_model(self):
        """Load the trained inference pipeline"""
        try:
            logger.info("Loading inference pipeline model...")
            model_path = Path("models/inference_pipeline.pkl")
            if not model_path.exists():
                logger.error(f"Model file not found at {model_path}")
                st.error("Model file not found. Please run the training pipeline first.")
                return

            with open(model_path, 'rb') as f:
                components = pickle.load(f)

            self.model = components['model']
            self.vectorizer = components['vectorizer']
            self.label_encoder = components['label_encoder']

            logger.info("Model loaded successfully")
            logger.info(f"Model type: {type(self.model).__name__}")
            logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
            st.success("Model loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            st.error(f"Error loading model: {str(e)}")

    def load_sample_texts(self):
        """Load sample texts from the dataset with ground truth"""
        try:
            data_path = Path("data/raw/osdg-community-data-v2024-04-01.csv")
            if not data_path.exists():
                st.warning("Dataset not found. Using default sample texts.")
                self.sample_texts = [
                    ("Renewable energy solutions for sustainable development", 7),
                    ("Gender equality and women's empowerment in rural communities", 5),
                    ("Quality education for all children worldwide", 4),
                    ("Clean water and sanitation access in developing countries", 6),
                    ("Climate change mitigation through carbon reduction", 13),
                    ("Poverty reduction through economic development programs", 1)
                ]
                return

            # Load dataset and sample diverse texts
            df = pd.read_csv(data_path, sep='\t')

            # Sample texts from different SDGs (up to 2 per SDG) with ground truth
            sampled_texts = []
            for sdg in range(1, 17):
                sdg_rows = df[df['sdg'] == sdg].head(2)
                for _, row in sdg_rows.iterrows():
                    # Truncate text if too long for display
                    text = row['text']
                    if len(text) > 200:
                        text = text[:200] + "..."
                    sampled_texts.append((text, int(row['sdg'])))

            # If we don't have enough samples, fill with random ones
            if len(sampled_texts) < 16:
                remaining_rows = df.sample(min(16 - len(sampled_texts), len(df)))
                for _, row in remaining_rows.iterrows():
                    text = row['text']
                    if len(text) > 200:
                        text = text[:200] + "..."
                    sampled_texts.append((text, int(row['sdg'])))

            self.sample_texts = sampled_texts[:16]  # Limit to 16 samples

        except Exception as e:
            st.warning(f"Error loading sample texts: {str(e)}. Using defaults.")
            self.sample_texts = [
                ("Renewable energy solutions for sustainable development", 7),
                ("Gender equality and women's empowerment in rural communities", 5),
                ("Quality education for all children worldwide", 4),
                ("Clean water and sanitation access in developing countries", 6)
            ]

    def predict_sdg(self, text):
        """Predict SDG for given text"""
        if not self.model or not self.vectorizer or not self.label_encoder:
            logger.warning("Model components not loaded")
            return None

        try:
            logger.info(f"Making prediction for text of length {len(text)}")
            
            # Vectorize text
            X_vec = self.vectorizer.transform([text])
            logger.debug(f"Text vectorized, shape: {X_vec.shape}")

            # Predict
            prediction = self.model.predict(X_vec)[0]
            predicted_sdg = int(self.label_encoder.inverse_transform([prediction])[0])
            logger.info(f"Predicted SDG: {predicted_sdg}")

            # Get probabilities
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_vec)[0]
                logger.debug(f"Prediction confidence: {probabilities.max():.4f}")

            result = {
                'predicted_sdg': predicted_sdg,
                'sdg_name': SDG_NAMES.get(predicted_sdg, f"SDG {predicted_sdg}"),
                'confidence': float(probabilities.max()) if probabilities is not None else None
            }

            if probabilities is not None:
                # Get top 3 predictions
                top_indices = probabilities.argsort()[-3:][::-1]
                result['top_predictions'] = [
                    {
                        'sdg': int(self.label_encoder.classes_[idx]),
                        'sdg_name': SDG_NAMES.get(int(self.label_encoder.classes_[idx]), f"SDG {int(self.label_encoder.classes_[idx])}"),
                        'probability': float(probabilities[idx])
                    }
                    for idx in top_indices
                ]
                logger.debug(f"Top 3 predictions: {[p['sdg'] for p in result['top_predictions']]}")

            return result

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}", exc_info=True)
            st.error(f"Error making prediction: {str(e)}")
            return None

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="SDG Classifier",
        layout="wide"
    )

    st.title("Sustainable Development Goals (SDG) Classifier")
    st.markdown("Classify research papers and documents into UN Sustainable Development Goals")

    # Initialize app
    app = SDGClassifierApp()

    # Create main content area
    st.subheader("Input Text")

    # Text input options
    input_option = st.radio(
        "Choose input method:",
        ["Enter your own text", "Select from predefined samples"],
        horizontal=True
    )

    if input_option == "Enter your own text":
        user_text = st.text_area(
            "Enter research paper text or description:",
            height=150,
            placeholder="Paste your text here..."
        )
        text_to_classify = user_text
        ground_truth_sdg = None
    else:
        if app.sample_texts:
            # Create options with SDG info
            sample_options = [f"SDG {sdg} - {SDG_NAMES.get(sdg, f'SDG {sdg}')}: {text[:100]}..." if len(text) > 100 else f"SDG {sdg} - {SDG_NAMES.get(sdg, f'SDG {sdg}')}: {text}"
                            for text, sdg in app.sample_texts]

            selected_option = st.selectbox(
                "Select a sample text:",
                sample_options,
                help="Samples from the dataset with their ground truth SDG labels"
            )

            # Find the corresponding text and SDG
            selected_index = sample_options.index(selected_option)
            selected_text, ground_truth_sdg = app.sample_texts[selected_index]
            text_to_classify = selected_text

            # Show selected text with ground truth
            st.markdown(f"**Ground Truth SDG:** {ground_truth_sdg} - {SDG_NAMES.get(ground_truth_sdg, f'SDG {ground_truth_sdg}')}")
            st.text_area("Selected text:", selected_text, height=100, disabled=True)
        else:
            st.error("No sample texts available")
            text_to_classify = ""
            ground_truth_sdg = None

    # Classify button
    if st.button("Classify Text", type="primary", use_container_width=True):
        if text_to_classify.strip():
            with st.spinner("Analyzing text..."):
                result = app.predict_sdg(text_to_classify)

                if result:
                    st.success("Classification Complete!")

                    # Display ground truth if using predefined sample
                    if ground_truth_sdg is not None:
                        st.info(f"**Ground Truth:** SDG {ground_truth_sdg} - {SDG_NAMES.get(ground_truth_sdg, f'SDG {ground_truth_sdg}')}")

                    # Display main prediction
                    st.subheader("Primary Prediction")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("SDG Number", result['predicted_sdg'])
                    with col_b:
                        st.metric("SDG Name", result['sdg_name'])
                    with col_c:
                        if result['confidence']:
                            st.metric("Confidence", ".1%")

                    # Display top predictions if available
                    if 'top_predictions' in result and result['top_predictions']:
                        st.subheader("Top 3 Predictions")
                        for i, pred in enumerate(result['top_predictions'], 1):
                            with st.container():
                                col_rank, col_sdg, col_name, col_conf = st.columns([1, 1, 3, 2])
                                with col_rank:
                                    st.write(f"#{i}")
                                with col_sdg:
                                    st.write(f"SDG {pred['sdg']}")
                                with col_name:
                                    st.write(pred['sdg_name'])
                                with col_conf:
                                    st.write(".1%")
                                st.progress(pred['probability'])
                else:
                    st.error("Failed to classify text")
        else:
            st.warning("Please enter some text to classify")

if __name__ == "__main__":
    main()
