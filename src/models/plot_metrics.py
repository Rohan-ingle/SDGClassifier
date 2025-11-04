"""
Generate a poster-style metrics chart from metrics JSON files.

Reads metrics/model_comparison.json or any metrics/*_metrics.json files and
produces `reports/result_metrics.png` (high-resolution) suitable for the poster.
"""

import os
import json
import glob
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
METRICS_DIR = "metrics"
REPORTS_DIR = "reports"
OUT_HTML = os.path.join(REPORTS_DIR, "result_metrics_plotly.html")


def load_metrics() -> List[Dict[str, Any]]:
    """
    Load metrics from various JSON files in the metrics directory.

    Attempts to load metrics in the following order:
    1. model_comparison.json (aggregated metrics)
    2. Individual *_metrics.json files
    3. evaluation_results.json (fallback)

    Returns:
        List of dictionaries containing metrics for each model
    """
    logger.info("Loading metrics from %s directory", METRICS_DIR)
    all_metrics = []

    # Prefer aggregated file
    agg_path = os.path.join(METRICS_DIR, "model_comparison.json")

    if os.path.exists(agg_path):
        logger.info("Found aggregated metrics file: %s", agg_path)
        try:
            with open(agg_path, "r") as f:
                all_metrics = json.load(f)
            logger.info("Loaded %d model metrics from aggregated file", len(all_metrics))
        except json.JSONDecodeError as e:
            logger.error("Failed to parse aggregated metrics file: %s", str(e))
        except Exception as e:
            logger.error("Error loading aggregated metrics: %s", str(e))
    else:
        logger.info("Aggregated metrics file not found, searching for individual metrics files")
        files = glob.glob(os.path.join(METRICS_DIR, "*_metrics.json"))
        logger.info("Found %d individual metrics files", len(files))

        for fp in files:
            logger.debug("Processing metrics file: %s", fp)
            try:
                with open(fp, "r") as f:
                    m = json.load(f)
                    if "algorithm" not in m:
                        base = os.path.basename(fp)
                        alg = base.replace("_metrics.json", "")
                        m["algorithm"] = alg
                        logger.debug("Inferred algorithm name: %s", alg)
                    all_metrics.append(m)
                    logger.debug("Successfully loaded metrics from %s", fp)
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse JSON in %s: %s", fp, str(e))
                continue
            except Exception as e:
                logger.warning("Error loading metrics from %s: %s", fp, str(e))
                continue

        logger.info("Loaded %d metrics from individual files", len(all_metrics))

    # Fallback to evaluation results if no metrics found
    if not all_metrics:
        logger.warning("No metrics found in standard files, checking evaluation_results.json")
        eval_path = os.path.join(METRICS_DIR, "evaluation_results.json")

        if os.path.exists(eval_path):
            logger.info("Loading fallback metrics from %s", eval_path)
            try:
                with open(eval_path, "r") as f:
                    evalm = json.load(f)
                    m = {
                        "algorithm": evalm.get("model_type", "model"),
                        "test_accuracy": evalm.get("test_accuracy")
                        or evalm.get("accuracy")
                        or np.nan,
                        "precision_macro": evalm.get("precision_macro") or np.nan,
                        "recall_macro": evalm.get("recall_macro") or np.nan,
                    }
                    all_metrics.append(m)
                    logger.info("Loaded fallback metrics successfully")
            except Exception as e:
                logger.error("Failed to load evaluation results: %s", str(e))
        else:
            logger.error("No metrics files found in %s directory", METRICS_DIR)

    logger.info("Total metrics loaded: %d", len(all_metrics))
    return all_metrics


def build_dataframe(metrics_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a pandas DataFrame from metrics list for easier visualization.

    Args:
        metrics_list: List of dictionaries containing model metrics

    Returns:
        DataFrame with columns: Algorithm, Accuracy, Precision, Recall
    """
    logger.info("Building DataFrame from %d metrics entries", len(metrics_list))

    if not metrics_list:
        logger.warning("Empty metrics list provided")
        return pd.DataFrame(columns=["Algorithm", "Accuracy", "Precision", "Recall"])

    rows = []
    for i, m in enumerate(metrics_list):
        algorithm_name = m.get("algorithm", "Unknown")
        formatted_name = algorithm_name.replace("_", " ").title()

        row = {
            "Algorithm": formatted_name,
            "Accuracy": m.get("test_accuracy", np.nan),
            "Precision": m.get("precision_macro", np.nan),
            "Recall": m.get("recall_macro", np.nan),
        }

        logger.debug(
            "Row %d - %s: Acc=%.4f, Prec=%.4f, Rec=%.4f",
            i,
            formatted_name,
            row["Accuracy"] if not np.isnan(row["Accuracy"]) else 0,
            row["Precision"] if not np.isnan(row["Precision"]) else 0,
            row["Recall"] if not np.isnan(row["Recall"]) else 0,
        )

        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info("DataFrame created with shape %s", df.shape)
    logger.info("Columns: %s", df.columns.tolist())

    # Log any missing values
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        logger.warning("Missing values detected:\n%s", missing_counts[missing_counts > 0])

    return df


def plot_metrics_plotly(df: pd.DataFrame, out_html: str) -> None:
    """
    Create an interactive Plotly bar chart visualization of model metrics.

    Args:
        df: DataFrame containing Algorithm, Accuracy, Precision, and Recall columns
        out_html: Output path for the HTML file
    """
    logger.info("Creating interactive Plotly visualization")
    logger.info("Plotting metrics for %d algorithms", len(df))

    try:
        # Define color scheme
        colors = {
            "Accuracy": "#ffd700",  # Gold
            "Precision": "#6a0dad",  # Purple
            "Recall": "#9b5de5",  # Light Purple
        }
        logger.debug("Using color scheme: %s", colors)

        # Create figure
        fig = go.Figure()
        x = df["Algorithm"].tolist()

        # Add traces for each metric
        for metric in ["Accuracy", "Precision", "Recall"]:
            y = df[metric].tolist()

            # Format text labels
            text_labels = [f"{v:.2f}" if v is not None and not np.isnan(v) else "" for v in y]

            logger.debug("Adding trace for %s metric", metric)
            fig.add_trace(
                go.Bar(
                    x=x,
                    y=y,
                    name=metric,
                    marker_color=colors[metric],
                    text=text_labels,
                    textposition="outside",
                )
            )

        # Update layout
        logger.debug("Configuring chart layout")
        fig.update_layout(
            title={
                "text": "Model Performance Metrics Comparison",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20, "family": "Arial Black"},
            },
            barmode="group",
            template="plotly_white",
            xaxis=dict(title="Algorithm", tickangle=-45),
            yaxis=dict(range=[0, 1], title="Score"),
            legend=dict(
                title="Metrics", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            paper_bgcolor="#E2DCDE",
            plot_bgcolor="#E2DCDE",
            hovermode="x unified",
        )

        # Ensure output directory exists
        output_dir = os.path.dirname(out_html)
        if output_dir:
            logger.debug("Creating output directory: %s", output_dir)
            os.makedirs(output_dir, exist_ok=True)

        # Save figure
        logger.info("Saving interactive chart to %s", out_html)
        fig.write_html(out_html)

        logger.info("Interactive chart saved successfully")
        logger.info("File size: %d bytes", os.path.getsize(out_html))

    except Exception as e:
        logger.error("Failed to create Plotly visualization: %s", str(e))
        logger.exception("Full traceback:")
        raise


def main() -> None:
    """
    Main function to orchestrate the metrics visualization pipeline.

    Workflow:
    1. Load metrics from JSON files
    2. Build DataFrame for visualization
    3. Create interactive Plotly chart
    """
    logger.info("=" * 60)
    logger.info("Starting Metrics Visualization Pipeline")
    logger.info("=" * 60)

    try:
        # Load metrics
        logger.info("Step 1: Loading metrics...")
        metrics_list = load_metrics()

        if not metrics_list:
            logger.error("No metrics found under %s/", METRICS_DIR)
            logger.error("Please run training or produce metrics first")
            print("No metrics found under metrics/; run training or produce metrics first.")
            return

        logger.info("Successfully loaded metrics for %d models", len(metrics_list))

        # Build DataFrame
        logger.info("Step 2: Building DataFrame...")
        df = build_dataframe(metrics_list)

        if df.empty:
            logger.error("DataFrame is empty - no usable metrics present")
            print("No usable metrics present")
            return

        logger.info("DataFrame built successfully")

        # Display summary statistics
        logger.info("Summary Statistics:")
        for metric in ["Accuracy", "Precision", "Recall"]:
            if metric in df.columns:
                valid_values = df[metric].dropna()
                if len(valid_values) > 0:
                    logger.info(
                        "  %s - Mean: %.4f, Std: %.4f, Min: %.4f, Max: %.4f",
                        metric,
                        valid_values.mean(),
                        valid_values.std(),
                        valid_values.min(),
                        valid_values.max(),
                    )

        # Create visualization
        logger.info("Step 3: Creating visualization...")
        plot_metrics_plotly(df, OUT_HTML)

        logger.info("=" * 60)
        logger.info("Metrics Visualization Pipeline Completed Successfully")
        logger.info("Output saved to: %s", OUT_HTML)
        logger.info("=" * 60)

        print(f"\nInteractive chart saved to {OUT_HTML}")
        print(f"Open this file in a web browser to view the visualization.")

    except Exception as e:
        logger.error("=" * 60)
        logger.error("Pipeline failed with error: %s", str(e))
        logger.exception("Full traceback:")
        logger.error("=" * 60)
        raise


if __name__ == "__main__":
    main()
