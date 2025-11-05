#!/bin/bash
# Script to collect and organize logs and metrics as artifacts

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARTIFACT_DIR="artifacts_${TIMESTAMP}"

echo "=========================================="
echo "Collecting artifacts for CI/CD pipeline"
echo "Timestamp: ${TIMESTAMP}"
echo "=========================================="

# Create artifact directory structure
mkdir -p "${ARTIFACT_DIR}/logs"
mkdir -p "${ARTIFACT_DIR}/metrics"
mkdir -p "${ARTIFACT_DIR}/models"
mkdir -p "${ARTIFACT_DIR}/reports"

# Collect logs
echo "Collecting logs..."
if [ -d "logs" ]; then
    cp -r logs/* "${ARTIFACT_DIR}/logs/" 2>/dev/null || echo "No logs found in logs/"
fi

# Collect metrics
echo "Collecting metrics..."
if [ -d "metrics" ]; then
    cp -r metrics/* "${ARTIFACT_DIR}/metrics/" 2>/dev/null || echo "No metrics found in metrics/"
fi

# Collect model metadata
echo "Collecting model metadata..."
if [ -f "models/training_metrics.json" ]; then
    cp models/training_metrics.json "${ARTIFACT_DIR}/models/"
fi
if [ -f "models/model_metadata.json" ]; then
    cp models/model_metadata.json "${ARTIFACT_DIR}/models/"
fi
if [ -f "models/model_registry.json" ]; then
    cp models/model_registry.json "${ARTIFACT_DIR}/models/"
fi

# Collect data statistics
echo "Collecting data statistics..."
if [ -f "data/processed/data_stats.json" ]; then
    cp data/processed/data_stats.json "${ARTIFACT_DIR}/reports/"
fi

# Collect evaluation results
echo "Collecting evaluation results..."
if [ -f "metrics/evaluation_results.json" ]; then
    cp metrics/evaluation_results.json "${ARTIFACT_DIR}/reports/"
fi
if [ -f "metrics/classification_report.txt" ]; then
    cp metrics/classification_report.txt "${ARTIFACT_DIR}/reports/"
fi

# Create summary report
echo "Creating summary report..."
cat > "${ARTIFACT_DIR}/ARTIFACT_SUMMARY.md" << EOF
# Artifact Collection Summary

**Generated:** ${TIMESTAMP}
**Commit:** ${GITHUB_SHA:-local}
**Branch:** ${GITHUB_REF:-local}
**Run ID:** ${GITHUB_RUN_ID:-N/A}

## Contents

### Logs
- Application logs from all modules
- Error logs
- JSON-formatted structured logs

### Metrics
- Training metrics
- Evaluation metrics
- Cross-validation results
- Inference metrics

### Models
- Model metadata
- Training parameters
- Model registry

### Reports
- Data statistics
- Classification reports
- Validation results

## Usage

These artifacts can be used for:
- Debugging pipeline issues
- Analyzing model performance
- Tracking experiment results
- Reproducing training runs

EOF

# List all collected files
echo "Listing collected artifacts..."
find "${ARTIFACT_DIR}" -type f > "${ARTIFACT_DIR}/file_list.txt"

# Calculate total size
TOTAL_SIZE=$(du -sh "${ARTIFACT_DIR}" | cut -f1)

echo ""
echo "=========================================="
echo "Artifact collection complete!"
echo "Location: ${ARTIFACT_DIR}"
echo "Total size: ${TOTAL_SIZE}"
echo "Files collected: $(wc -l < ${ARTIFACT_DIR}/file_list.txt)"
echo "=========================================="

# Display summary
cat "${ARTIFACT_DIR}/ARTIFACT_SUMMARY.md"
