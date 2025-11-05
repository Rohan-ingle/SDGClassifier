# Logging and Artifact Management

## Overview

This project implements comprehensive logging throughout the codebase with automatic artifact collection in the GitHub Actions CI/CD pipeline.

## Logging Architecture

### Centralized Logging Configuration

The project uses a centralized logging configuration module (`src/utils/logging_config.py`) that provides:

1. **Console Output** - Colored, formatted logs for easy reading during development
2. **File Logs** - Detailed rotating logs with full debugging information
3. **Error Logs** - Separate logs for errors and critical issues only
4. **JSON Logs** - Machine-readable structured logs for analysis

### Log Levels

- **DEBUG**: Detailed diagnostic information (vectorization details, function parameters)
- **INFO**: General progress information (training steps, data loading)
- **WARNING**: Important notices (missing files, deprecated features)
- **ERROR**: Error conditions that don't stop execution
- **CRITICAL**: Serious errors that may cause program termination

### Log Files Generated

All logs are stored in the `logs/` directory:

```
logs/
├── preprocess_YYYYMMDD.log          # Data preprocessing logs
├── preprocess_errors_YYYYMMDD.log   # Preprocessing errors only
├── preprocess_YYYYMMDD.jsonl        # Structured preprocessing logs
├── train_YYYYMMDD.log               # Training logs
├── train_errors_YYYYMMDD.log        # Training errors only
├── train_YYYYMMDD.jsonl             # Structured training logs
├── evaluate_YYYYMMDD.log            # Evaluation logs
├── evaluate_errors_YYYYMMDD.log     # Evaluation errors only
├── evaluate_YYYYMMDD.jsonl          # Structured evaluation logs
├── export_YYYYMMDD.log              # Model export logs
├── ui_app_YYYYMMDD.log              # UI application logs
└── ...
```

### Log Rotation

Logs automatically rotate when they reach 10MB in size, keeping up to 5 backup files per log type.

## Modules with Enhanced Logging

### 1. Data Preprocessing (`src/data/preprocess.py`)
Logs:
- Data loading and cleaning statistics
- Text preprocessing steps and token counts
- Feature engineering details
- Train/val/test split information
- Data statistics and class distributions

### 2. Feature Engineering (`src/data/feature_engineering.py`)
Logs:
- TF-IDF vectorizer configuration
- Vocabulary size and feature matrix density
- Feature transformation details

### 3. Model Training (`src/models/train.py`)
Logs:
- Model initialization and configuration
- Training progress and timing
- Training/validation accuracy per epoch (for applicable models)
- Cross-validation scores
- Model parameters and hyperparameters

### 4. Model Evaluation (`src/evaluation/evaluate.py`)
Logs:
- Test set loading
- Performance metrics calculation
- Classification report generation
- Confusion matrix details

### 5. Model Export (`src/models/export.py`)
Logs:
- Artifact loading process
- Pipeline creation steps
- Export format and destination

### 6. UI Application (`ui/app.py`)
Logs:
- Model loading status
- Prediction requests and results
- Error handling and debugging info

## GitHub Actions Artifact Collection

### Workflow Configuration

The CI/CD pipeline (`github/workflows/ci-cd.yml`) includes:

1. **test-and-train** job:
   - Runs preprocessing, training, and evaluation
   - Collects logs, metrics, and models as artifacts
   - Uploads artifacts with retention period of 30 days

2. **deploy** job:
   - Downloads trained models from artifacts
   - Deploys to Azure AKS
   - Uploads deployment logs

### Artifacts Collected

#### 1. Logs Artifact (`logs-{sha}`)
Contents:
- All application logs from each module
- Error logs
- JSON-formatted structured logs
- Timestamped for easy reference

Location in workflow: `logs/`

#### 2. Metrics Artifact (`metrics-{sha}`)
Contents:
- Training metrics (accuracy, loss, etc.)
- Evaluation metrics (precision, recall, F1)
- Cross-validation results
- Inference performance metrics
- Data statistics

Files included:
```
metrics/
├── training_metrics_*.json
├── evaluation_results.json
├── classification_report.txt
├── cv_metrics_*.json
├── inference_metrics_*.json
├── audit_report_*.json
└── ...
models/training_metrics.json
models/model_metadata.json
data/processed/data_stats.json
```

#### 3. Models Artifact (`models-{sha}`)
Contents:
- Trained model files (.pkl)
- Model metadata
- Model registry information

Files included:
```
models/
├── model.pkl
├── inference_pipeline.pkl
├── training_metrics.json
├── model_metadata.json
└── model_registry.json
```

#### 4. Deployment Logs Artifact (`deployment-logs-{sha}`)
Contents:
- Docker build logs
- Kubernetes deployment logs
- AKS rollout status

### Artifact Retention

- **Default Retention**: 30 days
- **Storage**: GitHub Actions artifact storage
- **Access**: Available via GitHub Actions UI or API

## Using the Artifact Collection Script

A dedicated script is available for local artifact collection:

```bash
./scripts/collect_artifacts.sh
```

This script:
1. Creates a timestamped artifact directory
2. Collects all logs, metrics, and reports
3. Generates a summary report
4. Lists all collected files

Output structure:
```
artifacts_YYYYMMDD_HHMMSS/
├── logs/              # All log files
├── metrics/           # Training and evaluation metrics
├── models/            # Model metadata
├── reports/           # Generated reports
├── ARTIFACT_SUMMARY.md
└── file_list.txt
```

## Accessing Artifacts in GitHub Actions

### Via GitHub UI

1. Navigate to the Actions tab in your repository
2. Select the workflow run
3. Scroll to the "Artifacts" section at the bottom
4. Download desired artifacts as ZIP files

### Via GitHub CLI

```bash
# List artifacts for a run
gh run list --repo OWNER/REPO

# Download artifacts
gh run download RUN_ID --repo OWNER/REPO

# Download specific artifact
gh run download RUN_ID --name logs-{sha} --repo OWNER/REPO
```

### Via GitHub API

```bash
# Get artifact list
curl -H "Authorization: token GITHUB_TOKEN" \
  https://api.github.com/repos/OWNER/REPO/actions/artifacts

# Download artifact
curl -L -H "Authorization: token GITHUB_TOKEN" \
  -o artifact.zip \
  ARTIFACT_DOWNLOAD_URL
```

## Log Analysis Tips

### 1. Finding Errors

```bash
# Search for errors in logs
grep -r "ERROR" logs/

# View only error logs
cat logs/*_errors_*.log

# Parse JSON logs for errors
jq 'select(.level=="ERROR")' logs/*.jsonl
```

### 2. Analyzing Performance

```bash
# Extract training times
grep "training_time" metrics/training_metrics*.json

# Compare model accuracies
jq '.validation_accuracy' models/training_metrics.json
```

### 3. Debugging Failed Runs

1. Download the logs artifact from failed run
2. Check error logs first: `*_errors_*.log`
3. Review main logs for context around errors
4. Check JSON logs for structured debugging info

## Best Practices

### For Developers

1. **Use appropriate log levels**
   - DEBUG for detailed diagnostic info
   - INFO for general progress
   - WARNING for important notices
   - ERROR for error conditions
   - CRITICAL for serious failures

2. **Include context in log messages**
   ```python
   logger.info(f"Processing {len(data)} samples with {n_features} features")
   ```

3. **Log before and after important operations**
   ```python
   logger.info("Starting model training...")
   model.fit(X_train, y_train)
   logger.info("Model training completed")
   ```

4. **Use exception logging**
   ```python
   try:
       process_data()
   except Exception as e:
       logger.error(f"Failed to process data: {str(e)}", exc_info=True)
       raise
   ```

### For CI/CD

1. **Always upload logs** even on failure (use `if: always()`)
2. **Use meaningful artifact names** with commit SHA
3. **Set appropriate retention periods** (30 days for important artifacts)
4. **Include summary files** in artifacts

## Monitoring and Alerting

Consider integrating with external monitoring services:

- **Elasticsearch + Kibana**: Ingest JSON logs for visualization
- **Datadog**: Monitor metrics and logs
- **Azure Monitor**: Track deployment and runtime metrics
- **Prometheus + Grafana**: Visualize training metrics

## Troubleshooting

### Logs Not Generated

1. Check if logging is properly initialized:
   ```python
   from src.utils.logging_config import setup_logging
   logger = setup_logging(log_dir="logs", module_name="my_module")
   ```

2. Verify logs directory exists and is writable

3. Check file handler permissions

### Artifacts Not Uploaded

1. Verify paths in workflow match actual file locations
2. Check if files are generated before upload step
3. Review workflow logs for upload errors
4. Ensure `if-no-files-found` is set appropriately

### Large Artifact Sizes

1. Use log rotation to limit file sizes
2. Compress artifacts before upload
3. Archive old artifacts periodically
4. Use selective artifact collection

## Future Enhancements

Planned improvements:

- [ ] Real-time log streaming during CI/CD runs
- [ ] Automated log analysis and error detection
- [ ] Integration with cloud-based log aggregation
- [ ] Performance metrics dashboard
- [ ] Automated alerting on errors or performance degradation
- [ ] Log compression for storage optimization

## References

- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)
- [GitHub Actions Artifacts](https://docs.github.com/en/actions/using-workflows/storing-workflow-data-as-artifacts)
- [Structured Logging Best Practices](https://www.structlog.org/en/stable/)
