# Quick Reference: Logging and Artifacts

## For Developers

### View Logs Locally
```bash
# View latest logs
ls -lht logs/ | head

# Follow training logs in real-time
tail -f logs/train_*.log

# Check for errors
grep -r "ERROR" logs/

# View error-only logs
cat logs/*_errors_*.log

# Parse JSON logs
jq '.' logs/preprocess_*.jsonl | less
```

### Collect Artifacts
```bash
# Run the collection script
./scripts/collect_artifacts.sh

# Output will be in artifacts_YYYYMMDD_HHMMSS/
```

### Log Levels in Code
```python
from src.utils.logging_config import setup_logging

# Initialize logger
logger = setup_logging(log_dir="logs", module_name="my_module", log_level="INFO")

# Use different levels
logger.debug("Detailed info for debugging")
logger.info("General progress information")
logger.warning("Important warning")
logger.error("Error occurred", exc_info=True)  # Include stack trace
logger.critical("Critical failure")
```

## For CI/CD

### Accessing Artifacts

#### Via GitHub UI
1. Go to: `https://github.com/YOUR_USERNAME/SDGClassifier/actions`
2. Click on a workflow run
3. Scroll to "Artifacts" section
4. Click to download (ZIP file)

#### Via GitHub CLI
```bash
# List recent runs
gh run list --repo YOUR_USERNAME/SDGClassifier

# Download all artifacts from a run
gh run download RUN_ID --repo YOUR_USERNAME/SDGClassifier

# Download specific artifact
gh run download RUN_ID --name logs-COMMIT_SHA --repo YOUR_USERNAME/SDGClassifier
```

#### Via curl/wget
```bash
# Get artifact list (requires GitHub token)
curl -H "Authorization: token YOUR_GITHUB_TOKEN" \
  https://api.github.com/repos/YOUR_USERNAME/SDGClassifier/actions/artifacts

# Download artifact (get URL from above)
curl -L -H "Authorization: token YOUR_GITHUB_TOKEN" \
  -o artifact.zip \
  ARTIFACT_DOWNLOAD_URL
```

### Artifacts Included

| Artifact Name | Contents | When Collected | Retention |
|--------------|----------|----------------|-----------|
| `logs-{sha}` | All application logs, error logs, JSON logs | Always | 30 days |
| `metrics-{sha}` | Training/eval metrics, model metadata | Always | 30 days |
| `models-{sha}` | Trained models (.pkl), model registry | On success | 30 days |
| `deployment-logs-{sha}` | Docker & K8s deployment logs | Deploy job | 30 days |

### Workflow Jobs

1. **test-and-train** (Always runs)
   - Runs preprocessing
   - Trains model
   - Evaluates model
   - Runs tests
   - Uploads logs, metrics, models

2. **deploy** (Only on main/master push)
   - Downloads trained models
   - Builds Docker image
   - Pushes to Azure ACR
   - Deploys to AKS
   - Uploads deployment logs

## Log File Reference

### Files Generated

```
logs/
├── preprocess_YYYYMMDD.log          # Full preprocessing logs
├── preprocess_errors_YYYYMMDD.log   # Errors only
├── preprocess_YYYYMMDD.jsonl        # Structured logs
├── train_YYYYMMDD.log               # Training logs
├── train_errors_YYYYMMDD.log        # Training errors
├── train_YYYYMMDD.jsonl             # Structured training logs
├── evaluate_YYYYMMDD.log            # Evaluation logs
├── evaluate_errors_YYYYMMDD.log     # Evaluation errors
├── evaluate_YYYYMMDD.jsonl          # Structured eval logs
├── export_YYYYMMDD.log              # Export logs
└── ui_app_YYYYMMDD.log              # UI application logs
```

### Log Rotation
- Max size: 10 MB per file
- Backups: 5 files kept
- Format: `{module}_{date}.log.1`, `.log.2`, etc.

## Common Tasks

### Debugging Failed Pipeline

1. **Download logs artifact:**
   ```bash
   gh run list --limit 5
   gh run download FAILED_RUN_ID --name logs-COMMIT_SHA
   ```

2. **Check error logs first:**
   ```bash
   cat logs/*_errors_*.log
   ```

3. **Review full context:**
   ```bash
   # Find time of error in error log
   # Then search main log for context
   grep -C 10 "ERROR_MESSAGE" logs/module_*.log
   ```

### Comparing Model Runs

1. **Download metrics from multiple runs:**
   ```bash
   gh run download RUN_ID_1 --name metrics-SHA1 -D run1_metrics
   gh run download RUN_ID_2 --name metrics-SHA2 -D run2_metrics
   ```

2. **Compare metrics:**
   ```bash
   jq '.validation_accuracy' run1_metrics/training_metrics*.json
   jq '.validation_accuracy' run2_metrics/training_metrics*.json
   ```

### Analyzing Performance

```bash
# Training time
jq '.training_time_seconds' metrics/training_metrics*.json

# Model accuracy
jq '.validation_accuracy' metrics/training_metrics*.json

# Cross-validation scores
jq '.cv_scores' metrics/training_metrics*.json

# Feature importance (if available)
jq '.feature_importance[:10]' metrics/training_metrics*.json
```

### Searching JSON Logs

```bash
# All errors
jq 'select(.level=="ERROR")' logs/*.jsonl

# Specific module errors
jq 'select(.level=="ERROR" and .module=="train")' logs/*.jsonl

# Time range
jq 'select(.timestamp >= "2025-11-05T00:00:00")' logs/*.jsonl

# Specific message pattern
jq 'select(.message | contains("accuracy"))' logs/*.jsonl
```

## Troubleshooting

### Problem: No logs generated
**Solution:**
```bash
# Check if logs directory exists
ls -ld logs/

# Create if missing
mkdir -p logs

# Check write permissions
touch logs/test.log && rm logs/test.log
```

### Problem: Artifacts not uploaded
**Solution:**
1. Check workflow file syntax
2. Verify paths exist: `ls -la logs/ metrics/ models/`
3. Review GitHub Actions workflow logs
4. Check artifact upload step for errors

### Problem: Import errors
**Solution:**
Import errors in the IDE are expected and will resolve at runtime:
```python
# These work at runtime even if IDE shows error
from src.utils.logging_config import setup_logging
```

### Problem: Large log files
**Solution:**
Logs auto-rotate at 10MB. To reduce size:
```python
# Change log level to INFO instead of DEBUG
logger = setup_logging(log_level="INFO")  # Less verbose
```

## Integration Examples

### Add Logging to New Module

```python
"""My new module for SDG Classification"""

import sys
from pathlib import Path

try:
    from ..utils.logging_config import setup_logging
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.utils.logging_config import setup_logging

# Initialize logger for this module
logger = setup_logging(
    log_dir="logs",
    module_name="my_module",
    log_level="INFO"
)

def my_function(data):
    """Process data with logging"""
    logger.info(f"Processing {len(data)} items")
    
    try:
        # Your code here
        result = process(data)
        logger.info("Processing completed successfully")
        return result
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise
```

### Log Function Decorator

```python
from src.utils.logging_config import log_function_call

@log_function_call(logger)
def expensive_operation(param1, param2):
    """This function's calls will be automatically logged"""
    # Function automatically logs entry and exit
    return result
```

## Best Practices

### DO:
✅ Use appropriate log levels (DEBUG for details, INFO for progress)  
✅ Include context in messages (sizes, counts, types)  
✅ Log before and after important operations  
✅ Use `exc_info=True` for exceptions  
✅ Keep messages concise but informative  

### DON'T:
❌ Log sensitive data (passwords, tokens, PII)  
❌ Log in tight loops (use counters instead)  
❌ Use print() instead of logging  
❌ Log huge data structures  
❌ Ignore log rotation settings  

## Resources

- 📚 Full Documentation: `docs/LOGGING.md`
- 📋 Update Summary: `LOGGING_UPDATES.md`
- 🔧 Collection Script: `scripts/collect_artifacts.sh`
- ⚙️ Logging Config: `src/utils/logging_config.py`
- 🚀 Workflow File: `.github/workflows/ci-cd.yml`

## Support

Questions? Check:
1. `docs/LOGGING.md` - Comprehensive documentation
2. Error logs: `logs/*_errors_*.log`
3. GitHub Actions artifacts
4. This quick reference guide

---

**Quick Start:** Just push to main/master and check the Actions tab for artifacts! 🚀
