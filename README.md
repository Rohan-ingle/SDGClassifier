# SDG Classification Model - MLOps Pipeline

[![CI/CD Pipeline](https://github.com/Rohan-ingle/SDGClassifier/workflows/MLOps%20CI/CD%20Pipeline/badge.svg)](https://github.com/Rohan-ingle/SDGClassifier/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![DVC](https://img.shields.io/badge/DVC-2.0+-orange.svg)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/kubernetes-AKS-326CE5.svg)](https://azure.microsoft.com/en-us/services/kubernetes-service/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **🚀 Now with Azure Kubernetes Service (AKS) deployment!** See [AKS_SETUP_SUMMARY.md](AKS_SETUP_SUMMARY.md) for quick setup.

A comprehensive MLOps pipeline for classifying research papers according to Sustainable Development Goals (SDGs) using the OSDG (Open Source for Sustainable Development Goals) dataset.

## Project Overview

This project implements a machine learning model to classify research papers into one of the 16 SDG categories. The pipeline includes:

- **Data Processing**: Automated text preprocessing and feature extraction
- **Model Training**: Support for multiple algorithms (Random Forest, SVM, Logistic Regression, Neural Networks)
- **Model Evaluation**: Comprehensive performance metrics and visualizations
- **MLOps Integration**: DVC for pipeline orchestration and experiment tracking
- **CI/CD Pipeline**: Automated testing, training, and deployment using GitHub Actions
- **Model Deployment**: Ready-to-deploy inference pipeline
- **Cloud Deployment**: Azure Kubernetes Service (AKS) deployment with Docker
- **Auto-scaling**: Horizontal Pod Autoscaler for dynamic scaling (2-10 pods)
- **Containerization**: Docker support for consistent environment across all platforms

## Dataset

The project uses the [OSDG Community Dataset v2024-04-01](https://zenodo.org/records/11441197) which contains:
- **43,025 research paper abstracts**
- **16 SDG categories** (SDG 1-16)
- **Multi-label classification** capability
- **Quality annotations** with agreement scores

### SDG Distribution in Dataset:
```
SDG 16 (Peace, Justice): 5,451 samples
SDG 5 (Gender Equality): 4,338 samples  
SDG 4 (Quality Education): 3,740 samples
SDG 7 (Clean Energy): 3,048 samples
SDG 6 (Clean Water): 2,815 samples
... and more
```

## Project Structure

```
SDGClassifier/
├── data/
│   ├── raw/                    # Raw dataset
│   └── processed/              # Processed features and labels
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocess.py       # Data preprocessing pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py           # Model training
│   │   └── export.py          # Model export for deployment
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluate.py        # Model evaluation
│       └── validate.py        # Model validation
├── models/                     # Trained models
├── metrics/                    # Evaluation metrics and plots
├── notebooks/                  # Jupyter notebooks for analysis
├── tests/                      # Unit tests
├── .github/workflows/          # CI/CD pipeline
├── params.yaml                 # Configuration parameters
├── dvc.yaml                    # DVC pipeline definition
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Quick Start

### Prerequisites

- Python 3.9+
- Git
- DVC
- Conda/pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/username/SDGClassifier.git
   cd SDGClassifier
   ```

2. **Create and activate virtual environment**
   ```bash
   conda create -n sdg-classifier python=3.9
   conda activate sdg-classifier
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize DVC (if not already done)**
   ```bash
   dvc init
   ```

### Running the Pipeline

#### Option 1: Run Complete Pipeline
```bash
# Run the entire ML pipeline
dvc repro

# View pipeline status
dvc dag
```

#### Option 2: Run Individual Stages
```bash
# Data preprocessing
dvc repro data_preprocessing

# Model training
dvc repro train_model

# Model evaluation
dvc repro evaluate_model

# Model validation
dvc repro model_validation

# Export final model
dvc repro export_model
```

#### Option 3: Run Scripts Directly
```bash
# Preprocess data
python src/data/preprocess.py

# Train model
python src/models/train.py

# Evaluate model
python src/evaluation/evaluate.py
```

## Configuration

All pipeline parameters are defined in `params.yaml`:

### Key Parameters:

- **Data Processing**
  - `test_size`: 0.2 (20% for testing)
  - `val_size`: 0.1 (10% for validation)
  - `random_state`: 42

- **Text Preprocessing**
  - `max_features`: 10,000 (TF-IDF vocabulary size)
  - `ngram_range`: [1, 2] (unigrams and bigrams)
  - `min_df`: 5 (minimum document frequency)

- **Model Training**
  - `algorithm`: "random_forest" (options: random_forest, svm, logistic_regression, neural_network)
  - `n_estimators`: 100
  - `class_weight`: "balanced"

- **Evaluation**
  - `cv_folds`: 5 (cross-validation folds)
  - `scoring_metrics`: ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

## Model Performance

The pipeline generates comprehensive evaluation metrics:

### Metrics Tracked:
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Macro and weighted F1 scores
- **Precision/Recall**: Per-class and averaged metrics
- **Confusion Matrix**: Visual representation of predictions
- **Feature Importance**: Top contributing features
- **Cross-Validation**: Robust performance estimation

### Output Files:
- `metrics/evaluation_results.json`: Detailed metrics
- `metrics/confusion_matrix.png`: Confusion matrix plot
- `metrics/feature_importance.png`: Feature importance plot
- `metrics/classification_report.txt`: Detailed classification report

## MLOps Workflow

### DVC Pipeline Stages:

1. **data_preprocessing**: Load, clean, and split data
2. **train_model**: Train ML model with specified algorithm
3. **evaluate_model**: Comprehensive model evaluation
4. **model_validation**: Validation set performance
5. **export_model**: Package model for deployment

### CI/CD Pipeline:

The GitHub Actions workflow includes:

1. **Code Quality Checks**
   - Linting with flake8
   - Code formatting with black
   - Import sorting with isort

2. **Testing**
   - Unit tests with pytest
   - Code coverage reporting

3. **Data Validation**
   - Schema validation
   - Data quality checks

4. **Model Training & Evaluation**
   - Automated model training
   - Performance validation
   - Artifact storage

5. **Security Scanning**
   - Dependency vulnerability checks
   - Code security analysis

6. **Deployment**
   - Model packaging
   - Deployment artifact creation

## Model Artifact Storage (Azure Blob Storage)

The project stores large data and model artifacts in Azure Blob Storage through a DVC remote named `azuremodelstore` (default `azure://sdgclassifier-artifacts/models`, configured in `.dvc/config`). Update the `url` value if your container and path differ.

- Configure local development with:
   ```bash
   export AZURE_STORAGE_CONNECTION_STRING="<your-connection-string>"
   # or
   export AZURE_STORAGE_ACCOUNT="<your-account-name>"
   export AZURE_STORAGE_KEY="<your-account-key>"
   ```
- Push updated artifacts: `dvc push`
- Pull artifacts for reproducibility: `dvc pull`
- GitHub Actions automatically pushes/pulls when the storage secrets are present.

## Model Deployment

### Cloud Deployment (Azure Kubernetes Service)

This project includes full Azure Kubernetes Service (AKS) deployment setup with Docker containerization.

#### Quick Deploy to AKS:
```bash
# Interactive deployment menu
./deploy-aks.sh

# Or automated deployment
./deploy-aks.sh --full
```

#### Features:
- ✅ **Docker Containerization**: Production-ready Dockerfile
- ✅ **Kubernetes Manifests**: Complete k8s configuration in `k8s/` directory
- ✅ **Auto-scaling**: HPA configuration (2-10 pods based on load)
- ✅ **CI/CD Pipeline**: GitHub Actions workflow for automated deployment
- ✅ **Load Balancer**: External access via Azure Load Balancer
- ✅ **Health Checks**: Liveness and readiness probes
- ✅ **Resource Management**: CPU/Memory limits and requests
- ✅ **Monitoring Ready**: Integrated with Azure Monitor

#### Deployment Documentation:
- **[AZURE_SETUP.md](AZURE_SETUP.md)**: Complete Azure setup guide
- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Comprehensive deployment instructions
- **[k8s/README.md](k8s/README.md)**: Kubernetes configuration details

#### Manual Deployment Steps:
```bash
# 1. Build Docker image
docker build -t sdgclassifier:latest .

# 2. Push to Azure Container Registry
az acr login --name <your-acr-name>
docker tag sdgclassifier:latest <acr-name>.azurecr.io/sdg-classifier:latest
docker push <acr-name>.azurecr.io/sdg-classifier:latest

# 3. Deploy to AKS
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/hpa.yaml

# 4. Get external IP
kubectl get service sdg-classifier-service -n sdg-classifier
```

### Local Inference Pipeline

The trained model can be deployed locally using the exported inference pipeline:

```python
import pickle

# Load inference pipeline
with open('models/inference_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Make predictions
text = "This research focuses on renewable energy solutions for sustainable development."
result = pipeline.predict(text)

print(f"Predicted SDG: {result['predicted_sdg']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Top predictions: {result['top_predictions']}")
```

### Streamlit Web Application

Run the interactive web interface locally:

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

### Docker Deployment

Run the application in a Docker container:

```bash
# Build the image
docker build -t sdg-classifier:latest .

# Run the container
docker run -p 8501:8501 sdg-classifier:latest

# Access at http://localhost:8501
```

## GitHub Actions CI/CD

### Workflows

The project includes two GitHub Actions workflows:

1. **Basic CI** (`.github/workflows/basic-ci.yml`)
   - Runs on every push/PR
   - Code quality checks
   - Basic syntax validation
   - Project structure verification

2. **AKS Deployment** (`.github/workflows/aks-deployment.yml`)
   - Full CI/CD pipeline
   - Automated testing with coverage
   - Docker image building and pushing to ACR
   - Automated deployment to AKS
   - Health checks and monitoring

### Required GitHub Secrets

Configure these in your repository settings:

| Secret Name | Description |
|------------|-------------|
| `AZURE_CREDENTIALS` | Azure service principal JSON |
| `ACR_NAME` | Azure Container Registry name |
| `AKS_CLUSTER_NAME` | AKS cluster name |
| `AKS_RESOURCE_GROUP` | Azure resource group |
| `AZURE_STORAGE_CONNECTION_STRING` | Connection string for the Azure storage account that backs DVC |
| `AZURE_STORAGE_ACCOUNT` (optional) | Storage account name (only needed if you prefer account/key auth) |
| `AZURE_STORAGE_KEY` (optional) | Storage account key (only needed if you prefer account/key auth) |

See [AZURE_SETUP.md](AZURE_SETUP.md) for detailed setup instructions.

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocess.py -v
```

## Notebooks

Jupyter notebooks for exploration and analysis:

- `notebooks/01_data_exploration.ipynb`: Dataset analysis
- `notebooks/02_model_comparison.ipynb`: Algorithm comparison
- `notebooks/03_error_analysis.ipynb`: Error analysis and insights

## Development

### Code Style

The project follows these standards:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **Type hints** for better code documentation

Format code:
```bash
black src --line-length=100
isort src
flake8 src
```

### Adding New Models

1. Implement model in `src/models/train.py`
2. Add algorithm to `params.yaml`
3. Update tests in `tests/`
4. Run pipeline to validate

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks
5. Submit a pull request

## Requirements

See `requirements.txt` for full dependency list. Key packages:

- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **nltk**: Natural language processing
- **matplotlib/seaborn**: Visualization
- **dvc**: Data version control
- **pyyaml**: Configuration management

## 📚 Complete Documentation

This project includes comprehensive documentation for all aspects of development and deployment:

### Quick Start & Deployment
- **[QUICKSTART.md](QUICKSTART.md)** - ⚡ Fast-start guide with multiple deployment options
- **[AKS_SETUP_SUMMARY.md](AKS_SETUP_SUMMARY.md)** - 🎉 Complete overview of AKS setup

### Cloud Deployment
- **[AZURE_SETUP.md](AZURE_SETUP.md)** - ☁️ Detailed Azure resource setup guide
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - 📖 Comprehensive deployment instructions
- **[GITHUB_SECRETS_SETUP.md](GITHUB_SECRETS_SETUP.md)** - 🔐 GitHub Actions secrets configuration

### Kubernetes
- **[k8s/README.md](k8s/README.md)** - 🎯 Kubernetes manifests and usage guide

### Scripts
- **`deploy-aks.sh`** - 🛠️ Automated deployment script with interactive menu

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run quality checks (`black`, `flake8`, `pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OSDG Community](https://zenodo.org/records/11441197) for the dataset
- [DVC](https://dvc.org/) for MLOps capabilities
- [scikit-learn](https://scikit-learn.org/) for ML algorithms
- [Devendhake18/MLOPS-Project](https://github.com/Devendhake18/MLOPS-Project) for AKS deployment inspiration
- [Azure](https://azure.microsoft.com/) for cloud infrastructure
- [Kubernetes](https://kubernetes.io/) for container orchestration

## 📞 Support & Contact

- **Issues**: [Open an issue](https://github.com/Rohan-ingle/SDGClassifier/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Rohan-ingle/SDGClassifier/discussions)
- **Documentation**: Check the comprehensive docs in this repository

---

**⭐ Star this repo if you find it useful!**

Made with ❤️ for Sustainable Development Goals

---

**Built with care for Sustainable Development Goals**