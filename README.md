# SDG Classification Model - MLOps Pipeline

[![CI/CD Pipeline](https://github.com/username/SDGClassifier/workflows/MLOps%20CI/CD%20Pipeline/badge.svg)](https://github.com/username/SDGClassifier/actions)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![DVC](https://img.shields.io/badge/DVC-2.0+-orange.svg)](https://dvc.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive MLOps pipeline for classifying research papers according to Sustainable Development Goals (SDGs) using the OSDG (Open Source for Sustainable Development Goals) dataset.

## 🎯 Project Overview

This project implements a machine learning model to classify research papers into one of the 16 SDG categories. The pipeline includes:

- **Data Processing**: Automated text preprocessing and feature extraction
- **Model Training**: Support for multiple algorithms (Random Forest, SVM, Logistic Regression, Neural Networks)
- **Model Evaluation**: Comprehensive performance metrics and visualizations
- **MLOps Integration**: DVC for pipeline orchestration and experiment tracking
- **CI/CD Pipeline**: Automated testing, training, and deployment using GitHub Actions
- **Model Deployment**: Ready-to-deploy inference pipeline

## 📊 Dataset

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

## 🏗️ Project Structure

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

## 🚀 Quick Start

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

## ⚙️ Configuration

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

## 📈 Model Performance

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

## 🔄 MLOps Workflow

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

## 🚀 Model Deployment

### Inference Pipeline

The trained model can be deployed using the exported inference pipeline:

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

### API Deployment

A Flask API is included in the deployment package:

```bash
cd deployment/
python serve.py
```

Then make predictions via HTTP:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Research on clean water and sanitation systems"}'
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocess.py -v
```

## 📚 Notebooks

Jupyter notebooks for exploration and analysis:

- `notebooks/01_data_exploration.ipynb`: Dataset analysis
- `notebooks/02_model_comparison.ipynb`: Algorithm comparison
- `notebooks/03_error_analysis.ipynb`: Error analysis and insights

## 🛠️ Development

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

## 📋 Requirements

See `requirements.txt` for full dependency list. Key packages:

- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **nltk**: Natural language processing
- **matplotlib/seaborn**: Visualization
- **dvc**: Data version control
- **pyyaml**: Configuration management

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OSDG Community](https://zenodo.org/records/11441197) for the dataset
- [DVC](https://dvc.org/) for MLOps capabilities
- [scikit-learn](https://scikit-learn.org/) for ML algorithms

## 📞 Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---

**Built with ❤️ for Sustainable Development Goals**