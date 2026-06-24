# Genetic Variant Classification Using ClinVar

## Project Status

This project is currently in an early development stage.

The current version includes exploratory notebooks and an initial reproducible Python pipeline for preprocessing, training, and evaluation using ClinVar-derived data.

The results reported here are preliminary and are intended to document the current progress of the project.

---

## Overview

This project investigates the classification of human genetic variants as **benign** or **pathogenic** using Machine Learning techniques and data derived from the ClinVar database.

The current workflow includes:

- Data exploration
- Data preprocessing
- Feature engineering
- Balanced sample generation
- Model comparison
- Initial model evaluation
- Probability-based evaluation metrics
- Logistic Regression coefficient analysis

The goal is to explore how genomic and clinical features can contribute to variant pathogenicity prediction.

---

## Dataset

### Source

- ClinVar Database

### Processed Dataset

The experiments use an engineered dataset generated from ClinVar records:

```text
clinvar_features.csv
```

The full raw and processed datasets are not intended to be committed to GitHub because of their size. The repository is expected to include the code, notebooks, reports, and generated outputs, while large data files should be downloaded or regenerated separately.

The current preprocessing script supports:

- a development mode for quick testing on a small number of rows;
- a balanced-sample mode that reads the large ClinVar file in chunks and collects an equal number of benign and pathogenic variants.

### Target Classes

| Clinical Significance | Target |
|----------------------|-------:|
| Benign | 0 |
| Likely benign | 0 |
| Pathogenic | 1 |
| Likely pathogenic | 1 |

---

## Current Project Structure

```text
genetic-variant-classification/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_models.ipynb
│   └── 05_final_model.ipynb
│
├── reports/
│   ├── initial_report.md
│   └── literature_triage.md
│
├── models/
│   └── generated model artifacts
│
├── results/
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   └── logistic_regression_coefficients.png
│   └── metrics/
│       ├── ablation_study.csv
│       ├── classification_report.txt
│       ├── logistic_regression_coefficients.csv
│       ├── model_comparison.csv
│       └── probability_metrics.txt
│
├── src/
│   ├── config.py
│   ├── features.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── ablation.py
│   └── metrics.py
│
├── requirements.txt
└── README.md
```

The `src/` directory now contains the first version of a reproducible pipeline. The notebooks are still kept for exploration and earlier experiments.

The `reports/` directory is used for written analysis and manually curated report artifacts. Automatically generated metrics and figures are saved under `results/metrics/` and `results/figures/`. Model artifacts are generated locally under `models/` and are not intended to be committed by default.

---

## Feature Engineering

The following features were extracted from ClinVar records.

### Structural Features

- Reference allele length
- Alternate allele length
- Length difference

### Mutation Features

- Mutation type
- SNV indicator
- Deletion indicator
- Insertion indicator

### Genomic Features

- Chromosome
- GeneSymbol

### Clinical Features

- OriginSimple
- ReviewStatus

---

## Current Pipeline

### 1. Preprocessing

Development mode:

```bash
python src/preprocessing.py --mode dev --nrows 10000
```

Balanced sample mode:

```bash
python src/preprocessing.py --mode balanced-sample --samples-per-class 10000
```

The balanced-sample mode reads the raw ClinVar file in chunks, maps clinical significance labels to binary targets, generates variant-level features, and saves the processed dataset to:

```text
data/processed/clinvar_features.csv
```

### 2. Training

Train the current comparison models:

```bash
python src/train.py
```

The training script currently compares:

- DummyClassifier
- Logistic Regression
- Random Forest Classifier

The trained models are generated locally under:

```text
models/dummy_model.pkl
models/logistic_regression_model.pkl
models/random_forest_model.pkl
```

The model comparison table is saved to:

```text
results/metrics/model_comparison.csv
```

### 3. Evaluation

Evaluate the saved model:

```bash
python src/evaluate.py
```

The evaluation script currently evaluates the Logistic Regression model, which is the strongest model in the current comparison. It saves:

```text
results/metrics/classification_report.txt
results/metrics/logistic_regression_coefficients.csv
results/metrics/probability_metrics.txt
results/figures/confusion_matrix.png
results/figures/logistic_regression_coefficients.png
```

### 4. Ablation Study

Run the feature-group ablation study:

```bash
python src/ablation.py
```

The ablation script compares Logistic Regression performance across structural, genomic, clinical metadata, and full feature sets. Results are saved to:

```text
results/metrics/ablation_study.csv
```

---

## Models Evaluated

| Model | Accuracy | F1-score (Pathogenic) |
|-------|---------:|----------------------:|
| DummyClassifier | 0.5000 | 0.0000 |
| Logistic Regression | 0.9183 | 0.9185 |
| Random Forest | 0.8175 | 0.7925 |

### Current Scripted Models

The current scripted models use a shared scikit-learn preprocessing pipeline based on:

- OneHotEncoder
- ColumnTransformer
- categorical feature encoding
- numeric feature passthrough

Logistic Regression and Random Forest use balanced class weights. These models are initial reproducible baselines and will likely be improved as the project develops.

---

## Preliminary Results

The latest scripted run used a balanced sample with 10,000 benign/likely benign variants and 10,000 pathogenic/likely pathogenic variants.

| Model | Accuracy | Precision (Pathogenic) | Recall (Pathogenic) | F1-score (Pathogenic) | ROC-AUC | PR-AUC |
|-------|---------:|------------------------:|---------------------:|----------------------:|--------:|-------:|
| DummyClassifier | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 |
| Logistic Regression | 0.9183 | 0.9160 | 0.9210 | 0.9185 | 0.9726 | 0.9728 |
| Random Forest | 0.8175 | 0.9183 | 0.6970 | 0.7925 | 0.9268 | 0.9287 |

In this preliminary run, Logistic Regression outperformed Random Forest across accuracy, pathogenic F1-score, ROC-AUC, and PR-AUC. This suggests that a simpler linear baseline is currently more effective for the engineered feature set than the Random Forest configuration.

These results are preliminary and should be interpreted as a current development checkpoint, not as a finalized benchmark.

---

## Ablation Study

The ablation study evaluates how different feature groups contribute to pathogenicity prediction using Logistic Regression.

| Feature Set | Accuracy | Precision (Pathogenic) | Recall (Pathogenic) | F1-score (Pathogenic) | ROC-AUC | PR-AUC |
|-------------|---------:|------------------------:|---------------------:|----------------------:|--------:|-------:|
| Structural only | 0.6025 | 0.7863 | 0.2815 | 0.4146 | 0.6066 | 0.6041 |
| Structural + genomic | 0.9003 | 0.8817 | 0.9245 | 0.9026 | 0.9605 | 0.9575 |
| Structural + clinical metadata | 0.7673 | 0.9102 | 0.5930 | 0.7181 | 0.8188 | 0.8248 |
| Full feature set | 0.9183 | 0.9160 | 0.9210 | 0.9185 | 0.9726 | 0.9728 |

The results suggest that structural variant features alone are not sufficient for strong classification performance. Adding genomic context, especially gene-level information, substantially improves performance. Clinical metadata also improves over structural features alone, but these features should be interpreted carefully because they may reflect ClinVar annotation and review patterns rather than purely biological signal.

---

## Model Interpretation

Because Logistic Regression is the current primary model, interpretation is based on model coefficients rather than Random Forest feature importance.

The coefficient analysis is saved to:

```text
results/metrics/logistic_regression_coefficients.csv
results/figures/logistic_regression_coefficients.png
```

The largest absolute coefficients currently include several gene-level and metadata-derived features. These coefficients should be interpreted carefully:

- positive coefficients increase the model's predicted probability of the pathogenic class;
- negative coefficients decrease the model's predicted probability of the pathogenic class;
- gene-level coefficients may reflect strong ClinVar-specific patterns in the current dataset;
- metadata-derived coefficients may reflect annotation and review patterns rather than purely biological signal.

Random Forest feature importance was used in earlier exploratory runs, but it is no longer the primary interpretation artifact for the current scripted evaluation.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/genetic-variant-classification.git
cd genetic-variant-classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Because the full data files are large, they are not included directly in the repository. The notebooks and scripts expect the processed data to be available locally under `data/processed/`.

The scripted pipeline expects the raw ClinVar file at:

```text
data/raw/variant_summary.txt
```

To generate the current balanced dataset and run the pipeline:

```bash
python src/preprocessing.py --mode balanced-sample --samples-per-class 10000
python src/train.py
python src/evaluate.py
```

---

## Development Roadmap

Planned improvements include:

- Add a larger or repeated balanced sampling strategy for more stable evaluation
- Add a small sample dataset for quick testing
- Add clear instructions for downloading or regenerating ClinVar data
- Add more robust validation
- Add permutation importance or SHAP-based interpretability
- Experiment with XGBoost and LightGBM
- Explore sequence-based features and k-mer representations
- Investigate deep learning or transformer-based genomic models
- Validate the model on independent genomic datasets

---

## Author

**Alexandra Blaga**

Computer Science Student interested in:

- Artificial Intelligence
- Bioinformatics
- Genomics
- Machine Learning
