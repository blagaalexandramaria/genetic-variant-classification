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
- Random Forest model training
- Initial model evaluation
- Probability-based evaluation metrics

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
│   ├── literature_triage.md
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── probability_metrics.txt
│   ├── feature_importance.csv
│   └── feature_importance.png
│
├── models/
│   ├── final_model.pkl
│   └── random_forest_model.pkl
│
├── src/
│   ├── config.py
│   ├── features.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
```

The `src/` directory now contains the first version of a reproducible pipeline. The notebooks are still kept for exploration and earlier experiments.

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
python src/preprocessing.py --mode balanced-sample --samples-per-class 1000
```

The balanced-sample mode reads the raw ClinVar file in chunks, maps clinical significance labels to binary targets, generates variant-level features, and saves the processed dataset to:

```text
data/processed/clinvar_features.csv
```

### 2. Training

Train the current Random Forest model:

```bash
python src/train.py
```

The trained model is saved to:

```text
models/random_forest_model.pkl
```

### 3. Evaluation

Evaluate the saved model:

```bash
python src/evaluate.py
```

The evaluation script saves:

```text
reports/classification_report.txt
reports/confusion_matrix.png
reports/probability_metrics.txt
```

---

## Models Evaluated

### Baseline Models

| Model | Accuracy | F1-score (Pathogenic) |
|-------|---------:|----------------------:|
| Baseline 1 | 84% | 0.53 |
| Baseline 2 | 77% | 0.44 |
| Baseline 3 | 79% | 0.50 |
| Baseline 4 | 81% | 0.53 |

### Current Scripted Model

The current scripted model combines:

- Random Forest Classifier
- OneHotEncoder
- ColumnTransformer
- Balanced class weights

This model is an initial reproducible baseline and will likely be improved as the project develops.

---

## Preliminary Results

The latest scripted run used a balanced sample with 1,000 benign/likely benign variants and 1,000 pathogenic/likely pathogenic variants.

| Metric | Value |
|--------|------:|
| Accuracy | 0.84 |
| Precision (Pathogenic) | 0.87 |
| Recall (Pathogenic) | 0.80 |
| F1-score (Pathogenic) | 0.84 |
| ROC-AUC | 0.951 |
| PR-AUC | 0.951 |

These results are preliminary and should be interpreted as a current development checkpoint, not as a finalized benchmark.

---

## Feature Importance

The most influential predictors identified by the current model include:

- Mutation type
- Length difference
- Reference allele length
- Alternate allele length
- Deletion mutations

Gene-level information also contributed to prediction performance, suggesting that both variant structure and biological context may be relevant for pathogenicity classification.

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

To generate a small balanced dataset and run the current model:

```bash
python src/preprocessing.py --mode balanced-sample --samples-per-class 1000
python src/train.py
python src/evaluate.py
```

---

## Development Roadmap

Planned improvements include:

- Add model comparison with DummyClassifier and Logistic Regression
- Add a larger balanced sample for more stable evaluation
- Add a small sample dataset for quick testing
- Add clear instructions for downloading or regenerating ClinVar data
- Add more robust validation
- Add ablation studies by feature group
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
