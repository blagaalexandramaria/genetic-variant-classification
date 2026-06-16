# Genetic Variant Classification Using ClinVar

## Project Status

This project is currently in an early development stage.

The current version focuses on exploratory analysis, preprocessing, feature engineering, baseline model comparison, and an initial final model experiment using ClinVar-derived data. The project is notebook-first at this stage, while the reusable Python scripts in `src/` are planned for future development.

The results reported here are preliminary and are intended to document the current progress of the project.

---

## Overview

This project investigates the classification of human genetic variants as **benign** or **pathogenic** using Machine Learning techniques and data derived from the ClinVar database.

The current workflow includes:

- Data exploration
- Data preprocessing
- Feature engineering
- Baseline model development
- Initial model evaluation
- Feature importance analysis

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
│   ├── confusion_matrix.png
│   ├── feature_importance.csv
│   └── feature_importance.png
│
├── models/
│   └── final_model.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
```

Note: the `src/` scripts are placeholders for the future reproducible pipeline. At the moment, the main implementation is contained in the notebooks.

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

## Models Evaluated

### Baseline Models

| Model | Accuracy | F1-score (Pathogenic) |
|-------|---------:|----------------------:|
| Baseline 1 | 84% | 0.53 |
| Baseline 2 | 77% | 0.44 |
| Baseline 3 | 79% | 0.50 |
| Baseline 4 | 81% | 0.53 |

### Initial Final Model

The current final experiment combines:

- Random Forest Classifier
- OneHotEncoder
- ColumnTransformer
- Balanced class weights

This model is an initial version and will likely be improved as the project develops.

---

## Preliminary Results

| Metric | Value |
|--------|------:|
| Accuracy | 86% |
| Precision (Pathogenic) | 0.67 |
| Recall (Pathogenic) | 0.54 |
| F1-score (Pathogenic) | 0.60 |

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

Because the full data files are large, they are not included directly in the repository. The notebooks currently expect the processed data to be available locally under `data/processed/`.

---

## Development Roadmap

Planned improvements include:

- Move notebook logic into reusable scripts in `src/`
- Add a small sample dataset for quick testing
- Add clear instructions for downloading or regenerating ClinVar data
- Improve reproducibility of the training and evaluation pipeline
- Add more robust validation
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
