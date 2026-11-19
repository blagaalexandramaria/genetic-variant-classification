# Genetic Variant Classification Using ClinVar

This project explores the classification of genetic variants as benign or pathogenic using machine learning and data from ClinVar.

It includes a Python pipeline for preparing the dataset, training models, and evaluating their performance. A desktop application built with CustomTkinter provides access to predictions, model comparisons, and experimental results.

The project is a research prototype. Its predictions have not been clinically validated and should not be used for medical decisions.

## Overview

The main goal is to examine how well common machine learning algorithms can classify genetic variants using a relatively small set of genomic and structural features.

The project covers:

- Data exploration and preprocessing
- Feature engineering
- Logistic Regression and Random Forest classification
- Model evaluation and comparison
- Feature importance analysis
- Ablation experiments
- Gene-based evaluation
- A graphical interface for predictions and model analysis

Three models were evaluated: Dummy Classifier, Logistic Regression, and Random Forest.

Logistic Regression achieved the best overall results in the main evaluation, although Random Forest had higher precision for the pathogenic class.

## Dataset

### Source

The data comes from ClinVar, a public database maintained by the National Center for Biotechnology Information (NCBI).

The preprocessing pipeline uses the `variant_summary.txt` file and selects variants associated with the GRCh38 genome assembly.

Four clinical significance labels are included:

| Clinical significance | Target |
|---|---:|
| Benign | 0 |
| Likely benign | 0 |
| Pathogenic | 1 |
| Likely pathogenic | 1 |

Other clinical significance categories are excluded.

### Preprocessing

The raw ClinVar file is processed in chunks to avoid loading the entire dataset into memory.

The preprocessing pipeline filters the records, assigns binary labels, removes duplicate variants, checks for conflicting classifications, and generates the features used during training.

Variants are identified using their chromosome, genomic position, reference allele, and alternate allele.

The preprocessing analysis identified 1,598,540 unique eligible variants before sampling.

The final dataset contains 100,000 variants:

| Class | Variants |
|---|---:|
| Benign | 50,000 |
| Pathogenic | 50,000 |
| Total | 100,000 |

The classes were balanced intentionally to make model comparisons easier. This distribution does not reflect the natural frequency of pathogenic variants.

The processed dataset is saved as:

```text
data/processed/clinvar_features.csv
```

The raw and processed datasets are kept locally because of their size.

## Features

The models use seven input features.

| Feature | Type | Description |
|---|---|---|
| Chromosome | Categorical | Chromosome containing the variant |
| GeneSymbol | Categorical | Gene associated with the variant |
| mutation_type | Categorical | Mutation category |
| OriginSimple | Categorical | Variant-origin metadata |
| ref_length | Numerical | Reference allele length |
| alt_length | Numerical | Alternate allele length |
| length_difference | Numerical | Difference between alternate and reference allele lengths |

Mutation types are derived from the reference and alternate alleles and grouped into SNV, insertion, deletion, or other.

`ReviewStatus` was excluded from the final predictor set to reduce the risk of learning patterns associated with ClinVar's annotation process.

The models do not use sequence conservation, population allele frequencies, functional annotations, or protein-level effects.

## Project Structure

```text
genetic-variant-classification/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── styles.py
│   ├── dashboard.py
│   ├── prediction.py
│   └── analysis.py
│
├── data/
│   ├── raw/
│   │   └── variant_summary.txt
│   └── processed/
│       └── clinvar_features.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_evaluation.ipynb
│   └── 03_experimental_analysis.ipynb
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── metrics.py
│   ├── evaluate.py
│   ├── ablation.py
│   └── gene_split_evaluation.py
│
├── models/
│   ├── dummy_model.pkl
│   ├── logistic_regression_model.pkl
│   └── random_forest_model.pkl
│
├── results/
│   ├── figures/
│   └── metrics/
│       ├── model_comparison.csv
│       ├── ablation_study.csv
│       ├── gene_split_comparison.csv
│       └── gene_split_summary.csv
│
│
├── requirements.txt
└── README.md
```

The `src/` directory contains the preprocessing, training, and evaluation scripts. The desktop application is implemented in `app/`.

The notebooks document the analyses, while generated metrics and figures are stored in `results/`.

Trained models are saved in `models/`.

## Machine Learning Pipeline

### Preprocessing

The preprocessing script supports a development mode for testing on smaller datasets and a balanced-sampling mode for generating the training dataset.

Development mode:

```bash
python3 -m src.preprocessing --mode dev --nrows 10000
```

Generate a balanced dataset with 50,000 variants per class:

```bash
python3 -m src.preprocessing --mode balanced-sample --samples-per-class 50000
```

The output is saved to `data/processed/clinvar_features.csv`.

### Training

Three models are trained using scikit-learn:

- Dummy Classifier, used as a baseline
- Logistic Regression
- Random Forest

Categorical variables are encoded with `OneHotEncoder`, while numerical variables are passed through the preprocessing pipeline.

Logistic Regression and Random Forest use balanced class weights.

The dataset is divided into training and testing subsets using a stratified 80/20 split.

| Subset | Variants |
|---|---:|
| Training | 80,000 |
| Testing | 20,000 |

Run the training script:

```bash
python3 -m src.train
```

The trained models are saved in `models/`, and the comparison results are written to `results/metrics/model_comparison.csv`.

### Evaluation

Model performance is measured using accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC.

Precision, recall, and F1-score are calculated for the pathogenic class.

The evaluation also generates confusion matrices, ROC curves, Precision–Recall curves, and model interpretation results.

Run:

```bash
python3 -m src.evaluate
```

Generated files are stored in `results/metrics/` and `results/figures/`.

## Model Results

The following results were obtained using the balanced dataset of 100,000 variants.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Dummy Classifier | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 |
| Logistic Regression | 0.7641 | 0.8173 | 0.6803 | 0.7425 | 0.8431 | 0.8578 |
| Random Forest | 0.7275 | 0.8973 | 0.5137 | 0.6534 | 0.8039 | 0.8301 |

Logistic Regression achieved the highest accuracy, recall, F1-score, ROC-AUC, and PR-AUC.

Random Forest had higher precision for pathogenic predictions, but its recall was considerably lower. In other words, a greater proportion of its pathogenic predictions were correct, but it missed more variants labeled as pathogenic.

Logistic Regression was selected as the primary model because it provided a better balance between precision and recall.

This comparison is specific to the current dataset and model configurations. It does not establish that Logistic Regression is generally better suited to genetic variant classification.

## Model Interpretation

Two methods were used to examine how the models make predictions.

### Logistic Regression

Logistic Regression coefficients describe the relationship between encoded input features and the predicted log-odds of the pathogenic class.

Positive coefficients increase the predicted log-odds of pathogenicity, while negative coefficients decrease them, with other features held constant.

Their magnitudes depend on feature encoding and scaling. Correlations between predictors can also affect their interpretation.

### Random Forest

Random Forest feature importance is based on reductions in node impurity across the decision trees.

Features with higher importance values contribute more to the model's splitting decisions.

Unlike Logistic Regression coefficients, these values do not indicate whether a feature is associated with a higher or lower pathogenicity score.

Both methods describe the behavior of the trained models. They should not be interpreted as evidence of biological causation.

## Ablation Study

An ablation study was conducted to examine how different groups of features affect Logistic Regression performance.

Four configurations were evaluated:

| Feature set | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Structural Only | 0.7069 | 0.8971 | 0.4674 | 0.6146 | 0.7073 | 0.6902 |
| Structural + Genomic | 0.7535 | 0.8001 | 0.6759 | 0.7328 | 0.8333 | 0.8465 |
| Structural + Origin | 0.7246 | 0.8978 | 0.5069 | 0.6480 | 0.7265 | 0.7154 |
| Full Features | 0.7641 | 0.8173 | 0.6803 | 0.7425 | 0.8431 | 0.8578 |

Structural features alone achieved an F1-score of 0.6146.

Adding genomic information increased the F1-score to 0.7328 and improved pathogenic recall from 0.4674 to 0.6759.

The full feature set achieved the highest F1-score, accuracy, recall, ROC-AUC, and PR-AUC.

These results indicate that genomic information contributes substantially to the model's performance. Origin metadata provides a smaller improvement when added to structural features.

The experiments also show a trade-off between precision and recall. Some reduced feature sets achieved higher precision but identified fewer pathogenic variants.

Run the ablation study with:

```bash
python3 -m src.ablation
```

Results are saved to:

```text
results/metrics/ablation_study.csv
```

## Gene-Based Evaluation

### Motivation

In a random train-test split, variants associated with the same gene may appear in both subsets.

This can make it difficult to determine how well a model performs on variants from genes it has not encountered during training.

To examine this, an additional evaluation was performed using gene-based splitting.

### Method

The experiment uses `GroupShuffleSplit` from scikit-learn, with gene symbols defining the groups.

For each split, genes assigned to the test set are excluded from training.

Five gene-based splits were evaluated.

`GeneSymbol` was removed from the predictors for both the random-split baseline and the gene-based experiments. This keeps the feature representation comparable.

### Results

| Metric | Random Split | Gene-Based Mean | Standard Deviation |
|---|---:|---:|---:|
| Accuracy | 0.7284 | 0.7260 | 0.0078 |
| Precision | 0.8568 | 0.8567 | 0.0180 |
| Recall | 0.5485 | 0.5435 | 0.0204 |
| F1-score | 0.6688 | 0.6650 | 0.0197 |
| ROC-AUC | 0.7584 | 0.7459 | 0.0208 |
| PR-AUC | 0.7945 | 0.7849 | 0.0259 |

The gene-based results are reported as the mean and sample standard deviation across five splits.

Average performance was close to the random-split baseline. For example, the F1-score decreased from 0.6688 to 0.6650.

However, the test sets differed in size, gene composition, and class distribution. Performance also varied between iterations.

The results suggest that the model can retain similar performance on the held-out gene groups examined here, but they are not sufficient to establish generalization to all unseen genes.

Repeated `GroupShuffleSplit` test sets may overlap, so the five experiments should not be treated as independent cross-validation folds.

Run the experiment:

```bash
python3 -m src.gene_split_evaluation
```

The results are saved to:

```text
results/metrics/gene_split_comparison.csv
results/metrics/gene_split_summary.csv
```

## Desktop Application

The project includes a desktop interface built with CustomTkinter.

It contains three pages.

### Dashboard

The Dashboard displays dataset statistics and model evaluation results.

Users can view the class distribution, compare model performance, and examine ROC and Precision–Recall curves.

### Variant Prediction

The prediction page allows users to enter information about a genetic variant and obtain a classification from a trained model.

The inputs include:

- Gene symbol and chromosome
- Reference and alternate alleles
- Variant origin
- Classification model

Users can choose between Logistic Regression and Random Forest.

The application derives the required structural features from the supplied alleles and passes the resulting data to the selected model.

The output includes a predicted class and a model-generated pathogenicity score.

The score reflects the model's learned patterns in the training data. It is not a clinically calibrated probability of pathogenicity.

### Model Analysis

The Model Analysis page displays the experimental results generated by the Python pipeline.

It includes Logistic Regression coefficients, Random Forest feature importance, ablation results, and comparisons between random and gene-based evaluation.

### Running the Application

From the project directory:

```bash
python3 -m app.main
```

The application requires the trained models and generated evaluation files to be available locally.

## Jupyter Notebooks

Three notebooks document the main analyses.

### 01 — Data Exploration

`01_data_exploration.ipynb`

Examines the processed dataset, including class distribution, mutation types, chromosomes, genes, allele lengths, missing values, and duplicate records.

### 02 — Model Evaluation

`02_model_evaluation.ipynb`

Compares the three classification models using evaluation metrics, confusion matrices, ROC curves, and Precision–Recall curves.

It also explains the selection of Logistic Regression as the primary model.

### 03 — Experimental Analysis

`03_experimental_analysis.ipynb`

Covers model interpretation, the ablation study, and gene-based evaluation.

It also examines performance variability and differences in class distribution across the gene-based splits.

The notebooks use the datasets and results generated by the pipeline. They do not repeat model training.

## Installation

### Requirements

The project uses Python and the libraries listed in `requirements.txt`.

The main dependencies are Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Joblib, CustomTkinter, Pillow, and Jupyter.

### 1. Clone the Repository

```bash
git clone https://github.com/blagaalexandramaria/genetic-variant-classification
cd genetic-variant-classification
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the Dataset

Download the ClinVar variant summary file from NCBI and place it in:

```text
data/raw/variant_summary.txt
```

Generate the balanced dataset:

```bash
python -m src.preprocessing --mode balanced-sample --samples-per-class 50000
```

### 5. Train and Evaluate the Models

Run the scripts in the following order:

```bash
python -m src.train
python -m src.evaluate
python -m src.ablation
python -m src.gene_split_evaluation
```

The scripts generate the model files, evaluation metrics, and figures used by the application and notebooks.

### 6. Launch the Application

```bash
python -m app.main
```

### 7. Open the Notebooks

```bash
jupyter notebook
```

Open the notebooks from the `notebooks/` directory.

## Limitations

The models use a limited representation of genetic variants. Features such as population allele frequencies, functional consequences, and evolutionary conservation are not included.

The dataset was sampled from ClinVar and balanced artificially. As a result, the class distribution differs from that of naturally occurring variants, and the reported precision values may not transfer to other populations or datasets.

ClinVar classifications may also reflect differences in submission practices and available evidence.

Although gene-based splitting was used to examine generalization, the evaluation was limited to five repeated splits. No independent external dataset was used.

The predicted pathogenicity scores have not been calibrated or validated for clinical interpretation.

The application is intended for research and demonstration purposes. Its predictions should not be used for diagnosis or other medical decisions.

## Future Work

Possible extensions include independent validation, probability calibration, and more detailed analysis of performance across individual genes.

Additional biological features could also be incorporated, including population frequencies, conservation scores, and functional annotations.

Other approaches worth investigating include permutation importance, SHAP-based interpretation, and sequence-based representations.

## Author

**Alexandra Blaga**

Computer Science student with interests in artificial intelligence, machine learning, bioinformatics, and genomics.
