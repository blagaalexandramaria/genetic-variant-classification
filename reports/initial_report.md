# Genetic Variant Classification Using ClinVar

## 1. Introduction

Genetic variants are fundamental to understanding inherited diseases, cancer susceptibility, and personalized medicine. Public genomic databases such as ClinVar provide large collections of clinically annotated variants that can be leveraged for machine learning applications.

The objective of this project is to investigate whether machine learning models can distinguish between benign and pathogenic genetic variants using genomic and clinical information extracted from ClinVar records.

The project follows a complete machine learning workflow:

1. Data exploration
2. Data preprocessing
3. Feature engineering
4. Baseline model development
5. Model comparison
6. Final model construction
7. Model evaluation and interpretation

The final outcome is a predictive pipeline capable of classifying genetic variants based on biologically meaningful features.

---

# 2. Dataset Description

## Source

The data originates from the ClinVar database, a publicly available repository maintained by the National Center for Biotechnology Information (NCBI).

The original ClinVar release was processed and transformed into a machine-learning-ready dataset through several preprocessing and feature engineering stages.

The final models were trained using the engineered dataset:

```text
clinvar_features.csv
```

## Processed Dataset

After filtering, preprocessing and feature generation, the dataset contained:

- 1,612,093 genetic variants
- More than 28,000 unique genes

The dataset combines structural, genomic and clinical information extracted from ClinVar records.

## Target Variable

Clinical significance labels were transformed into a binary classification problem.

| Original Clinical Significance | Target |
|--------------------------------|---------|
| Benign | 0 |
| Likely benign | 0 |
| Pathogenic | 1 |
| Likely pathogenic | 1 |

### Class Distribution

| Class | Count |
|---------|---------:|
| Benign / Likely Benign | 1,644,069 |
| Pathogenic / Likely Pathogenic | 646,347 |

The resulting dataset exhibits moderate class imbalance, motivating the use of balanced class weights during model training.

---

# 3. Exploratory Data Analysis

Initial exploration revealed that most variants correspond to Single Nucleotide Variants (SNVs), followed by deletions and insertions.

## Mutation Type Distribution

| Mutation Type | Count |
|--------------|---------:|
| SNV | 1,390,106 |
| Deletion | 135,933 |
| Insertion | 69,333 |
| Other | 3,173 |

Several clinically relevant genes appeared frequently throughout the dataset, including:

- BRCA1
- BRCA2
- NF1
- ATM
- LDLR
- MLH1

These observations suggest that both mutation characteristics and gene-level information may contribute to pathogenicity prediction.

---

# 4. Feature Engineering

To transform raw ClinVar records into machine learning features, several biologically meaningful attributes were generated.

## Structural Features

- Reference allele length
- Alternate allele length
- Length difference

These features capture the physical size and structural impact of genomic alterations.

## Mutation Features

- Mutation type
- SNV indicator
- Deletion indicator
- Insertion indicator

These variables describe the biological nature of the variant.

## Genomic Features

- Chromosome
- GeneSymbol

These features provide genomic context for each variant.

## Clinical Features

- OriginSimple
- ReviewStatus

These attributes capture information regarding variant origin and confidence of clinical interpretation.

---

# 5. Baseline Models

## Baseline Model 1

### Features

- Chromosome
- Mutation type
- Reference allele length
- Alternate allele length
- Length difference

### Results

| Metric | Value |
|----------|----------|
| Accuracy | 84% |
| F1-score (Pathogenic) | 0.53 |

This model established the initial performance benchmark.

---

## Baseline Model 2

### Additional Feature

- GeneSymbol (LabelEncoder)

### Results

| Metric | Value |
|----------|----------|
| Accuracy | 77% |
| F1-score (Pathogenic) | 0.44 |

Performance decreased substantially.

The results indicate that LabelEncoder introduces artificial ordinal relationships between genes that do not exist biologically, leading to suboptimal learning.

---

## Baseline Model 3

### Additional Feature

- GeneSymbol (OneHotEncoder)

### Results

| Metric | Value |
|----------|----------|
| Accuracy | 79% |
| F1-score (Pathogenic) | 0.50 |

Replacing LabelEncoder with OneHotEncoder improved performance and provided a more appropriate representation of categorical genomic information.

---

## Baseline Model 4

### Additional Features

- OriginSimple
- ReviewStatus

### Results

| Metric | Value |
|----------|----------|
| Accuracy | 81% |
| F1-score (Pathogenic) | 0.53 |

The inclusion of clinical metadata improved the model's ability to identify pathogenic variants while maintaining strong overall performance.

---

# 6. Final Model

## Model Architecture

The final model combines:

- Random Forest Classifier
- OneHotEncoder
- ColumnTransformer
- Balanced class weights

## Features Used

- Chromosome
- GeneSymbol
- Mutation type
- Reference allele length
- Alternate allele length
- Length difference
- OriginSimple
- ReviewStatus

## Preprocessing Pipeline

Categorical variables were encoded using OneHotEncoder.

Numerical variables were passed directly to the classifier through a ColumnTransformer pipeline.

Class imbalance was addressed using balanced class weights during model training.

---

# 7. Final Model Performance

## Classification Results

| Metric | Value |
|----------|----------|
| Accuracy | 86% |
| Precision (Pathogenic) | 0.67 |
| Recall (Pathogenic) | 0.54 |
| F1-score (Pathogenic) | 0.60 |

The final model achieved the strongest overall performance among all evaluated approaches.

---

# 8. Confusion Matrix

| | Predicted Benign | Predicted Pathogenic |
|---|---:|---:|
| Actual Benign | 7607 | 501 |
| Actual Pathogenic | 873 | 1019 |

The model successfully identified 1,019 pathogenic variants while maintaining high specificity for benign variants.

---

# 9. Feature Importance Analysis

Feature importance analysis revealed that the model relies primarily on structural genomic characteristics.

The most influential predictors include:

1. Mutation type
2. Length difference
3. Reference allele length
4. Alternate allele length
5. Deletion-related features

Gene-level information also contributed significantly to prediction performance.

These findings suggest that both mutation structure and genomic context play important roles in determining variant pathogenicity.

---

# 10. Comparative Summary

| Model | Accuracy | F1-score (Pathogenic) |
|---------|---------:|---------:|
| Baseline 1 | 84% | 0.53 |
| Baseline 2 | 77% | 0.44 |
| Baseline 3 | 79% | 0.50 |
| Baseline 4 | 81% | 0.53 |
| Final Model | 86% | 0.60 |

The final model outperformed all baseline approaches and achieved the best balance between accuracy and pathogenic variant detection.

---

# 11. Conclusions

This project demonstrates that machine learning can effectively leverage genomic and clinical information from ClinVar-derived data to predict variant pathogenicity.

### Key Findings

- Structural mutation characteristics are strong predictors of pathogenicity.
- Proper encoding of categorical genomic information is essential.
- Clinical metadata improves classification quality.
- Gene-level information contributes useful biological context.
- The final model achieved the strongest overall performance among all evaluated configurations.

The resulting pipeline provides a reproducible framework for genetic variant classification using real-world genomic data.

---

# 12. Future Work

Potential extensions of this work include:

- XGBoost models
- LightGBM models
- Sequence-based feature extraction
- k-mer representations
- Deep learning approaches
- Transformer-based genomic architectures
- Integration of additional ClinVar annotations
- Validation on independent genomic datasets

These directions could further improve predictive performance and biological interpretability.