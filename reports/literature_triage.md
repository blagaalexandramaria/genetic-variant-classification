# Literature Triage for Genetic Variant Classification Project

## Purpose

This document summarizes the most useful papers from the local literature folder for improving the current ClinVar-based genetic variant classification project.

The goal is not to cite every collected paper, but to select a focused set of references that can support:

- the problem formulation;
- the model design;
- feature engineering choices;
- evaluation strategy;
- interpretability;
- limitations and ethical framing.

The current project should be positioned as an exploratory, literature-informed machine learning pipeline for ClinVar variant pathogenicity classification and prioritization.

---

## Tier A: Core Papers To Use Directly

These papers are the most relevant to the current project and should be used in the Related Work / Methodology section.

| Priority | Paper | Why It Matters | How To Use It In This Project |
|---:|---|---|---|
| 1 | WilsonGenAI: a deep learning approach to classify pathogenic variants in Wilson Disease | Directly studies ML/deep learning for pathogenic vs benign variant classification. It also frames the motivation around ACMG/AMP guidelines and the difficulty of manually interpreting large numbers of variants. | Use as a direct precedent for disease/gene-specific AI-based variant pathogenicity classification. Cite it when explaining why ML can support variant interpretation. |
| 2 | A machine learning approach based on ACMG/AMP guidelines for genomic variant classification and prioritization | Very close to the project goal. It combines ACMG/AMP logic with annotation features and uses a probabilistic pathogenicity score. | Use as the strongest methodological inspiration. Adapt the idea of producing a pathogenicity probability and ranking variants by priority, not only predicting a hard class. |
| 3 | Opportunities and challenges for the computational interpretation of rare variation in clinically important genes | Strong conceptual paper about rare variant interpretation, scalability, clinical importance, and limitations. | Use in the Introduction and Limitations sections. This supports the argument that variant interpretation is difficult and that computational support is useful but not clinically definitive. |
| 4 | Translating cancer genomics into precision medicine with artificial intelligence | Broader review of AI in cancer genomics, including variant interpretation, clinical translation, transparency, reproducibility, and real-world adoption. | Use for background and limitations. Helpful when discussing why interpretability and reproducibility matter in genomic ML. |
| 5 | The Role of Artificial Intelligence in Identifying NF1 Gene Variants and Improving Diagnosis | Gene-specific review focused on AI tools for variant detection, classification, VUS reclassification, and structural modeling. | Use as support for future work: gene-specific models, VUS prioritization, REVEL/ClinPred-like scores, and structural features. |

---

## Tier B: Methodology And Framing Papers

These papers are useful, but they should support the project rather than define it.

| Priority | Paper | Why It Matters | How To Use It |
|---:|---|---|---|
| 6 | Artificial intelligence and machine learning approaches using gene expression and variant data for personalized medicine | Broad review of AI/ML approaches in genomics and precision medicine. | Use in Related Work to show that the project fits within genomic ML and personalized medicine. |
| 7 | Enabling the clinical application of artificial intelligence in genomics: AMIA perspective | Discusses clinical AI in genomics, data standards, dataset scale, bias, uncertainty, and translation barriers. | Use in Limitations / Ethics. Good support for saying the model is exploratory and not intended for clinical use. |
| 8 | Artificial intelligence-based approaches for the detection and prioritization of genomic mutations in congenital surgical diseases | Focuses on mutation detection/prioritization and compares multiple AI-based approaches. | Use as support for the phrase "variant prioritization", especially if adding ranked pathogenicity probabilities. |
| 9 | Applying artificial intelligence to uncover the genetic landscape of coagulation factors | A domain-specific AI/genomics review. | Use only if you want more examples of gene/domain-specific interpretation. Not essential. |

---

## Tier C: Interpretability And Algorithm Inspiration

These papers are not directly about ClinVar benign/pathogenic classification, but they are useful for improving the technical maturity of the project.

| Priority | Paper | Useful Idea | Possible Adaptation |
|---:|---|---|---|
| 10 | A Comparison of Global Sensitivity Analysis Methods for Explainable AI With an Application in Genomic Prediction | Compares sensitivity analysis methods in an XAI context. | Add permutation importance or sensitivity-style feature analysis as a more robust alternative to basic Random Forest feature importance. |
| 11 | DeepCOMBI: explainable artificial intelligence for genome-wide association studies | Shows how explainable AI can be used to extract knowledge from genomic models. | Use as inspiration for interpretability discussion, not as a direct method to implement immediately. |
| 12 | Cancer Metastasis Prediction and Genomic Biomarker Identification through ML and XAI | Uses LightGBM, CatBoost, XGBoost, AdaBoost, metrics like AUC/F1/recall, and SHAP. | Use as algorithm inspiration: compare gradient boosting models and add SHAP explanations. |
| 13 | IntelliGenes / Bioinformatics application note | Presents a pipeline for biomarker discovery and ML-based gene importance. | Use for future work around gene-level importance profiles and biomarker-style interpretation. |

---

## Tier D: Lower Priority / Context Only

The following categories are less directly useful for the current ClinVar classification project:

- cancer subtype classification from genetic signatures;
- treatment response prediction;
- polygenic risk score optimization;
- cloud/genomic data security;
- broad AI in precision medicine reviews;
- general ethics papers not focused on variant interpretation.

These papers can be cited only if the report expands into a broader discussion of AI in genomics. They should not dominate the current project because they move away from the main task: binary pathogenicity classification and variant prioritization.

---

## Recommended Short Reading List

For the current version of the project, focus on these 8 papers:

1. WilsonGenAI: a deep learning approach to classify pathogenic variants in Wilson Disease
2. A machine learning approach based on ACMG/AMP guidelines for genomic variant classification and prioritization
3. Opportunities and challenges for the computational interpretation of rare variation in clinically important genes
4. Artificial intelligence and machine learning approaches using gene expression and variant data for personalized medicine
5. Enabling the clinical application of artificial intelligence in genomics
6. The Role of Artificial Intelligence in Identifying NF1 Gene Variants and Improving Diagnosis
7. A Comparison of Global Sensitivity Analysis Methods for Explainable AI With an Application in Genomic Prediction
8. Cancer Metastasis Prediction and Genomic Biomarker Identification through Machine Learning and XAI

This is enough for a strong Related Work section without making the project look unfocused.

---

## Algorithmic Ideas To Extract

### 1. Probabilistic pathogenicity score

Instead of reporting only:

```text
Benign / Pathogenic
```

the model should output:

```text
pathogenic_probability = 0.82
```

This allows the project to be framed as both classification and prioritization.

### 2. Variant prioritization

After prediction, sort variants by predicted pathogenicity probability:

```text
variant_id,gene,mutation_type,pathogenic_probability
12345,BRCA1,SNV,0.94
12346,LDLR,Deletion,0.88
```

This makes the project more useful and closer to literature on clinical variant prioritization.

### 3. ACMG-aware feature engineering

The current project should not claim to implement ACMG/AMP guidelines fully, but it can include ACMG-inspired features where available:

- ReviewStatus
- NumberSubmitters
- clinical significance confidence
- conflict/no conflict indicators
- variant type
- origin
- gene context
- loss-of-function proxy
- allele length and length difference

Important framing:

> The project uses ACMG-inspired metadata where available, but does not implement a clinical ACMG/AMP rules engine.

### 4. Stronger model comparison

Add a structured comparison:

- DummyClassifier
- Logistic Regression
- Random Forest
- XGBoost or LightGBM
- optional CatBoost

Recommended metrics:

- Accuracy
- Precision for pathogenic class
- Recall for pathogenic class
- F1 for pathogenic class
- ROC-AUC
- PR-AUC
- Confusion matrix
- Brier score, if probability calibration is evaluated

### 5. Explainable AI

Use at least two interpretation methods:

- built-in model feature importance;
- permutation importance;
- SHAP on a sample, if feasible.

The report should distinguish between:

- biological signal, such as mutation type and allele length;
- database/annotation signal, such as ReviewStatus and NumberSubmitters.

### 6. Ablation studies

Run separate experiments with feature groups:

| Feature Set | Included Features | Purpose |
|---|---|---|
| Structural only | ref_length, alt_length, length_difference, mutation_type | Tests biological/variant-level signal. |
| Genomic context | structural + Chromosome + GeneSymbol | Tests gene/chromosome contribution. |
| Clinical metadata | structural + OriginSimple + ReviewStatus | Tests metadata contribution. |
| Full model | all features | Best current model. |

This is one of the easiest ways to make the project feel research-oriented.

### 7. Leakage-aware evaluation

The project should eventually compare:

- random stratified split;
- gene-based split.

Gene-based split is important because random splits may let the model see the same genes in both training and testing, inflating performance.

Suggested wording:

> In addition to a random stratified split, a stricter gene-level split is planned to evaluate whether the model generalizes to variants in genes not observed during training.

---

## How To Use This In The Report

### Related Work Structure

Suggested subsection:

```text
Related Work
1. AI for genetic variant interpretation
2. ACMG/AMP-inspired computational classification
3. Variant prioritization and probabilistic scoring
4. Explainable AI in genomic prediction
5. Limitations of clinical translation
```

### Methods Structure

Suggested subsection:

```text
Methods
1. Data source and label definition
2. Feature engineering
3. Model training
4. Model comparison
5. Evaluation metrics
6. Interpretability analysis
```

### Limitations Structure

Suggested subsection:

```text
Limitations
1. ClinVar label noise and reporting bias
2. Possible gene-level leakage
3. ReviewStatus as a database-derived signal
4. Lack of external validation
5. Not intended for clinical diagnosis
```

---

## Final Recommendation

For the next development step, do not try to implement deep learning immediately.

The most professional next version should focus on:

1. a reproducible pipeline;
2. a clear baseline comparison;
3. probabilistic pathogenicity scoring;
4. variant prioritization;
5. ablation studies;
6. interpretability;
7. transparent limitations.

This direction is more defensible in a master interview than simply adding a complex neural network.
