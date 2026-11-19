import pandas as pd


def normalize_allele(value: object) -> str:
    # Return a clean allele string for length-based feature engineering.
    if pd.isna(value):
        return ""

    value = str(value).strip()

    # Treat ClinVar placeholders and missing-like values as empty alleles.
    if value.lower() in {"na", "nan", "-", ""}:
        return ""

    return value


def classify_mutation_type(ref: object, alt: object) -> str:
    # Classify a variant as SNV, Deletion, Insertion, or Other. 
    ref = normalize_allele(ref)
    alt = normalize_allele(alt)

    if not ref or not alt or ref == alt:
        return "Other"
    
    if len(ref) == 1 and len(alt) == 1:
        return "SNV"
    
    if len(ref) > len(alt):
        return "Deletion"
    
    if len(alt) > len(ref):
        return "Insertion"

    return "Other"


def add_variant_features(df: pd.DataFrame) -> pd.DataFrame:
    # Add allele length and mutation type features to a ClinVar dataframe.
    df = df.copy()

    # Use VCF alleles because they are suitable for length-based features.
    df["ref"] = df["ReferenceAlleleVCF"].apply(normalize_allele)
    df["alt"] = df["AlternateAlleleVCF"].apply(normalize_allele)

    df["ref_length"] = df["ref"].apply(len)
    df["alt_length"] = df["alt"].apply(len)
    df["length_difference"] = df["alt_length"] - df["ref_length"]

    df["mutation_type"] = [
        classify_mutation_type(ref, alt)
        for ref, alt in zip(df["ref"], df["alt"])
    ]

    df["is_snv"] = (df["mutation_type"] == "SNV").astype(int)
    df["is_deletion"] = (df["mutation_type"] == "Deletion").astype(int)
    df["is_insertion"] = (df["mutation_type"] == "Insertion").astype(int)

    return df
