import argparse

import pandas as pd

from config import(
    RAW_CLINVAR_FILE,
    PROCESSED_CLINVAR_FILE,
    BENIGN_LABELS,
    PATHOGENIC_LABELS,
)
from features import add_variant_features

USECOLS = [
    "ClinicalSignificance",
    "GeneSymbol",
    "Chromosome",
    "OriginSimple",
    "ReviewStatus",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
]

def parse_args() -> argparse.Namespace:
    # Parse command-line arguments for the preprocessing script
    parser = argparse.ArgumentParser(
        description="Preprocess ClinVar data for variant classification."
    )

    parser.add_argument(
        "--mode",
        choices=["dev", "balanced-sample"],
        default="dev",
        help="Preprocessing mode. Currently only dev is implemented.",
    )

    parser.add_argument(
        "--nrows",
        type=int,
        default=10_000,
        help="Number of rows to read in development mode.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=50_000,
        help="Number of benign and pathogenic rows to keep in balanced-sample mode.",
    )

    return parser.parse_args()

def load_balanced_sample(samples_per_class: int) -> pd.DataFrame:
    # Load a balanced benign / pathogenic sample using chunked reading
    benign_chunks = []
    pathogenic_chunks = []

    benign_count = 0
    pathogenic_count = 0

    chunks = pd.read_csv(
        RAW_CLINVAR_FILE,
        sep="\t",
        usecols=USECOLS,
        dtype=str,
        chunksize=100_000,
    )

    for chunk in chunks:
        # Map labels early so we can filter only the binary classification classes
        chunk["target"] = chunk["ClinicalSignificance"].apply(map_target)
        chunk = chunk.dropna(subset=["target"]).copy()
        chunk["target"] = chunk["target"].astype(int)

        benign = chunk[chunk["target"] == 0]
        pathogenic = chunk[chunk["target"] == 1]

        if benign_count < samples_per_class:
            needed = samples_per_class - benign_count
            selected = benign.head(needed)
            benign_chunks.append(selected)
            benign_count += len(selected)
        
        if pathogenic_count < samples_per_class:
            needed = samples_per_class - pathogenic_count
            selected = pathogenic.head(needed)
            pathogenic_chunks.append(selected)
            pathogenic_count += len(selected)
        
        print(
            f"Collected benign={benign_count:,},"
            f"pathogenic={pathogenic_count:,}"
        )

        if benign_count >= samples_per_class and pathogenic_count >= samples_per_class:
            break

        if benign_count == 0 or pathogenic_count == 0:
            raise ValueError("Could not collect both classes from the ClinVar file.")

        df = pd.concat(benign_chunks +pathogenic_chunks, ignore_index=True)

        # Shuffle so classes are not grouped together
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df
    
def map_target(label: object) -> int | None:
    # Map ClinVar clinical significance labels to binary targets
    if pd.isna(label):
        return None

    label = str(label).strip()

    if label in BENIGN_LABELS:
        return 0

    if label in PATHOGENIC_LABELS:
        return 1

    return None


def load_dev_data(nrows: int) -> pd.DataFrame:
    # Load a small subset of ClinVar data for fast development runs
    return pd.read_csv(
        RAW_CLINVAR_FILE,
        sep="\t",
        usecols=USECOLS,
        dtype=str,
        nrows=nrows,
    )


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Create target labels and variant-level features
    df = df.copy()

    if "target" not in df.columns:
        # Keep only labels that can be mapped to the binary classification task
        df["target"] = df["ClinicalSignificance"].apply(map_target)
        df = df.dropna(subset=["target"]).copy()
        df["target"] = df["target"].astype(int)

    # Add allele length, mutation type, and binary mutation indicators
    df = add_variant_features(df)

    return df


def save_processed_data(df: pd.DataFrame) -> None:
    # Save the processed dataframe to the configured output path
    PROCESSED_CLINVAR_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CLINVAR_FILE, index=False)

    print(f"Saved processed dataset to: {PROCESSED_CLINVAR_FILE}")
    print(f"Rows: {len(df):,}")
    print(df["target"].value_counts())


def main() -> None:
    args = parse_args()

    if args.mode == "dev":
        df = load_dev_data(nrows=args.nrows)
    
    elif args.mode == "balanced-sample":
        df = load_balanced_sample(samples_per_class=args.samples_per_class)
    
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    df = preprocess_dataframe(df)
    save_processed_data(df)


if __name__ == "__main__":
    main()