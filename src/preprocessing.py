import argparse
import random
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

from config import(
    RAW_CLINVAR_FILE,
    PROCESSED_CLINVAR_FILE,
    BENIGN_LABELS,
    PATHOGENIC_LABELS,
)
from features import add_variant_features

USECOLS = [
    "VariationID",
    "Assembly",
    "ClinicalSignificance",
    "GeneSymbol",
    "Chromosome",
    "PositionVCF",
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
        help="Preprocessing mode: dev or balanced-sample.",
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
    # Sample unique, non-conflicting GRCh38 variants
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be greater than zero.")

    rng = random.Random(42)

    reservoirs = {
        0: [],
        1: [],
    }

    counts = {
        0: 0,
        1: 0,
    }

    # Store the SQLite database in a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "clinvar_variants.db"

        # First pass: identify unique variants and conflicting labels
        build_variant_database(db_path)

        # Second stage: sample eligible variants from SQLite
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    VariationID,
                    Assembly,
                    ClinicalSignificance,
                    GeneSymbol,
                    Chromosome,
                    PositionVCF,
                    OriginSimple,
                    ReviewStatus,
                    ReferenceAlleleVCF,
                    AlternateAlleleVCF,
                    target
                FROM variants
                WHERE conflicting = 0
                ORDER BY
                    Chromosome,
                    PositionVCF,
                    ReferenceAlleleVCF,
                    AlternateAlleleVCF
            """)

            columns = [description[0] for description in cursor.description]

            for values in cursor:
                row = dict(zip(columns, values))

                target = row["target"]
                counts[target] += 1

                reservoir = reservoirs[target]

                if len(reservoir) < samples_per_class:
                    reservoir.append(row)

                else:
                    # Reservoir sampling gives each unique variant
                    # an equal probability of selection
                    index = rng.randrange(counts[target])

                    if index < samples_per_class:
                        reservoir[index] = row

    # Ensure enough unique variants exist in both classes
    for target, count in counts.items():
        if count < samples_per_class:
            raise ValueError(
                f"Not enough unique variants for class {target}. "
                f"Required: {samples_per_class:,}, available: {count:,}."
            )

    # Combine both balanced samples
    df = pd.DataFrame(reservoirs[0] + reservoirs[1])

    # Shuffle the final dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nBalanced sampling completed.")
    print(f"Unique benign variants available: {counts[0]:,}")
    print(f"Unique pathogenic variants available: {counts[1]:,}")
    print(f"Selected variants per class: {samples_per_class:,}")

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

def filter_valid_variants(df: pd.DataFrame)-> pd.DataFrame:
    # Keep only GRCh38 variants with valid genomic coordinates
    df = df.copy()

    df = df[df["Assembly"] == "GRCh38"].copy()

    required_columns = [
        "Chromosome",
        "PositionVCF",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
    ]

    df = df.dropna(subset=required_columns)

    # Validate genomic positions
    positions = pd.to_numeric(df["PositionVCF"], errors="coerce")

    df = df.loc[positions > 0].copy()

    # Validate reference and alternate alleles
    invalid_values = {"na", "-", "", "none", "nan"}

    for column in ["ReferenceAlleleVCF", "AlternateAlleleVCF"]:
        df = df[
            ~df[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(invalid_values)
        ]
    
    return df

def build_variant_database(db_path: Path) -> None:
    # Create a temporary database for unique GRCh38 variants
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE variants (
                Chromosome TEXT NOT NULL,
                PositionVCF TEXT NOT NULL,
                ReferenceAlleleVCF TEXT NOT NULL,
                AlternateAlleleVCF TEXT NOT NULL,
                target INTEGER NOT NULL,
                conflicting INTEGER NOT NULL DEFAULT 0,
                VariationID TEXT,
                Assembly TEXT,
                ClinicalSignificance TEXT,
                GeneSymbol TEXT,
                OriginSimple TEXT,
                ReviewStatus TEXT,
                PRIMARY KEY (
                    Chromosome,
                    PositionVCF,
                    ReferenceAlleleVCF,
                    AlternateAlleleVCF
                )
            )
        """)

        chunks = pd.read_csv(
            RAW_CLINVAR_FILE,
            sep="\t",
            usecols=USECOLS,
            dtype=str,
            chunksize=100_000,
        )

        total_records = 0

        for chunk_number, chunk in enumerate(chunks, start=1):
            # Keep valid GRCh38 variants
            chunk = filter_valid_variants(chunk)

            # Map clinical significance labels
            chunk["target"] = chunk["ClinicalSignificance"].apply(map_target)
            chunk = chunk.dropna(subset=["target"]).copy()
            chunk["target"] = chunk["target"].astype(int)

            total_records += len(chunk)

            records = [
                (
                    row["Chromosome"],
                    row["PositionVCF"],
                    row["ReferenceAlleleVCF"],
                    row["AlternateAlleleVCF"],
                    row["target"],
                    row["VariationID"],
                    row["Assembly"],
                    row["ClinicalSignificance"],
                    row["GeneSymbol"],
                    row["OriginSimple"],
                    row["ReviewStatus"],
                )
                for row in chunk.to_dict("records")
            ]

            # Insert new variants and mark conflicting labels
            conn.executemany("""
                INSERT INTO variants (
                    Chromosome,
                    PositionVCF,
                    ReferenceAlleleVCF,
                    AlternateAlleleVCF,
                    target,
                    VariationID,
                    Assembly,
                    ClinicalSignificance,
                    GeneSymbol,
                    OriginSimple,
                    ReviewStatus
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

                ON CONFLICT (
                    Chromosome,
                    PositionVCF,
                    ReferenceAlleleVCF,
                    AlternateAlleleVCF
                )
                DO UPDATE SET
                    conflicting = CASE
                        WHEN variants.target != excluded.target
                        THEN 1
                        ELSE variants.conflicting
                    END
            """, records)

            conn.commit()

            print(
                f"Chunk {chunk_number}: "
                f"eligible GRCh38 records={total_records:,}"
            )

        print("\n--- Database Summary ---")

        total_unique = conn.execute(
            "SELECT COUNT(*) FROM variants"
        ).fetchone()[0]

        conflicting = conn.execute(
            "SELECT COUNT(*) FROM variants WHERE conflicting = 1"
        ).fetchone()[0]

        print(f"Eligible records processed: {total_records:,}")
        print(f"Unique variants: {total_unique:,}")
        print(f"Variants with conflicting labels: {conflicting:,}")

def load_dev_data(nrows: int) -> pd.DataFrame:
    # Load a small subset of ClinVar data for fast development runs
    return pd.read_csv(
        RAW_CLINVAR_FILE,
        sep="\t",
        usecols=USECOLS,
        dtype=str,
        nrows=nrows,
    )

def analyze_grch38_distribution() -> None:
    # Count eligible GRCh38 records before sampling
    counts = {0: 0, 1: 0}

    chunks = pd.read_csv(RAW_CLINVAR_FILE, sep="\t", usecols=USECOLS, dtype=str, chunksize=100_000,)

    for chunk in chunks:
        chunk = filter_valid_variants(chunk)

        chunk["target"] = chunk["ClinicalSignificance"].apply(map_target)
        chunk = chunk.dropna(subset=["target"])

        for target, count in chunk["target"].value_counts().items():
            counts[int(target)] += int(count)
    
    print("\n --- GRCh38 Distribution ---")
    print(f"Benign records: {counts[0]:,}")
    print(f"Pathogenic records: {counts[1]:,}")
    print(f"Total eligible records: {sum(counts.values()):,}")

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

def analyze_duplicates(df: pd.DataFrame) -> None:
    # Identify duplicate variants using valid genomic coordinates
    variant_columns = [
        "Assembly",
        "Chromosome",
        "PositionVCF",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
    ]

    print("\n--- Duplicate Analysis ---")

    valid = df.dropna(subset=variant_columns).copy()

    # Exclude records without valid genomic positions
    positions = pd.to_numeric(
        valid["PositionVCF"],
        errors="coerce"
    )

    valid = valid.loc[positions > 0].copy()

    # Exclude missing or invalid alleles
    invalid_values = {"na", "-", "", "none", "nan"}

    for column in ["ReferenceAlleleVCF", "AlternateAlleleVCF"]:
        valid = valid[
            ~valid[column].astype(str).str.strip().str.lower().isin(invalid_values)
        ]

    # Count duplicate records
    duplicates = valid.duplicated(
        subset=variant_columns,
        keep=False
    )

    # Count unique variants
    unique_variants = valid.drop_duplicates(
        subset=variant_columns
    )

    # Identify conflicting labels
    conflicting = (
        valid.groupby(variant_columns)["target"]
        .nunique()
    )

    conflicting_keys = conflicting[conflicting > 1].index

    print(f"Total records: {len(df):,}")
    print(f"Records with valid genomic identifiers: {len(valid):,}")
    print(f"Records excluded from coordinate analysis: {len(df) - len(valid):,}")
    print(f"Records involved in duplicates: {duplicates.sum():,}")
    print(f"Unique genetic variants: {len(unique_variants):,}")
    print(f"Variants with conflicting labels: {len(conflicting_keys):,}")

    print("\n--- Conflicting Variant Details ---")

    if len(conflicting_keys) == 0:
        print("No conflicting variants found.")
    else:
        conflicting_records = (
            valid.set_index(variant_columns)
            .loc[conflicting_keys]
            .reset_index()
        )

        columns_to_display = [
            "VariationID",
            "Assembly",
            "Chromosome",
            "PositionVCF",
            "ReferenceAlleleVCF",
            "AlternateAlleleVCF",
            "ClinicalSignificance",
            "ReviewStatus",
            "target",
        ]

        print(
            conflicting_records[columns_to_display]
            .head(20)
            .to_string(index=False)
        )

    print("--- End Duplicate Analysis ---\n")

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
        df = filter_valid_variants(df)

    elif args.mode == "balanced-sample":
        df = load_balanced_sample(
            samples_per_class=args.samples_per_class
        )

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    df = preprocess_dataframe(df)

    analyze_duplicates(df)

    save_processed_data(df)


if __name__ == "__main__":
    main()
