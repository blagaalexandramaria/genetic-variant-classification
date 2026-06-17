from pathlib import Path

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

RAW_CLINVAR_FILE = RAW_DATA_DIR / "variant_summary.txt"
PROCESSED_CLINVAR_FILE = PROCESSED_DATA_DIR / "clinvar_features.csv"

TARGET_COLUMN = "target"

BENIGN_LABELS = {
    "Benign",
    "Likely benign",
}

PATHOGENIC_LABELS = {
    "Pathogenic",
    "Likely pathogenic",
}

CATEGORICAL_FEATURES = [
    "Chromosome",
    "GeneSymbol",
    "mutation_type",
    "OriginSimple",
    "ReviewStatus",
]

NUMERIC_FEATURES = [
    "ref_length",
    "alt_length",
    "length_difference",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
