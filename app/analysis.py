from pathlib import Path

import customtkinter as ctk
import pandas as pd

from PIL import Image

from .styles import (
    BACKGROUND,
    CARD,
    ACCENT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    BORDER,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


METRIC_COLUMNS = [
    ("Accuracy", "accuracy"),
    ("Precision", "precision_pathogenic"),
    ("Recall", "recall_pathogenic"),
    ("F1-score", "f1_pathogenic"),
    ("ROC-AUC", "roc_auc"),
    ("PR-AUC", "pr_auc"),
]


FEATURE_SET_NAMES = {
    "structural_only": "Structural Only",
    "structural_plus_genomic": "Structural + Genomic",
    "structural_plus_origin": "Structural + Origin",
    "full_features": "Full Features",
}


class AnalysisPage(ctk.CTkFrame):
    # Display feature importance and experimental results.

    def __init__(self, master):
        super().__init__(
            master,
            fg_color=BACKGROUND,
            corner_radius=0,
        )

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.images = []

        self.scroll_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=BACKGROUND,
            corner_radius=0,
        )

        self.scroll_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        self.scroll_frame.grid_columnconfigure(0, weight=1)

        self.create_header()
        self.create_feature_importance()
        self.create_ablation_study()
        self.create_gene_split_evaluation()

    def create_header(self):
        # Create the page title and description.

        header = ctk.CTkFrame(self.scroll_frame,fg_color="transparent")
        header.grid(row=0,column=0,sticky="ew",padx=30,pady=(30, 20))

        ctk.CTkLabel(
            header,
            text="Model Analysis",
            font=ctk.CTkFont(
                size=28,
                weight="bold",
            ),
            text_color=TEXT_PRIMARY,
        ).pack(anchor="w")

        ctk.CTkLabel(
            header,
            text="Explore model interpretation and experimental results.",
            font=ctk.CTkFont(size=14),
            text_color=TEXT_SECONDARY,
        ).pack(
            anchor="w",
            pady=(5, 0),
        )

    def create_section(self, row, title, description):
        # Create a reusable analysis section.
        section = ctk.CTkFrame(self.scroll_frame,fg_color="transparent")
        section.grid(row=row,column=0,sticky="ew",padx=30,pady=(0, 30))

        ctk.CTkLabel(
            section,
            text=title,
            font=ctk.CTkFont(
                size=19,
                weight="bold",
            ),
            text_color=TEXT_PRIMARY,
        ).pack(
            anchor="w",
            pady=(0, 5),
        )

        ctk.CTkLabel(
            section,
            text=description,
            font=ctk.CTkFont(size=13),
            text_color=TEXT_SECONDARY,
            wraplength=750,
            justify="left",
        ).pack(
            anchor="w",
            pady=(0, 15),
        )

        return section

    def create_card(self, parent):
        # Create a reusable content card.

        card = ctk.CTkFrame(parent,fg_color=CARD,corner_radius=12,border_width=1,border_color=BORDER)
        card.pack(fill="x",pady=(0, 15),)

        return card

    def show_message(self, parent, message):
        # Display a message when data is unavailable.

        ctk.CTkLabel(
            parent,
            text=message,
            text_color=TEXT_SECONDARY,
            font=ctk.CTkFont(size=13),
        ).pack(
            padx=20,
            pady=20,
        )

    def add_image(self, parent, filename):
        # Load and display an existing analysis figure.

        image_path = FIGURES_DIR / filename

        if not image_path.exists():
            self.show_message(
                parent,
                f"Figure not found: {filename}",
            )
            return

        try:
            with Image.open(image_path) as source:
                image = source.copy()

            image.thumbnail((750, 430))

            chart_image = ctk.CTkImage(
                light_image=image,
                dark_image=image,
                size=image.size,
            )

            self.images.append(chart_image)

            ctk.CTkLabel(
                parent,
                image=chart_image,
                text="",
            ).pack(
                padx=20,
                pady=(0, 20),
            )

        except (OSError, ValueError) as error:
            self.show_message(
                parent,
                f"Unable to load figure: {error}",
            )

    def create_table(self, parent, columns, rows):
        # Create a table using CustomTkinter labels.

        table = ctk.CTkFrame(parent,fg_color="transparent")

        table.pack(fill="x",padx=15,pady=15,)

        for column in range(len(columns)):
            table.grid_columnconfigure(column, weight=1)

        # Table header
        for column, heading in enumerate(columns):
            ctk.CTkLabel(
                table,
                text=heading,
                font=ctk.CTkFont(
                    size=12,
                    weight="bold",
                ),
                text_color=ACCENT,
            ).grid(
                row=0,
                column=column,
                padx=8,
                pady=(10, 15),
                sticky="ew",
            )

        # Table rows
        for row_index, values in enumerate(rows, start=1):

            for column, value in enumerate(values):
                ctk.CTkLabel(
                    table,
                    text=str(value),
                    font=ctk.CTkFont(size=12),
                    text_color=TEXT_PRIMARY,
                ).grid(
                    row=row_index,
                    column=column,
                    padx=8,
                    pady=10,
                    sticky="ew",
                )

    def create_feature_importance(self):
        # Display model interpretation figures.
        section = self.create_section(
            row=1,
            title="Feature Importance",
            description=(
                "Explore the features influencing the predictions "
                "of Logistic Regression and Random Forest."
            ),
        )

        figures = [
            (
                "Logistic Regression Coefficients",
                "logistic_regression_coefficients.png",
            ),
            (
                "Random Forest Feature Importance",
                "random_forest_feature_importance.png",
            ),
        ]

        for title, filename in figures:
            card = self.create_card(section)

            ctk.CTkLabel(
                card,
                text=title,
                font=ctk.CTkFont(
                    size=16,
                    weight="bold",
                ),
                text_color=TEXT_PRIMARY,
            ).pack(
                anchor="w",
                padx=20,
                pady=(20, 15),
            )

            self.add_image(card, filename)

        explanation = self.create_card(section)

        ctk.CTkLabel(
            explanation,
            text=(
                "Logistic Regression coefficients indicate the direction "
                "of each feature's association with the model prediction.\n\n"
                "Random Forest feature importance indicates how much "
                "features contribute to impurity reduction across trees.\n\n"
                "These values describe model behavior and should not "
                "be interpreted as causal biological effects."
            ),
            font=ctk.CTkFont(size=13),
            text_color=TEXT_SECONDARY,
            wraplength=720,
            justify="left",
        ).pack(
            anchor="w",
            padx=20,
            pady=20,
        )

    def create_ablation_study(self):
        # Display the feature-group ablation study.
        section = self.create_section(
            row=2,
            title="Ablation Study",
            description=(
                "Compare Logistic Regression performance using "
                "different groups of input features."
            ),
        )

        card = self.create_card(section)

        file_path = METRICS_DIR / "ablation_study.csv"

        if not file_path.exists():
            self.show_message(
                card,
                "Ablation study results are unavailable.",
            )
            return

        try:
            df = pd.read_csv(file_path)

            columns = ["Feature Set"] + [
                name for name, _ in METRIC_COLUMNS
            ]

            rows = []

            for _, record in df.iterrows():
                feature_set = FEATURE_SET_NAMES.get(
                    record["feature_set"],
                    record["feature_set"],
                )

                values = [feature_set]

                for _, metric in METRIC_COLUMNS:
                    values.append(
                        f"{record[metric]:.4f}"
                    )

                rows.append(values)

            self.create_table(
                card,
                columns,
                rows,
            )

            # Identify the configuration with the highest F1-score.
            best_row = df.loc[df["f1_pathogenic"].idxmax()]

            best_name = FEATURE_SET_NAMES.get(
                best_row["feature_set"],
                best_row["feature_set"],
            )

            ctk.CTkLabel(
                card,
                text=(
                    f"Highest pathogenic F1-score: {best_name} "
                    f"({best_row['f1_pathogenic']:.4f})"
                ),
                font=ctk.CTkFont(
                    size=13,
                    weight="bold",
                ),
                text_color=ACCENT,
            ).pack(
                anchor="w",
                padx=20,
                pady=(0, 20),
            )

        except (OSError, ValueError, KeyError) as error:
            self.show_message(
                card,
                f"Unable to read ablation results: {error}",
            )

    def create_gene_split_evaluation(self):
        # Display random and gene-based split evaluation results.
        section = self.create_section(
            row=3,
            title="Gene-Based Evaluation",
            description=(
                "Evaluate model performance when genes in the "
                "test set are excluded from the training set."
            ),
        )

        comparison_file = (METRICS_DIR / "gene_split_comparison.csv")

        summary_file = (METRICS_DIR / "gene_split_summary.csv")

        card = self.create_card(section)

        if not comparison_file.exists() or not summary_file.exists():
            self.show_message(card,"Gene-based evaluation results are unavailable.")
            return

        try:
            comparison = pd.read_csv(comparison_file)

            summary = pd.read_csv(summary_file,index_col=0)

            # Random split baseline
            random_row = comparison.loc[
                comparison["split"] == "random_split"
            ].iloc[0]

            # Mean and standard deviation across gene-based splits
            gene_mean = summary.loc["mean"]
            gene_std = summary.loc["std"]

            columns = ["Evaluation"] + [name for name, _ in METRIC_COLUMNS]

            rows = []

            rows.append(
                ["Random Split"] + [
                    f"{random_row[metric]:.4f}"
                    for _, metric in METRIC_COLUMNS
                ]
            )

            rows.append(
                ["Gene-Based Mean"] + [
                    f"{gene_mean[metric]:.4f}"
                    for _, metric in METRIC_COLUMNS
                ]
            )

            rows.append(
                ["Gene-Based Std"] + [
                    f"{gene_std[metric]:.4f}"
                    for _, metric in METRIC_COLUMNS
                ]
            )

            self.create_table(card,columns,rows)

            ctk.CTkLabel(
                card,
                text=(
                    "Gene-based results are summarized across five "
                    "repeated group-based splits."
                ),
                font=ctk.CTkFont(size=12),
                text_color=TEXT_SECONDARY,
            ).pack(
                anchor="w",
                padx=20,
                pady=(0, 20),
            )

            # Individual gene-based split results
            details_card = self.create_card(section)

            ctk.CTkLabel(
                details_card,
                text="Individual Gene-Based Splits",
                font=ctk.CTkFont(
                    size=16,
                    weight="bold",
                ),
                text_color=TEXT_PRIMARY,
            ).pack(
                anchor="w",
                padx=20,
                pady=(20, 10),
            )

            gene_splits = comparison[
                comparison["split"].str.startswith(
                    "gene_based_split_"
                )
            ]

            detail_columns = [
                "Split",
                "Test Samples",
                "Accuracy",
                "F1-score",
                "ROC-AUC",
            ]

            detail_rows = []

            for _, record in gene_splits.iterrows():
                detail_rows.append(
                    [
                        f"Split {int(record['iteration'])}",
                        f"{int(record['test_samples']):,}",
                        f"{record['accuracy']:.4f}",
                        f"{record['f1_pathogenic']:.4f}",
                        f"{record['roc_auc']:.4f}",
                    ]
                )

            self.create_table(details_card,detail_columns,detail_rows)

            explanation = self.create_card(section)

            ctk.CTkLabel(
                explanation,
                text=(
                    "The gene-based evaluation measures performance "
                    "on held-out gene groups.\n\n"
                    "The model excludes GeneSymbol as a predictor "
                    "during this experiment.\n\n"
                    "Random and gene-based splits use different test "
                    "sets and may have different class distributions. "
                    "Their metrics are therefore descriptive rather "
                    "than directly controlled comparisons."
                ),
                font=ctk.CTkFont(size=13),
                text_color=TEXT_SECONDARY,
                wraplength=720,
                justify="left",
            ).pack(
                anchor="w",
                padx=20,
                pady=20,
            )

        except (OSError, ValueError, KeyError, IndexError) as error:
            self.show_message(card,f"Unable to read gene-based results: {error}")