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

MODEL_COMPARISON_FILE = METRICS_DIR / "model_comparison.csv"


MODEL_NAMES = {
    "dummy": "Dummy Classifier",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


class DashboardPage(ctk.CTkFrame):
    """Display dataset statistics and model performance."""

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
        self.create_dataset_overview()
        self.create_model_overview()
        self.create_model_comparison()
        self.create_performance_charts()

    def create_header(self):
        """Create the dashboard title and description."""

        header = ctk.CTkFrame(
            self.scroll_frame,
            fg_color="transparent",
        )

        header.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=30,
            pady=(30, 20),
        )

        title = ctk.CTkLabel(
            header,
            text="Dashboard",
            font=ctk.CTkFont(
                size=28,
                weight="bold",
            ),
            text_color=TEXT_PRIMARY,
        )

        title.pack(anchor="w")

        subtitle = ctk.CTkLabel(
            header,
            text="Overview of genetic variant classification results.",
            font=ctk.CTkFont(size=14),
            text_color=TEXT_SECONDARY,
        )

        subtitle.pack(anchor="w", pady=(5, 0))

    def create_section_title(self, parent, text):
        """Create a reusable section heading."""

        title = ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(
                size=19,
                weight="bold",
            ),
            text_color=TEXT_PRIMARY,
        )

        title.pack(
            anchor="w",
            pady=(0, 15),
        )

    def create_stat_card(self, parent, title, value, description):
        """Create a card displaying a dataset statistic."""

        card = ctk.CTkFrame(
            parent,
            fg_color=CARD,
            corner_radius=12,
            border_width=1,
            border_color=BORDER,
        )

        title_label = ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=13),
            text_color=TEXT_SECONDARY,
        )

        title_label.pack(
            anchor="w",
            padx=20,
            pady=(18, 8),
        )

        value_label = ctk.CTkLabel(
            card,
            text=value,
            font=ctk.CTkFont(
                size=27,
                weight="bold",
            ),
            text_color=ACCENT,
        )

        value_label.pack(
            anchor="w",
            padx=20,
        )

        description_label = ctk.CTkLabel(
            card,
            text=description,
            font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY,
        )

        description_label.pack(
            anchor="w",
            padx=20,
            pady=(8, 18),
        )

        return card

    def create_dataset_overview(self):
        """Display information about the balanced ClinVar dataset."""

        section = ctk.CTkFrame(
            self.scroll_frame,
            fg_color="transparent",
        )

        section.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=30,
            pady=(0, 25),
        )

        self.create_section_title(
            section,
            "Dataset Overview",
        )

        cards = ctk.CTkFrame(
            section,
            fg_color="transparent",
        )

        cards.pack(fill="x")

        for column in range(3):
            cards.grid_columnconfigure(column, weight=1)

        statistics = [
            (
                "Total Variants",
                "100,000",
                "Balanced ClinVar dataset",
            ),
            (
                "Benign Variants",
                "50,000",
                "50% of the dataset",
            ),
            (
                "Pathogenic Variants",
                "50,000",
                "50% of the dataset",
            ),
        ]

        for column, (title, value, description) in enumerate(statistics):
            card = self.create_stat_card(
                cards,
                title,
                value,
                description,
            )

            card.grid(
                row=0,
                column=column,
                sticky="nsew",
                padx=(0, 10) if column < 2 else 0,
            )

    def create_model_overview(self):
        """Display the best-performing model."""

        section = ctk.CTkFrame(
            self.scroll_frame,
            fg_color="transparent",
        )

        section.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=30,
            pady=(0, 25),
        )

        self.create_section_title(
            section,
            "Best Performing Model",
        )

        card = ctk.CTkFrame(
            section,
            fg_color=CARD,
            corner_radius=12,
            border_width=1,
            border_color=BORDER,
        )

        card.pack(fill="x")

        if not MODEL_COMPARISON_FILE.exists():
            self.show_message(
                card,
                "Model comparison data is unavailable.",
            )
            return

        df = pd.read_csv(MODEL_COMPARISON_FILE)

        # Select the model with the highest pathogenic F1-score.
        best_model = df.loc[df["f1_pathogenic"].idxmax()]

        model_name = MODEL_NAMES.get(
            best_model["model"],
            best_model["model"],
        )

        name_label = ctk.CTkLabel(
            card,
            text=model_name,
            font=ctk.CTkFont(
                size=22,
                weight="bold",
            ),
            text_color=ACCENT,
        )

        name_label.pack(
            anchor="w",
            padx=20,
            pady=(20, 5),
        )

        description = ctk.CTkLabel(
            card,
            text="Highest pathogenic F1-score among the evaluated models.",
            font=ctk.CTkFont(size=13),
            text_color=TEXT_SECONDARY,
        )

        description.pack(
            anchor="w",
            padx=20,
        )

        metrics_frame = ctk.CTkFrame(
            card,
            fg_color="transparent",
        )

        metrics_frame.pack(
            fill="x",
            padx=20,
            pady=20,
        )

        metrics = [
            ("Accuracy", best_model["accuracy"]),
            ("F1-score", best_model["f1_pathogenic"]),
            ("ROC-AUC", best_model["roc_auc"]),
            ("PR-AUC", best_model["pr_auc"]),
        ]

        for column in range(4):
            metrics_frame.grid_columnconfigure(column, weight=1)

        for column, (name, value) in enumerate(metrics):
            metric_frame = ctk.CTkFrame(
                metrics_frame,
                fg_color=BACKGROUND,
                corner_radius=8,
            )

            metric_frame.grid(
                row=0,
                column=column,
                sticky="nsew",
                padx=(0, 10) if column < 3 else 0,
            )

            ctk.CTkLabel(
                metric_frame,
                text=name,
                text_color=TEXT_SECONDARY,
                font=ctk.CTkFont(size=12),
            ).pack(pady=(12, 5))

            ctk.CTkLabel(
                metric_frame,
                text=f"{value:.4f}",
                text_color=TEXT_PRIMARY,
                font=ctk.CTkFont(
                    size=18,
                    weight="bold",
                ),
            ).pack(pady=(0, 12))

    def create_model_comparison(self):
        """Display the model comparison table."""

        section = ctk.CTkFrame(
            self.scroll_frame,
            fg_color="transparent",
        )

        section.grid(
            row=3,
            column=0,
            sticky="ew",
            padx=30,
            pady=(0, 25),
        )

        self.create_section_title(
            section,
            "Model Comparison",
        )

        table = ctk.CTkFrame(
            section,
            fg_color=CARD,
            corner_radius=12,
        )

        table.pack(fill="x")

        if not MODEL_COMPARISON_FILE.exists():
            self.show_message(
                table,
                "Model comparison data is unavailable.",
            )
            return

        df = pd.read_csv(MODEL_COMPARISON_FILE)

        columns = [
            ("Model", "model"),
            ("Accuracy", "accuracy"),
            ("Precision", "precision_pathogenic"),
            ("Recall", "recall_pathogenic"),
            ("F1-score", "f1_pathogenic"),
            ("ROC-AUC", "roc_auc"),
            ("PR-AUC", "pr_auc"),
        ]

        for column in range(len(columns)):
            table.grid_columnconfigure(column, weight=1)

        for column, (label, _) in enumerate(columns):
            header = ctk.CTkLabel(
                table,
                text=label,
                font=ctk.CTkFont(
                    size=12,
                    weight="bold",
                ),
                text_color=ACCENT,
            )

            header.grid(
                row=0,
                column=column,
                padx=8,
                pady=15,
            )

        for row_index, (_, model) in enumerate(df.iterrows(), start=1):
            for column, (_, key) in enumerate(columns):

                if key == "model":
                    value = MODEL_NAMES.get(
                        model[key],
                        model[key],
                    )
                else:
                    value = f"{model[key]:.4f}"

                label = ctk.CTkLabel(
                    table,
                    text=value,
                    font=ctk.CTkFont(size=12),
                    text_color=TEXT_PRIMARY,
                )

                label.grid(
                    row=row_index,
                    column=column,
                    padx=8,
                    pady=10,
                )

    def create_performance_charts(self):
        """Display ROC and Precision-Recall comparison charts."""

        section = ctk.CTkFrame(
            self.scroll_frame,
            fg_color="transparent",
        )

        section.grid(
            row=4,
            column=0,
            sticky="ew",
            padx=30,
            pady=(0, 30),
        )

        self.create_section_title(
            section,
            "Performance Charts",
        )

        charts = [
            (
                "ROC Curve Comparison",
                "roc_curve_comparison.png",
            ),
            (
                "Precision-Recall Comparison",
                "precision_recall_comparison.png",
            ),
        ]

        for title, filename in charts:
            card = ctk.CTkFrame(
                section,
                fg_color=CARD,
                corner_radius=12,
            )

            card.pack(
                fill="x",
                pady=(0, 15),
            )

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
                pady=(20, 10),
            )

            image_path = FIGURES_DIR / filename

            if not image_path.exists():
                self.show_message(
                    card,
                    f"Chart not found: {filename}",
                )
                continue

            with Image.open(image_path) as source:
                image = source.copy()

            image.thumbnail((750, 420))

            chart_image = ctk.CTkImage(
                light_image=image,
                dark_image=image,
                size=image.size,
            )

            self.images.append(chart_image)

            image_label = ctk.CTkLabel(
                card,
                image=chart_image,
                text="",
            )

            image_label.pack(
                padx=20,
                pady=(0, 20),
            )

    def show_message(self, parent, message):
        """Display a message when data is unavailable."""

        label = ctk.CTkLabel(
            parent,
            text=message,
            text_color=TEXT_SECONDARY,
            font=ctk.CTkFont(size=13),
        )

        label.pack(padx=20, pady=20)