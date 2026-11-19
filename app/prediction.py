import customtkinter as ctk
import joblib
import pandas as pd

from .styles import (
    BACKGROUND,
    CARD,
    ACCENT,
    ACCENT_HOVER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    BORDER,
)

from src.config import MODELS_DIR, FEATURE_COLUMNS
from src.features import add_variant_features


MODEL_FILES = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Random Forest": "random_forest_model.pkl",
}


class PredictionPage(ctk.CTkFrame):
    """Page for predicting genetic variant classifications."""

    def __init__(self, master):
        super().__init__(
            master,
            fg_color=BACKGROUND,
            corner_radius=0,
        )

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.models = {}

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
        self.create_input_form()
        self.create_result_section()

    def create_header(self):
        """Create the page title and description."""

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

        ctk.CTkLabel(
            header,
            text="Variant Prediction",
            font=ctk.CTkFont(
                size=28,
                weight="bold",
            ),
            text_color=TEXT_PRIMARY,
        ).pack(anchor="w")

        ctk.CTkLabel(
            header,
            text="Classify genetic variants using trained machine learning models.",
            font=ctk.CTkFont(size=14),
            text_color=TEXT_SECONDARY,
        ).pack(
            anchor="w",
            pady=(5, 0),
        )

    def create_input_field(self, parent, label, placeholder):
        """Create a labeled text input."""

        container = ctk.CTkFrame(
            parent,
            fg_color="transparent",
        )

        ctk.CTkLabel(
            container,
            text=label,
            font=ctk.CTkFont(size=13),
            text_color=TEXT_PRIMARY,
        ).pack(
            anchor="w",
            pady=(0, 8),
        )

        entry = ctk.CTkEntry(
            container,
            height=42,
            placeholder_text=placeholder,
            fg_color=BACKGROUND,
            border_color=BORDER,
            text_color=TEXT_PRIMARY,
        )

        entry.pack(fill="x")

        return container, entry

    def create_dropdown(self, parent, label, values):
        """Create a labeled dropdown menu."""

        container = ctk.CTkFrame(
            parent,
            fg_color="transparent",
        )

        ctk.CTkLabel(
            container,
            text=label,
            font=ctk.CTkFont(size=13),
            text_color=TEXT_PRIMARY,
        ).pack(
            anchor="w",
            pady=(0, 8),
        )

        dropdown = ctk.CTkOptionMenu(
            container,
            values=values,
            height=42,
            fg_color=BACKGROUND,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            text_color=TEXT_PRIMARY,
            dropdown_fg_color=CARD,
        )

        dropdown.pack(fill="x")

        return container, dropdown

    def create_input_form(self):
        """Create the variant information form."""

        section = ctk.CTkFrame(
            self.scroll_frame,
            fg_color=CARD,
            corner_radius=12,
            border_width=1,
            border_color=BORDER,
        )

        section.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=30,
            pady=(0, 25),
        )

        ctk.CTkLabel(
            section,
            text="Variant Information",
            font=ctk.CTkFont(
                size=19,
                weight="bold",
            ),
            text_color=TEXT_PRIMARY,
        ).pack(
            anchor="w",
            padx=25,
            pady=(25, 20),
        )

        form = ctk.CTkFrame(
            section,
            fg_color="transparent",
        )

        form.pack(
            fill="x",
            padx=25,
            pady=(0, 20),
        )

        form.grid_columnconfigure(0, weight=1)
        form.grid_columnconfigure(1, weight=1)

        # Gene symbol
        gene_container, self.gene_entry = self.create_input_field(
            form,
            "Gene Symbol",
            "e.g. BRCA1",
        )

        gene_container.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 10),
            pady=(0, 20),
        )

        # Chromosome
        chromosome_values = [
            str(i) for i in range(1, 23)
        ] + ["X", "Y", "MT"]

        chromosome_container, self.chromosome_dropdown = (
            self.create_dropdown(
                form,
                "Chromosome",
                chromosome_values,
            )
        )

        chromosome_container.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(10, 0),
            pady=(0, 20),
        )

        # Reference allele
        ref_container, self.ref_entry = self.create_input_field(
            form,
            "Reference Allele",
            "e.g. A",
        )

        ref_container.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=(0, 10),
            pady=(0, 20),
        )

        # Alternate allele
        alt_container, self.alt_entry = self.create_input_field(
            form,
            "Alternate Allele",
            "e.g. G",
        )

        alt_container.grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(10, 0),
            pady=(0, 20),
        )

        # Origin
        origin_container, self.origin_dropdown = self.create_dropdown(
            form,
            "Origin",
            [
                "germline",
                "somatic",
                "unknown",
            ],
        )

        origin_container.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=(0, 10),
            pady=(0, 20),
        )

        # Model selection
        model_container, self.model_dropdown = self.create_dropdown(
            form,
            "Classification Model",
            list(MODEL_FILES.keys()),
        )

        model_container.grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(10, 0),
            pady=(0, 20),
        )

        self.model_dropdown.set("Logistic Regression")
        self.origin_dropdown.set("germline")
        self.chromosome_dropdown.set("17")

        # Action buttons
        buttons = ctk.CTkFrame(
            section,
            fg_color="transparent",
        )

        buttons.pack(
            fill="x",
            padx=25,
            pady=(0, 25),
        )

        self.predict_button = ctk.CTkButton(
            buttons,
            text="Predict Classification",
            height=45,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            font=ctk.CTkFont(
                size=14,
                weight="bold",
            ),
            command=self.predict_variant,
        )

        self.predict_button.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 10),
        )

        self.clear_button = ctk.CTkButton(
            buttons,
            text="Clear",
            height=45,
            width=120,
            fg_color=BACKGROUND,
            hover_color=BORDER,
            text_color=TEXT_PRIMARY,
            command=self.clear_form,
        )

        self.clear_button.pack(side="right")

    def create_result_section(self):
        """Create the prediction results panel."""

        self.result_card = ctk.CTkFrame(
            self.scroll_frame,
            fg_color=CARD,
            corner_radius=12,
            border_width=1,
            border_color=BORDER,
        )

        self.result_card.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=30,
            pady=(0, 30),
        )

        ctk.CTkLabel(
            self.result_card,
            text="Prediction Results",
            font=ctk.CTkFont(
                size=19,
                weight="bold",
            ),
            text_color=TEXT_PRIMARY,
        ).pack(
            anchor="w",
            padx=25,
            pady=(25, 15),
        )

        self.prediction_label = ctk.CTkLabel(
            self.result_card,
            text="No prediction available.",
            font=ctk.CTkFont(
                size=21,
                weight="bold",
            ),
            text_color=TEXT_SECONDARY,
        )

        self.prediction_label.pack(
            anchor="w",
            padx=25,
            pady=(0, 10),
        )

        self.probability_label = ctk.CTkLabel(
            self.result_card,
            text="",
            font=ctk.CTkFont(size=14),
            text_color=TEXT_PRIMARY,
        )

        self.probability_label.pack(
            anchor="w",
            padx=25,
        )

        self.details_label = ctk.CTkLabel(
            self.result_card,
            text="",
            font=ctk.CTkFont(size=13),
            text_color=TEXT_SECONDARY,
            justify="left",
        )

        self.details_label.pack(
            anchor="w",
            padx=25,
            pady=(15, 20),
        )

        disclaimer = ctk.CTkLabel(
            self.result_card,
            text=(
                "Research use only. Predictions are experimental "
                "and must not be used for clinical diagnosis."
            ),
            font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY,
            wraplength=650,
            justify="left",
        )

        disclaimer.pack(
            anchor="w",
            padx=25,
            pady=(0, 25),
        )

    def load_model(self, model_name):
        """Load a trained model and cache it for future predictions."""

        if model_name in self.models:
            return self.models[model_name]

        model_path = MODELS_DIR / MODEL_FILES[model_name]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        model = joblib.load(model_path)

        self.models[model_name] = model

        return model

    def predict_variant(self):
        """Validate user input and predict variant classification."""

        gene = self.gene_entry.get().strip().upper()
        chromosome = self.chromosome_dropdown.get()
        ref = self.ref_entry.get().strip().upper()
        alt = self.alt_entry.get().strip().upper()
        origin = self.origin_dropdown.get()
        model_name = self.model_dropdown.get()

        # Validate required fields.
        if not gene or not ref or not alt:
            self.show_error(
                "Please complete all required fields."
            )
            return

        # Validate DNA alleles.
        valid_bases = {"A", "C", "G", "T"}

        if not set(ref).issubset(valid_bases):
            self.show_error(
                "Reference allele must contain only A, C, G or T."
            )
            return

        if not set(alt).issubset(valid_bases):
            self.show_error(
                "Alternate allele must contain only A, C, G or T."
            )
            return

        if ref == alt:
            self.show_error(
                "Reference and alternate alleles must be different."
            )
            return

        try:
            # Create the input dataframe.
            variant = pd.DataFrame(
                [
                    {
                        "GeneSymbol": gene,
                        "Chromosome": chromosome,
                        "ReferenceAlleleVCF": ref,
                        "AlternateAlleleVCF": alt,
                        "OriginSimple": origin,
                    }
                ]
            )

            # Apply the existing feature engineering pipeline.
            variant = add_variant_features(variant)

            # Select the features used during training.
            X = variant[FEATURE_COLUMNS]

            # Load the selected trained model.
            model = self.load_model(model_name)

            # Generate the prediction.
            prediction = int(model.predict(X)[0])

            # Get the model score for the pathogenic class.
            probabilities = model.predict_proba(X)[0]

            class_index = list(model.classes_).index(1)
            pathogenic_score = float(probabilities[class_index])

            classification = (
                "Pathogenic" if prediction == 1 else "Benign"
            )

            self.display_prediction(
                classification,
                pathogenic_score,
                variant.iloc[0],
                model_name,
            )

        except Exception as error:
            self.show_error(
                f"Prediction failed: {error}"
            )

    def display_prediction(
        self,
        classification,
        pathogenic_score,
        variant,
        model_name,
    ):
        """Display the predicted class and variant characteristics."""

        self.prediction_label.configure(
            text=f"Predicted Classification: {classification}",
            text_color=ACCENT,
        )

        self.probability_label.configure(
            text=(
                "Model score for pathogenic class: "
                f"{pathogenic_score * 100:.2f}%"
            )
        )

        details = (
            f"Model: {model_name}\n"
            f"Mutation Type: {variant['mutation_type']}\n"
            f"Reference Length: {variant['ref_length']}\n"
            f"Alternate Length: {variant['alt_length']}\n"
            f"Length Difference: {variant['length_difference']}"
        )

        self.details_label.configure(text=details)

    def show_error(self, message):
        """Display an error message in the results panel."""

        self.prediction_label.configure(
            text="Prediction Error",
            text_color="#E57373",
        )

        self.probability_label.configure(text="")

        self.details_label.configure(text=message)

    def clear_form(self):
        """Reset the input fields and prediction results."""

        self.gene_entry.delete(0, "end")
        self.ref_entry.delete(0, "end")
        self.alt_entry.delete(0, "end")

        self.chromosome_dropdown.set("17")
        self.origin_dropdown.set("germline")
        self.model_dropdown.set("Logistic Regression")

        self.prediction_label.configure(
            text="No prediction available.",
            text_color=TEXT_SECONDARY,
        )

        self.probability_label.configure(text="")
        self.details_label.configure(text="")