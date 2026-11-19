import customtkinter as ctk

from .styles import (
    configure_theme,
    BACKGROUND,
    SIDEBAR,
    ACCENT,
    ACCENT_HOVER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

from .dashboard import DashboardPage
from .prediction import PredictionPage
from .analysis import AnalysisPage


class GeneticVariantApp(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.title("Genetic Variant Classification")
        self.geometry("1200x750")
        self.minsize(1000, 650)
        self.configure(fg_color=BACKGROUND)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.create_sidebar()
        self.create_pages()

        self.show_page("dashboard")

    def create_sidebar(self):
        """Create the application navigation sidebar."""

        self.sidebar = ctk.CTkFrame(
            self,
            width=250,
            corner_radius=0,
            fg_color=SIDEBAR,
        )
        self.sidebar.grid(
            row=0,
            column=0,
            sticky="nsew",
        )
        self.sidebar.grid_propagate(False)

        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(5, weight=1)

        logo = ctk.CTkLabel(
            self.sidebar,
            text="GENETIC\nVARIANT CLASSIFIER",
            font=ctk.CTkFont(size=19, weight="bold"),
            text_color=TEXT_PRIMARY,
            justify="left",
        )
        logo.grid(
            row=0,
            column=0,
            sticky="w",
            padx=25,
            pady=(35, 35),
        )

        self.buttons = {}

        navigation = [
            ("dashboard", "Dashboard"),
            ("prediction", "Variant Prediction"),
            ("analysis", "Model Analysis"),
        ]

        for index, (page_id, label) in enumerate(navigation, start=1):
            button = ctk.CTkButton(
                self.sidebar,
                text=label,
                height=45,
                corner_radius=8,
                fg_color="transparent",
                hover_color=ACCENT_HOVER,
                text_color=TEXT_PRIMARY,
                anchor="w",
                font=ctk.CTkFont(size=14),
                command=lambda page=page_id: self.show_page(page),
            )

            button.grid(
                row=index,
                column=0,
                sticky="ew",
                padx=15,
                pady=5,
            )

            self.buttons[page_id] = button

        footer = ctk.CTkLabel(
            self.sidebar,
            text="ClinVar ML Project",
            text_color=TEXT_SECONDARY,
            font=ctk.CTkFont(size=12),
        )
        footer.grid(
            row=6,
            column=0,
            pady=25,
        )

    def create_pages(self):
        """Initialize application pages."""

        self.content = ctk.CTkFrame(
            self,
            fg_color=BACKGROUND,
            corner_radius=0,
        )
        self.content.grid(
            row=0,
            column=1,
            sticky="nsew",
        )

        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        self.pages = {
            "dashboard": DashboardPage(self.content),
            "prediction": PredictionPage(self.content),
            "analysis": AnalysisPage(self.content),
        }

        for page in self.pages.values():
            page.grid(
                row=0,
                column=0,
                sticky="nsew",
            )

    def show_page(self, page_name):
        """Display the selected page and update navigation buttons."""

        self.pages[page_name].tkraise()

        for name, button in self.buttons.items():
            button.configure(
                fg_color=ACCENT if name == page_name else "transparent"
            )


if __name__ == "__main__":
    configure_theme()

    app = GeneticVariantApp()
    app.mainloop()