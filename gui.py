# =============================================================================
# gui.py
# Author  : Tayyab Jadoon (Roll No. 050)
# Purpose : Desktop GUI application for the Email Spam Classifier.
#           Provides a user-friendly interface to paste any email text,
#           trigger classification, and display the result with color-coded
#           feedback and a confidence percentage.
# Dependency: predict.py must be in the same directory and model files
#             must exist in the model/ folder.
# =============================================================================

import tkinter as tk
from tkinter import font, messagebox

from predict import predict_spam

# ---------------------------------------------------------------------------
# Color Palette — Catppuccin Mocha dark theme
# ---------------------------------------------------------------------------
BG      = "#1e1e2e"   # main background
SURFACE = "#313244"   # elevated surface (text box)
TEXT    = "#cdd6f4"   # primary text
SUBTEXT = "#a6adc8"   # secondary / placeholder text
GREEN   = "#a6e3a1"   # safe / not spam
RED     = "#f38ba8"   # spam / danger
BLUE    = "#89b4fa"   # accent / button
YELLOW  = "#f9e2af"   # warning / invalid input


# ---------------------------------------------------------------------------
# Application Class
# ---------------------------------------------------------------------------
class SpamClassifierApp:
    """
    Main application window for the Email Spam Classifier.

    Builds and manages all GUI widgets: title bar, text input area,
    classification result display, and action buttons.
    """

    def __init__(self, root: tk.Tk) -> None:
        """
        Initialize the application window and render all widgets.

        Parameters
        ----------
        root : tk.Tk
            The top-level Tkinter window passed from the entry point.
        """
        self.root = root
        self._configure_window()
        self._define_fonts()
        self._build_ui()

    # -----------------------------------------------------------------------
    # Window Configuration
    # -----------------------------------------------------------------------
    def _configure_window(self) -> None:
        """Set window title, size, background color, and disable resizing."""
        self.root.title("Email Spam Classifier")
        self.root.geometry("650x590")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

    # -----------------------------------------------------------------------
    # Font Definitions
    # -----------------------------------------------------------------------
    def _define_fonts(self) -> None:
        """Define reusable font objects for consistent typography."""
        self.font_title  = font.Font(family="Helvetica", size=20, weight="bold")
        self.font_label  = font.Font(family="Helvetica", size=11)
        self.font_result = font.Font(family="Helvetica", size=16, weight="bold")
        self.font_small  = font.Font(family="Helvetica", size=9)
        self.font_mono   = font.Font(family="Courier", size=10)

    # -----------------------------------------------------------------------
    # UI Construction
    # -----------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Construct and pack all GUI widgets in layout order."""
        self._build_header()
        self._build_divider()
        self._build_text_input()
        self._build_result_area()
        self._build_buttons()
        self._build_footer()

    def _build_header(self) -> None:
        """Render the application title and subtitle."""
        tk.Label(
            self.root,
            text="Email Spam Classifier",
            font=self.font_title,
            bg=BG, fg=TEXT,
        ).pack(pady=(25, 5))

        tk.Label(
            self.root,
            text="AI-powered spam detection using Naive Bayes",
            font=self.font_small,
            bg=BG, fg=SUBTEXT,
        ).pack()

    def _build_divider(self) -> None:
        """Render a horizontal separator line below the header."""
        tk.Frame(self.root, bg=SURFACE, height=2, width=600).pack(pady=15)

    def _build_text_input(self) -> None:
        """
        Render the email input label and multi-line text box.
        The text box is stored as self.text_box for later access.
        """
        tk.Label(
            self.root,
            text="Paste email text here:",
            font=self.font_label,
            bg=BG, fg=SUBTEXT,
        ).pack(anchor="w", padx=40)

        # Thin blue border frame wraps the text box
        border_frame = tk.Frame(self.root, bg=BLUE, padx=2, pady=2)
        border_frame.pack(padx=40, pady=(5, 15))

        self.text_box = tk.Text(
            border_frame,
            height=10, width=68,
            font=self.font_mono,
            bg=SURFACE, fg=TEXT,
            insertbackground="white",   # cursor color
            relief="flat",
            padx=12, pady=10,
            wrap="word",
        )
        self.text_box.pack()

    def _build_result_area(self) -> None:
        """
        Render the result label and confidence label.
        Both labels are stored as instance attributes so _run_analysis()
        can update them dynamically.
        """
        result_frame = tk.Frame(self.root, bg=BG)
        result_frame.pack()

        self.result_label = tk.Label(
            result_frame,
            text="Result will appear here...",
            font=self.font_result,
            bg=BG, fg=SUBTEXT,
        )
        self.result_label.pack()

        self.confidence_label = tk.Label(
            result_frame,
            text="",
            font=self.font_label,
            bg=BG, fg=SUBTEXT,
        )
        self.confidence_label.pack(pady=3)

    def _build_buttons(self) -> None:
        """Render the Analyze and Clear action buttons side by side."""
        btn_frame = tk.Frame(self.root, bg=BG)
        btn_frame.pack(pady=20)

        tk.Button(
            btn_frame,
            text="Analyze Email",
            font=self.font_label,
            bg=BLUE, fg="#1e1e2e",
            relief="flat",
            padx=25, pady=10,
            cursor="hand2",
            command=self._run_analysis,
        ).grid(row=0, column=0, padx=12)

        tk.Button(
            btn_frame,
            text="Clear",
            font=self.font_label,
            bg=RED, fg="#1e1e2e",
            relief="flat",
            padx=25, pady=10,
            cursor="hand2",
            command=self._clear_input,
        ).grid(row=0, column=1, padx=12)

    def _build_footer(self) -> None:
        """Render a static footer with project metadata."""
        tk.Label(
            self.root,
            text="BSCS Project  |  Naive Bayes Classifier  |  Accuracy ~97%",
            font=self.font_small,
            bg=BG, fg=SUBTEXT,
        ).pack(side="bottom", pady=10)

    # -----------------------------------------------------------------------
    # Event Handlers
    # -----------------------------------------------------------------------
    def _run_analysis(self) -> None:
        """
        Read text from the input box, call predict_spam(), and update
        the result and confidence labels with color-coded feedback.

        Shows a warning dialog if the input box is empty.
        """
        email_text = self.text_box.get("1.0", tk.END).strip()

        # Guard: require non-empty input before calling the model
        if not email_text:
            messagebox.showwarning(
                "Empty Input",
                "Please paste some email text before analyzing.",
            )
            return

        result, confidence = predict_spam(email_text)

        if result == "SPAM":
            self.result_label.config(text="SPAM EMAIL DETECTED!", fg=RED)
            self.confidence_label.config(
                text=f"Confidence: {confidence}%  |  This email is likely spam.",
                fg=RED,
            )

        elif result == "NOT SPAM":
            self.result_label.config(text="This Email is Safe!", fg=GREEN)
            self.confidence_label.config(
                text=f"Confidence: {confidence}%  |  This email looks legitimate.",
                fg=GREEN,
            )

        else:
            # Covers the INVALID case returned by predict_spam()
            self.result_label.config(text="Invalid Input", fg=YELLOW)
            self.confidence_label.config(
                text="Please enter valid email text.",
                fg=YELLOW,
            )

    def _clear_input(self) -> None:
        """
        Clear the text input box and reset both result labels
        to their default placeholder state.
        """
        self.text_box.delete("1.0", tk.END)
        self.result_label.config(text="Result will appear here...", fg=SUBTEXT)
        self.confidence_label.config(text="")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SpamClassifierApp(root)
    root.mainloop()
