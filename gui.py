import tkinter as tk
from tkinter import font, messagebox
from predict import predict_spam

# Colors
BG = "#1e1e2e"
SURFACE = "#313244"
TEXT = "#cdd6f4"
SUBTEXT = "#a6adc8"
GREEN = "#a6e3a1"
RED = "#f38ba8"
BLUE = "#89b4fa"


class SpamClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Email Spam Classifier")
        self.root.geometry("680x600")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self._setup_fonts()
        self._build_ui()

    def _setup_fonts(self):
        self.f_title = ("Helvetica", 20, "bold")
        self.f_label = ("Helvetica", 11)
        self.f_result = ("Helvetica", 16, "bold")
        self.f_small = ("Helvetica", 9)
        self.f_mono = ("Courier", 10)

    def _build_ui(self):
        tk.Label(self.root, text="Email Spam Classifier", font=self.f_title, bg=BG, fg=TEXT).pack(pady=(25, 4))
        tk.Label(self.root, text="AI-powered spam detection", font=self.f_small, bg=BG, fg=SUBTEXT).pack()

        tk.Frame(self.root, bg=SURFACE, height=2, width=620).pack(pady=14)

        tk.Label(self.root, text="Paste your email text below:", font=self.f_label, bg=BG, fg=SUBTEXT).pack(anchor="w", padx=40)

        border = tk.Frame(self.root, bg=BLUE, padx=2, pady=2)
        border.pack(padx=40, pady=(5, 15))

        self.text_box = tk.Text(border, height=10, width=70, font=self.f_mono, bg=SURFACE, fg=TEXT, insertbackground="white")
        self.text_box.pack()

        self.lbl_result = tk.Label(self.root, text="Result will appear here...", font=self.f_result, bg=BG, fg=SUBTEXT)
        self.lbl_result.pack(pady=(5, 0))

        self.lbl_confidence = tk.Label(self.root, text="", font=self.f_label, bg=BG, fg=SUBTEXT)
        self.lbl_confidence.pack(pady=(3, 0))

        btn_frame = tk.Frame(self.root, bg=BG)
        btn_frame.pack(pady=20)

        tk.Button(btn_frame, text="Analyze Email", font=self.f_label, bg=BLUE, fg="#1e1e2e", padx=18, pady=9, command=self._analyze).grid(row=0, column=0, padx=12)
        tk.Button(btn_frame, text="Clear", font=self.f_label, bg=RED, fg="#1e1e2e", padx=18, pady=9, command=self._clear).grid(row=0, column=1, padx=12)

        tk.Label(self.root, text="BSCS Project | Accuracy ~97%", font=self.f_small, bg=BG, fg=SUBTEXT).pack(side="bottom", pady=10)

    def _analyze(self):
        email_text = self.text_box.get("1.0", tk.END).strip()
        if not email_text:
            messagebox.showwarning("Empty", "Please paste an email.")
            return

        try:
            result, confidence = predict_spam(email_text)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        if result == "SPAM":
            self.lbl_result.config(text="⚠ SPAM EMAIL DETECTED!", fg=RED)
            self.lbl_confidence.config(text=f"Confidence: {confidence}%", fg=RED)
        else:
            self.lbl_result.config(text="✔ This Email Looks Safe!", fg=GREEN)
            self.lbl_confidence.config(text=f"Confidence: {confidence}%", fg=GREEN)

    def _clear(self):
        self.text_box.delete("1.0", tk.END)
        self.lbl_result.config(text="Result will appear here...", fg=SUBTEXT)
        self.lbl_confidence.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    SpamClassifierApp(root)
    root.mainloop()