"""
Simple UI for converting attraction reviews into a structured dataset.

Run from backend/:
	python review_processor_ui.py
"""

import csv
import io
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

from src.review_processor import process_attraction_reviews


class ReviewProcessorUI:
	def __init__(self, root: tk.Tk):
		self.root = root
		self.root.title("Attraction Review Processor")
		self.root.geometry("1200x700")

		main = ttk.Frame(root, padding=12)
		main.pack(fill="both", expand=True)

		left = ttk.Frame(main)
		right = ttk.Frame(main)
		left.pack(side="left", fill="both", expand=True, padx=(0, 8))
		right.pack(side="left", fill="both", expand=True, padx=(8, 0))

		ttk.Label(left, text="Input Review(s)").pack(anchor="w")
		self.input_reviews = ScrolledText(left, wrap="word", height=30)
		self.input_reviews.pack(fill="both", expand=True, pady=(0, 8))

		btn_row = ttk.Frame(left)
		btn_row.pack(fill="x")
		ttk.Button(btn_row, text="Process", command=self.process).pack(side="left")
		ttk.Button(btn_row, text="Clear", command=self.clear_all).pack(side="left", padx=(8, 0))
		ttk.Button(btn_row, text="Copy All", command=self.copy_all_to_clipboard).pack(side="left", padx=(8, 0))

		ttk.Label(right, text="category").pack(anchor="w")
		self.category_box = ScrolledText(right, wrap="word", height=2)
		self.category_box.pack(fill="both", expand=True, pady=(0, 8))

		ttk.Label(right, text="vibes").pack(anchor="w")
		self.vibes_box = ScrolledText(right, wrap="word", height=3)
		self.vibes_box.pack(fill="both", expand=True, pady=(0, 8))

		ttk.Label(right, text="environmental_tags").pack(anchor="w")
		self.environmental_box = ScrolledText(right, wrap="word", height=3)
		self.environmental_box.pack(fill="both", expand=True, pady=(0, 8))

		ttk.Label(right, text="demographic_tags").pack(anchor="w")
		self.demographic_box = ScrolledText(right, wrap="word", height=3)
		self.demographic_box.pack(fill="both", expand=True, pady=(0, 8))

		ttk.Label(right, text="summary_desc").pack(anchor="w")
		self.summary_box = ScrolledText(right, wrap="word", height=6)
		self.summary_box.pack(fill="both", expand=True, pady=(0, 8))

		ttk.Label(right, text="facilities_desc").pack(anchor="w")
		self.facilities_box = ScrolledText(right, wrap="word", height=6)
		self.facilities_box.pack(fill="both", expand=True, pady=(0, 8))

		ttk.Label(right, text="tips_desc").pack(anchor="w")
		self.tips_box = ScrolledText(right, wrap="word", height=6)
		self.tips_box.pack(fill="both", expand=True)

	def _write_output(self, widget: ScrolledText, value: str) -> None:
		widget.delete("1.0", tk.END)
		widget.insert(tk.END, value)

	def clear_all(self) -> None:
		self.input_reviews.delete("1.0", tk.END)
		self._write_output(self.category_box, "")
		self._write_output(self.vibes_box, "")
		self._write_output(self.environmental_box, "")
		self._write_output(self.demographic_box, "")
		self._write_output(self.summary_box, "")
		self._write_output(self.facilities_box, "")
		self._write_output(self.tips_box, "")

	def copy_all_to_clipboard(self) -> None:
		category = self.category_box.get("1.0", tk.END).strip()
		vibes = self.vibes_box.get("1.0", tk.END).strip()
		environmental_tags = self.environmental_box.get("1.0", tk.END).strip()
		demographic_tags = self.demographic_box.get("1.0", tk.END).strip()
		summary = self.summary_box.get("1.0", tk.END).strip()
		facilities = self.facilities_box.get("1.0", tk.END).strip()
		tips = self.tips_box.get("1.0", tk.END).strip()

		if not (category or vibes or environmental_tags or demographic_tags or summary or facilities or tips):
			messagebox.showwarning("Nothing to Copy", "Please process reviews first.")
			return

		# Build one tab-separated row so Excel paste fills all columns in one shot.
		buffer = io.StringIO(newline="")
		writer = csv.writer(buffer, dialect="excel-tab", lineterminator="")
		writer.writerow([
			category,
			vibes,
			environmental_tags,
			demographic_tags,
			summary,
			facilities,
			tips,
		])
		clipboard_text = buffer.getvalue()

		self.root.clipboard_clear()
		self.root.clipboard_append(clipboard_text)
		self.root.update()
		messagebox.showinfo("Copied", "Copied all results. Paste directly into Excel.")

	def process(self) -> None:
		review_text = self.input_reviews.get("1.0", tk.END).strip()

		if not review_text:
			messagebox.showwarning("Missing Input", "Please enter at least one review.")
			return

		try:
			result = process_attraction_reviews(reviews=review_text)
		except Exception as exc:
			messagebox.showerror("Processing Failed", str(exc))
			return

		self._write_output(self.category_box, result.get("category") or "")
		self._write_output(self.vibes_box, ", ".join(result.get("vibes", [])))
		self._write_output(self.environmental_box, ", ".join(result.get("environmental_tags", [])))
		self._write_output(self.demographic_box, ", ".join(result.get("demographic_tags", [])))
		self._write_output(self.summary_box, result["summary_desc"])
		self._write_output(self.facilities_box, result["facilities_desc"])
		self._write_output(self.tips_box, result["tips_desc"])


def main() -> None:
	root = tk.Tk()
	ReviewProcessorUI(root)
	root.mainloop()


if __name__ == "__main__":
	main()
