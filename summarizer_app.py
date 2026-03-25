import os
import threading
import tkinter as tk
import time
from tkinter import filedialog, messagebox, scrolledtext
import PyPDF2
from transformers import pipeline, AutoTokenizer
from rouge_score import rouge_scorer


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# Store current PDF name
current_pdf_name = None

# ------------------------------
# Load Model Once
# ------------------------------
def load_model():
    global summarizer
    status_label.config(text="Loading AI model (this may take ~15s)...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    status_label.config(text="Model loaded. Ready!")


# ------------------------------
# PDF → Text
# ------------------------------
def extract_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# ------------------------------
# Chunk Text
# ------------------------------
def chunk_text(text, max_tokens=800):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        tokens = tokenizer(" ".join(current_chunk), return_tensors="pt")["input_ids"]

        if tokens.size(1) > max_tokens:
            chunks.append(" ".join(current_chunk[:-1]))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ------------------------------
# Load Reference Summaries
# ------------------------------
def load_references():
    ref_path = "references.txt"
    references = {}

    if not os.path.exists(ref_path):
        return references

    with open(ref_path, "r", encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                filename, summary = line.strip().split("|", 1)
                references[filename] = summary

    return references


# ------------------------------
# ROUGE Evaluation
# ------------------------------
def evaluate_summary(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)

    return scores['rouge1'].fmeasure, scores['rougeL'].fmeasure


# ------------------------------
# Summarize
# ------------------------------
def summarize_text():
    input_text = text_area.get("1.0", tk.END).strip()

    if not input_text:
        messagebox.showerror("Error", "No PDF text to summarize.")
        return

    summary_output.delete("1.0", tk.END)
    threading.Thread(target=run_summary, args=(input_text,)).start()


def run_summary(input_text):
    chunks = chunk_text(input_text, max_tokens=800)
    combined_summary = ""

    for i, chunk in enumerate(chunks):
        status_label.config(text=f"Summarizing chunk {i+1}/{len(chunks)}...")

        summary = summarizer(
            chunk,
            max_length=200,
            min_length=50,
            truncation=True
        )[0]["summary_text"]

        combined_summary += summary + "\n\n"

    summary_output.insert(tk.END, combined_summary)
    status_label.config(text="Summary complete!")

    # ------------------------------
    # ROUGE Evaluation
    # ------------------------------
    references = load_references()

    if current_pdf_name and current_pdf_name in references:
        reference = references[current_pdf_name]

        rouge1, rougeL = evaluate_summary(reference, combined_summary)

        summary_output.insert(
            tk.END,
            f"\n--- ROUGE Evaluation ---\nROUGE-1: {rouge1:.4f}\nROUGE-L: {rougeL:.4f}\n"
        )

        status_label.config(
            text=f"Summary complete! ROUGE-1: {rouge1:.3f}, ROUGE-L: {rougeL:.3f}"
        )
    else:
        summary_output.insert(
            tk.END,
            "\n--- ROUGE Evaluation ---\nNo reference summary found.\n"
        )


# ------------------------------
# Handle File Upload
# ------------------------------
def upload_pdf():
    global current_pdf_name

    file_path = filedialog.askopenfilename(
        title="Select PDF",
        filetypes=[("PDF Files", "*.pdf")]
    )

    if not file_path:
        return

    current_pdf_name = os.path.basename(file_path)

    status_label.config(text="Extracting text from PDF...")

    def extract_thread():
        try:
            text = extract_pdf_text(file_path)
            text_area.delete("1.0", tk.END)
            text_area.insert(tk.END, text)
            status_label.config(text="PDF text loaded.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    threading.Thread(target=extract_thread).start()


# ------------------------------
# Save Summary
# ------------------------------
def save_summary():
    text = summary_output.get("1.0", tk.END).strip()
    if not text:
        messagebox.showerror("Error", "Nothing to save.")
        return

    save_path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt")]
    )

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        messagebox.showinfo("Saved", "Summary saved successfully!")


# ------------------------------
# GUI Setup
# ------------------------------
root = tk.Tk()
root.title("AI PDF Summarizer + ROUGE Evaluation")
root.geometry("900x700")
root.resizable(False, False)

title_label = tk.Label(
    root,
    text="AI-Powered PDF Summarizer",
    font=("Arial", 18, "bold")
)
title_label.pack(pady=10)

btn_frame = tk.Frame(root)
btn_frame.pack()

upload_btn = tk.Button(btn_frame, text="Upload PDF", command=upload_pdf, width=20)
upload_btn.grid(row=0, column=0, padx=10)

summarize_btn = tk.Button(btn_frame, text="Summarize", command=summarize_text, width=20)
summarize_btn.grid(row=0, column=1, padx=10)

save_btn = tk.Button(btn_frame, text="Save Summary", command=save_summary, width=20)
save_btn.grid(row=0, column=2, padx=10)

status_label = tk.Label(root, text="Loading AI model...", font=("Arial", 10))
status_label.pack()

text_area = scrolledtext.ScrolledText(root, height=15)
text_area.pack(padx=20, pady=10)

summary_output = scrolledtext.ScrolledText(root, height=15, bg="#f7f7f7")
summary_output.pack(padx=20, pady=10)

threading.Thread(target=load_model).start()

root.mainloop()
