import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import PyPDF2
from transformers import pipeline
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

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
    """Split text into chunks that the BART model can safely summarize."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        tokens = tokenizer(" ".join(current_chunk), return_tensors="pt")["input_ids"]

        # If chunk is too long, push previous chunk & start new one
        if tokens.size(1) > max_tokens:
            chunks.append(" ".join(current_chunk[:-1]))  # everything except the last word
            current_chunk = [word]  # start the next chunk

    # add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ------------------------------
# Summarize
# ------------------------------
def summarize_text():
    input_text = text_area.get("1.0", tk.END).strip()

    if not input_text:
        messagebox.showerror("Error", "No PDF text to summarize.")
        return

    status_label.config(text="Summarizing...")
    summary_output.delete("1.0", tk.END)

    # Start summarizing in a thread
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
            truncation=True,
            clean_up_tokenization_spaces=True
        )[0]["summary_text"]

        combined_summary += summary + "\n\n"

    # Insert final text back into UI
    summary_output.insert(tk.END, combined_summary)
    status_label.config(text="Summary complete!")



    threading.Thread(target=run_summary).start()



# ------------------------------
# Handle File Upload
# ------------------------------
def upload_pdf():
    file_path = filedialog.askopenfilename(
        title="Select PDF",
        filetypes=[("PDF Files", "*.pdf")]
    )

    if not file_path:
        return

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
root.title("AI PDF Summarizer")
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

# Input PDF text
text_area = scrolledtext.ScrolledText(root, height=15)
text_area.pack(padx=20, pady=10)

# Output Summary text
summary_output = scrolledtext.ScrolledText(root, height=15, bg="#f7f7f7")
summary_output.pack(padx=20, pady=10)

# Load model in background
threading.Thread(target=load_model).start()

root.mainloop()
