import os
from pathlib import Path
import subprocess
import shutil
from llama_cpp import Llama
import customtkinter
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from docagent import DocAgent

DOCUMENTS_DIR = './documents/'
llm_path = "/home/nnpy/Desktop/Semica/DocGPT-local/openchat-3.5-0106.Q3_K_M.gguf"
modelPath = "sentence-transformers/all-MiniLM-L12-v2"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': False}

agent = DocAgent(
    modelPath=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    llm_path=llm_path
)

customtkinter.set_appearance_mode("dark")
root = customtkinter.CTk()
root.title("Document Chatbot")
root.geometry('900x700')

mainframe = customtkinter.CTkFrame(root)
# mainframe = tk.Frame(root, bg='white', width=800, height=500)
mainframe.pack(fill="both", expand=True, padx=80, pady=20)
mainframe.place(relx=.2, rely=.1, anchor="nw")
mainframe.columnconfigure(2, weight=1)
mainframe.rowconfigure(0, weight=1)
mainframe.rowconfigure(1, weight=1)

sidebar = tk.LabelFrame(root, text="Options", padx=5, pady=5, bg="#393E46", fg="#EEEEEE")
sidebar.place(relwidth=.2, relheight=1, anchor="nw")

def clear_text():
    """Clear output textarea."""
    output_textarea['state'] = 'normal'
    output_textarea.delete("1.0","end")
    output_textarea['state'] = 'disable'

def show_message(msg):
    """Display messages in output textarea."""
    output_textarea['state'] = 'normal'
    output_textarea.insert("end", msg + "\n\n")
    output_textarea['state'] = 'disable'

def check_dir():
    """Check if DOCUMENTS_DIR exists, otherwise create it."""
    try:
        dirpath = Path(DOCUMENTS_DIR).resolve().parent
        os.makedirs(dirpath, exist_ok=True)
    except OSError as error:
        print(error)

def upload_doc():
    """Open File Dialog Box to select and upload documents."""
    filename = filedialog.askopenfilename(initialdir="/", title="Select a pdf files", filetypes=[("PDF Files", ".pdf")])
    if os.path.exists(DOCUMENTS_DIR):
        files = os.listdir(DOCUMENTS_DIR)
        files.remove("getting_real_basecamp.pdf")
        for file in files:
            file_path = os.path.join(DOCUMENTS_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    if len(filename) > 0:
        dest_folder = "./documents/"
        try:
            os.makedirs(dest_folder, exist_ok=True)
            shutil.copy(str(filename), dest_folder)
            print('folder contains: ', os.listdir(dest_folder))
            agent.create_db(dest_folder)
            messagebox.showinfo("Success", "Document uploaded successfully!")
        except Exception as err:
            show_message(err)
            messagebox.showerror("Error", "An error occurred while uploading document!")

def clear_documents():
    """Clear documents from documents directory."""
    if os.path.exists(DOCUMENTS_DIR):
        files = os.listdir(DOCUMENTS_DIR)
        files.remove("getting_real_basecamp.pdf")
        for file in files:
            file_path = os.path.join(DOCUMENTS_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
    else:
        messagebox.showerror("Error", "No documents found to clear!")
    # show_message("Documents cleared successfully!")
    agent.db = None
    messagebox.showinfo("Success", "Documents cleared successfully!")

def submit_query(*args):
    """Submit query entered in input textbox."""
    # write loading message to output textarea
    show_message("Loading...")
    query = query_entry.get()
    if len(query) == 0:
        return
    clear_text()
    full_response = agent.get_response(query)
    print(full_response)
    clear_text()
    show_message(full_response)

    agent.messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )

query_entry = customtkinter.CTkEntry(mainframe, font=("Poppins", 17), width=230, height=45, placeholder_text="Enter query here...", text_color="#EEEEEE", corner_radius=15)
query_entry.bind("<Return>", submit_query)
query_entry.grid(row=0, column=2, padx=10, pady=10, sticky="ew")

output_textarea = customtkinter.CTkTextbox(mainframe, corner_radius=15, font=("Poppins", 17), width=700, height=500, border_width=3, wrap="word", text_color="#EEEEEE")
output_textarea['state'] = 'disabled'
output_textarea.grid(row=1, column=2, padx=10, pady=5, sticky="ew")

upload_button = customtkinter.CTkButton(sidebar, text="Upload docs", command=upload_doc, corner_radius=15, border_width=3, fg_color="transparent", height=34, width=145, font=("Poppins", 15), border_color="#00ADB5", text_color="#EEEEEE", hover_color="#00ADB5")
upload_button.pack(pady=10)

clear_documents_button = customtkinter.CTkButton(sidebar, text="Clear documents", command=clear_documents, corner_radius=15, border_width=3, fg_color="transparent", height=34, width=145, font=("Poppins", 15), border_color="#00ADB5", text_color="#EEEEEE", hover_color="#00ADB5")
clear_documents_button.pack(pady=10)

def update_settings(context_length, n_threads, *args):
    """Update settings for context length and number of threads."""
    if int(context_length) > 0:
        agent.llm.n_ctx = int(context_length)
    if int(n_threads) > 0:
        agent.llm.n_threads = int(n_threads)
    messagebox.showinfo("Success", "Settings updated successfully!")

def update_context_length_label(value):
    context_length_label.configure(text=f"Context Length: {int(value)}")

def update_n_threads_label(value):
    n_threads_label.configure(text=f"Threads: {int(value)}")

def settings():
    """Open settings dialog box."""
    settings_window = tk.Toplevel(root)
    settings_window.title("Settings")
    settings_window.geometry("400x300")
    settings_window.tk_setPalette(background="black", foreground="white")

    global context_length_label, n_threads_label

    context_length_label = customtkinter.CTkLabel(settings_window, text="Context Length: 4096", text_color="#EEEEEE")
    context_length_label.pack(pady=5)
    context_length_entry = customtkinter.CTkSlider(settings_window, from_=2000, to=8096, number_of_steps=8096-2000)
    context_length_entry.set(4096)
    context_length_entry.pack(pady=5)
    context_length_entry.bind("<ButtonRelease-1>", lambda event: update_context_length_label(context_length_entry.get()))

    n_threads_label = customtkinter.CTkLabel(settings_window, text=f"Threads: {os.cpu_count()-4}", text_color="#EEEEEE")
    n_threads_label.pack(pady=15)
    n_threads_entry = customtkinter.CTkSlider(settings_window, from_=1, to=os.cpu_count()-1, number_of_steps=os.cpu_count()-2)
    n_threads_entry.set(os.cpu_count()-4)
    n_threads_entry.pack(pady=5)
    n_threads_entry.bind("<ButtonRelease-1>", lambda event: update_n_threads_label(n_threads_entry.get()))

    btn = customtkinter.CTkButton(settings_window, text="Save", command=lambda: update_settings(context_length_entry.get(), n_threads_entry.get(), settings_window.destroy()), corner_radius=15, border_width=3, fg_color="transparent", height=34, width=145, font=("Poppins", 15), border_color="#00ADB5", text_color="#EEEEEE", hover_color="#00ADB5")
    btn.pack(pady=10)

settings_button = customtkinter.CTkButton(sidebar, text="Settings", command=settings, corner_radius=15, border_width=3, fg_color="transparent", height=34, width=145, font=("Poppins", 15), border_color="#00ADB5", text_color="#EEEEEE", hover_color="#00ADB5")
settings_button.pack(pady=10)

check_dir()
clear_text()
show_message("Welcome! start to chat with your documents...")

root.mainloop()