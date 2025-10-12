# Local RAG Chatbot with LangChain & Ollama
This project is a simple yet powerful command-line chatbot that uses the Retrieval-Augmented Generation (RAG) pattern to answer questions about a specific document. It runs 100% locally using Ollama and an open-source Large Language Model (LLM), ensuring your data remains private.


![Chatbot Demo](https://raw.githubusercontent.com/Sadikn7i/LLM-Large-Language-Model--ex/master/chat_bot_screenshot.png)



---
## Key Features ✨

* **Private & Secure:** Runs entirely on your local machine. Your data is never sent to a third-party API.
* **Powered by Open-Source:** Uses powerful open-source models like Llama 3 via Ollama.
* **Document Q&A:** Ask questions about any text document (`.txt`, `.pdf`, etc.) or even content from a website.
* **Interactive:** Features a simple command-line interface for a conversational experience.

---
## How It Works ⚙️

The application follows the RAG pattern to provide context-aware answers:

1.  **Load:** A document (from a local file or a URL) is loaded.
2.  **Split:** The document is broken down into smaller, manageable text chunks.
3.  **Embed & Store:** Each chunk is converted into a numerical vector (embedding) using the local Ollama model and stored in a ChromaDB vector store.
4.  **Retrieve:** When a user asks a question, the application creates an embedding for the question and searches the vector store for the most relevant text chunks.
5.  **Generate:** The retrieved chunks (context) and the original question are passed to the LLM, which generates an answer based *only* on the provided information.

---
## Setup and Installation 🛠️

Follow these steps to get the project running on your local machine.

### 1. Prerequisites

* [Python 3.10+](https://www.python.org/downloads/)
* [Ollama](https://ollama.com/) installed and running.

### 2. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
```

### 3. Install Ollama & Download a Model

If you haven't already, install Ollama and pull a model. This project is configured to use `llama3`.

```bash
ollama run llama3
```

### 4. Set Up a Python Virtual Environment

It's a best practice to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment (Windows)
.\.venv\Scripts\activate

# On macOS/Linux, use: source .venv/bin/activate
```

### 5. Install Dependencies

Create a file named `requirements.txt` with the content below, then install the packages using `pip`.

**`requirements.txt`:**
```
langchain
langchain-ollama
chromadb
pypdf
beautifulsoup4
```

**Install command:**
```bash
pip install -r requirements.txt
```

---
## How to Run the Application 🚀

Once the setup is complete, you can run the chatbot with a single command:

```bash
python app.py
```

The script will index the default document and present you with an interactive prompt where you can start asking questions. Type `exit` to quit.

---
## How to Customize 🎨

You can easily adapt this chatbot to use your own data.

### Use a Local File

1.  Place your document (`.txt`, `.pdf`, etc.) in the project folder.
2.  In `app.py`, change the `FILE_PATH` variable to your filename.
3.  **Note:** For PDFs, you'll need to change the loader from `TextLoader` to `PyPDFLoader` and import it from `langchain_community.document_loaders`.

### Use a Website URL

1.  In `app.py`, replace the `setup_knowledge_base` function call with the one for URLs (see the example script for web scraping).
2.  Change the `URL` variable to the website you want to query.

---
## Technologies Used

* **Python**
* **LangChain**
* **Ollama**
* **ChromaDB**
