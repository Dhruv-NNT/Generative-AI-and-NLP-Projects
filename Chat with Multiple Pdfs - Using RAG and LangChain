# 📚 Chat with Multiple PDFs

Welcome to the "Chat with Multiple PDFs" project! This application allows you to chat with the contents of multiple PDFs simultaneously. Upload your PDFs, ask questions about the information they contain, and get answers using a large language model that finds relevant sections of the PDFs.

## 📝 Project Overview

This application simplifies the process of extracting and querying information from multiple PDF documents. By leveraging powerful language models, it provides coherent and contextually accurate answers based on the content within the uploaded PDFs.

## 🔍 How It Works

1. **Upload PDFs**: Drag and drop your PDFs into the designated area.
2. **Process**: Click the "Process" button to parse the PDFs and create a searchable database of their contents.
3. **Ask a Question**: Type your question in the text box.
4. **Answer**: The application will search the database and use a large language model to provide an answer based on the relevant information from your PDFs.

## 🌟 Features

- **Upload Multiple PDFs**: Easily upload multiple PDF files to query simultaneously.
- **Searchable Content**: Parses and indexes the content for quick retrieval.
- **Intelligent Q&A**: Uses a large language model to answer questions based on the PDF contents.

## 📦 Requirements

- Python 3.9
- Streamlit
- PyPDF2 (for reading PDFs)
- LangChain (for vector embeddings and database)
- OpenAI API key (optional, for paid embeddings)
- Hugging Face Hub API token (optional, for free instructor embeddings)

## ⚙️ Setup

1. **Create a virtual environment and install the required dependencies:**

```bash
python -m venv env
source env/bin/activate
pip install streamlit pypdf2 langchain openai transformers
```

2. **Create an API key for OpenAI (if you want to use paid embeddings) and a token for Hugging Face Hub (if you want to use free instructor embeddings). Store them securely in a `.env` file.**

3. **Load the API keys from the `.env` file:**

```python
from dotenv import load_dotenv
load_dotenv()
```

## 🚀 Running the Application

1. **Save the code in a file named `app.py`.**

2. **Run the application:**

```bash
streamlit run app.py
```

## 📘 Example Usage

1. **Upload PDFs**: Upload two PDFs, such as the Constitution and the Bill of Rights.
2. **Process**: Click "Process".
3. **Ask a Question**: Ask a question related to the content of the PDFs, such as "What are the three branches of the United States government?".
4. **Get an Answer**: The application will search the PDFs and use a large language model to answer your question.

## ⚠️ Disclaimer

The instructor embeddings may be slower than OpenAI embeddings if run on your local CPU. Consider using a GPU for better performance.

## 📧 Contact

For support, questions, or feedback, feel free to reach out.
