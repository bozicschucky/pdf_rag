# Resume Question-Answering Application

## Overview
A Streamlit-based application that allows users to query a resume or any pdf file you like using natural language processing and retrieve relevant information through semantic search and the google gemini API.

---

## Core Technologies
- **Python 3.8+**
- **ChromaDB**: Vector database for storing and retrieving document embeddings.
- **Streamlit**: Web interface framework.
- **PyMuPDF (fitz)**: PDF processing library.
- **Sentence Transformers**: Document embedding generation.
- **Google Gemini**: LLM for generating natural language responses.

---

## Features
- PDF resume parsing and chunking.
- Semantic search using document embeddings.
- Interactive Q&A interface.
- Markdown-formatted responses.

---

## Installation

### Environment Variables
Create a `.env` file with the required configuration for your API keys and environment variables.

---

## Usage
1. Place your resume PDF in the project directory.
2. Run the Streamlit application:
   ```bash
   streamlit run resume_rag.py
