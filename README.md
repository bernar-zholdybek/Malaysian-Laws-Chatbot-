# PDF RAG Chatbot Assistant
An AI-powered chatbot built with Streamlit and LangChain that allows to upload PDF documents, index them into a local vector database, and have context-aware conversations using the Google Gemini API.


## Features
* **PDF Document Ingestion:** Extracts and splits text from PDF files seamlessly
* **Vector Storage:** Uses ChromaDB and HuggingFace sentence-transformers to store and retrieve document contexts
* **Smart Answers:** Leverages Google's Gemini models via LangChain for precise, grounded responses
* **Interactive UI:** A clean, easy-to-use web interface powered by Streamlit

    
## Tech Stack
* **UI:** Streamlit
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace (`sentence-transformers`)
* **LLM:** Google Generative AI (Gemini)

## Required Libraries
    streamlit
    langchain
    langchain-huggingface
    langchain-chroma
    langchain-google-genai
    langchain-text-splitters
    google-generativeai
    chromadb
    pypdf
    sentence-transformers


## 1. How to Install

Install the required packages for the RAG and LLM

```bash
pip install -r requirements.txt
```
    
## 2. How to Run
move to project folder 

``` bash 
cd Downloads/your_project_folder

streamlit run newcode.py
```
