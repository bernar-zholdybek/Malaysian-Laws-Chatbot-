# PDF RAG Malaysian Laws Chatbot
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


## 1. Dependecy Installation

Install all the required packages

```bash
pip install -r requirements.txt
```

## 2. How to Run
Move to the project folder 

``` bash 
cd Downloads/example_location_project_folder
```

Run the programm
```bash
streamlit run newcode.py
```
