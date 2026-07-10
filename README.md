# PDF RAG Malaysian Laws Chatbot
An AI-powered chatbot built with Streamlit and LangChain that allows to upload PDF documents, index them into a local vector database, and have context-aware conversations using the Google Gemini API.


## Features
* **PDF Document Ingestion:** Extracts and splits text from PDF files seamlessly
* **Vector Storage:** Uses ChromaDB and HuggingFace sentence-transformers to store and retrieve document contexts
* **Smart Answers:** Leverages Google's Gemini models via LangChain for precise, grounded responses
* **Interactive UI:** A clean, easy-to-use web interface powered by Streamlit


## Project Structure
    Malaysian-Laws-Chatbot/
    │
    ├── laws/                        
    │   ├── employment_act.pdf
    │   ├── contracts_act.pdf
    │   ├── companies_act.pdf
    │   ├── consumer_protection_act.pdf
    │   ├── domestic_violence_act.pdf
    │   ├── penal_code.pdf
    │   └── road_transport_act.pdf
    │
    ├── chroma_db/                   # auto-generated on first run
    │
    ├── app.py                       # main code
    ├── requirements.txt             # required dependencies
    ├── .env                         # API key 
    └── README.md

## Prerequisites
* Python 3.10
* Free Google Gemini API key (can be acquired at *aistudio.google.com*)
* PDF files of law acts, placed in the laws/ folder
  

## Required Libraries
    python-dotenv
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


## Setup

### 1. Clone Repository
```bash
git clone https://github.com/bernar-zholdybek/Malaysian-Laws-Chatbot.git
cd Malaysian-Laws-Chatbot
```

### 2. Dependecy Installation
Install all the required packages
```bash
pip install -r requirements.txt
```

### 3. Set API key
Get API key at *aistudio.google.com*. Create a ".env" (without name) file in the project root with this text inside:
```bash
GOOGLE_API_KEY = API_key_from_website
```
Example provided in picture below

<img width="339" height="76" alt="image" src="https://github.com/user-attachments/assets/b6dc268f-c184-473c-a7d3-29a84fc2041a" />



### 4. How to Run
Run the programm
```bash
streamlit run app.py
```
(note: app will build the vector database from the PDFs automatically. This takes a few minutes. Subsequent starts are instant.)

    
## Tech Stack
* **UI:** Streamlit
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace (`sentence-transformers`)
* **LLM:** Google Generative AI (Gemini)

## About Chatbot Limits
*    Coverage is limited to the seven acts listed above; questions outside these will return no results
*    The chatbot cannot provide legal advice, it only summarises statute text
