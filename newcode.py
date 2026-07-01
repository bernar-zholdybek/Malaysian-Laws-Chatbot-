import os
import streamlit as st
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

GOOGLE_API_KEY = "AQ.Ab8RN6LGGsz3G-8t5Va3O_wSwPgERTZozrYdLG1LazJowv0efQ"

LAW_PDFS = {
    "laws/companies_act.pdf": {
        "act_name": "Companies Act",
        "act_year": 2016,
        "category": "Corporate",
    },
    "laws/consumer_protection_act.pdf": {
        "act_name": "Consumer Protection Act",
        "act_year": 1999,
        "category": "Consumer",
    },
    "laws/contracts_act.pdf": {
        "act_name": "Contracts Act",
        "act_year": 1950,
        "category": "Civil",
    },
    "laws/domestic_violence_act.pdf": {
        "act_name": "Domestic Violence Act",
        "act_year": 1994,
        "category": "Family",
    },
    "laws/employment_act.pdf": {
        "act_name": "Employment Act",
        "act_year": 1955,
        "category": "Employment",
    },
    "laws/penal_code.pdf": {
        "act_name": "Penal Code",
        "act_year": 1997,
        "category": "Criminal",
    },
    "laws/road_transport_act.pdf": {
        "act_name": "Road Transport Act",
        "act_year": 1987,
        "category": "Transport",
    },
}

DB_DIR        = "./chroma_db"
INGESTED_FLAG = os.path.join(DB_DIR, ".ingested_acts")

# ---------------------------------------------------------------------------
# PAGE CONFIG — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MY-LawBot",
    page_icon="🏛️",
    layout="centered",
)

st.title("🏛️ MY-LawBot: Malaysian Law Assistant")
st.caption(
    "Ask me anything about Malaysian law. "
    "I'll give you a plain-English summary with sources."
)

# ---------------------------------------------------------------------------
# CACHED BACKEND: vector store (runs once, survives reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def initialize_system():
    """
    Load or build the Chroma vector store.
    - On first run: ingests all PDFs found on disk.
    - On subsequent runs: only ingests PDFs not yet in the store.
    - Missing PDFs are skipped with a warning (app still starts).
    """
    # Ensure the DB directory exists before we try to write the flag file
    os.makedirs(DB_DIR, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    splitter   = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    # Find out which PDFs are already ingested
    already_done = set()
    if os.path.exists(INGESTED_FLAG):
        with open(INGESTED_FLAG) as f:
            already_done = set(f.read().splitlines())

    # Build chunks for any PDFs not yet in the store
    new_chunks       = []
    newly_ingested   = []

    for filename, meta in LAW_PDFS.items():
        if filename in already_done:
            continue
        if not os.path.exists(filename):
            st.warning(f"PDF not found: `{filename}` — skipping. Add it to the app folder to enable this act.")
            continue

        with st.spinner(f"Compiling legal database: {meta['act_name']}…"):
            try:
                reader = pypdf.PdfReader(filename)
                docs = [
                    Document(
                        page_content=page.extract_text() or "",
                        metadata={
                            "source_file": filename,
                            "act_name":    meta["act_name"],
                            "act_year":    meta["act_year"],
                            "category":    meta["category"],
                            "page":        i + 1,   # 1-based page number
                        },
                    )
                    for i, page in enumerate(reader.pages)
                    if page.extract_text()  # skip blank / image-only pages
                ]
                new_chunks.extend(splitter.split_documents(docs))
                newly_ingested.append(filename)
            except Exception as e:
                st.warning(f"Could not read `{filename}`: {e}")

    # Connect to (or create) the Chroma store
    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )

    # Add new chunks and record which files are now done
    if new_chunks:
        vector_store.add_documents(new_chunks)
        with open(INGESTED_FLAG, "a") as f:
            for filename in newly_ingested:
                f.write(filename + "\n")

    return vector_store


# ---------------------------------------------------------------------------
# CACHED BACKEND: LLM (runs once, not recreated on every message)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,   # low = factual, not creative
    )


# ---------------------------------------------------------------------------
# BOOT
# ---------------------------------------------------------------------------

vector_store = initialize_system()

# ---------------------------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": (
                "Hello! I'm ready to answer your legal questions. "
                "Ask me anything about Malaysian law and I'll explain it in plain language."
            ),
        }
    ]

# ---------------------------------------------------------------------------
# DISPLAY EXISTING CHAT HISTORY
# ---------------------------------------------------------------------------

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show source clauses only for assistant messages that have them
        if message.get("references"):
            with st.expander("⚖️ View source clauses"):
                for ref in message["references"]:
                    st.markdown(
                        f"**📄 {ref['act_name']} ({ref['act_year']})** "
                        f"— Page {ref['page']} · *{ref['category']}*"
                    )
                    st.info(ref["text"])
                    st.divider()

# ---------------------------------------------------------------------------
# HANDLE NEW USER INPUT
# ---------------------------------------------------------------------------

if user_query := st.chat_input("Type your legal question here…"):

    # Show the user message immediately
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):

        # Guard: need an API key to call the LLM
        if not GOOGLE_API_KEY:
            msg = (
                "⚠️ No Google Gemini API key found. "
                "Set the `GOOGLE_API_KEY` environment variable and restart the app."
            )
            st.warning(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})

        else:
            with st.spinner("Searching statutes and generating summary…"):
                try:
                    # 1. RETRIEVAL — find the most relevant legal chunks
                    results = vector_store.similarity_search_with_relevance_scores(
                        user_query, k=5
                    )

                    # Filter out low-confidence results (below 0.4 cosine similarity)
                    results = [(doc, score) for doc, score in results if score >= 0.4]

                    if not results:
                        response = (
                            "I cannot find a relevant legal clause in my database "
                            "to answer that question. Please try rephrasing, or the "
                            "topic may not be covered by the acts currently loaded."
                        )
                        st.markdown(response)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": response}
                        )

                    else:
                        # 2. BUILD CONTEXT — include act name in each clause header
                        context_parts  = []
                        references_data = []

                        for idx, (doc, score) in enumerate(results):
                            act  = doc.metadata.get("act_name", "Unknown Act")
                            year = doc.metadata.get("act_year", "")
                            pg   = doc.metadata.get("page", 0)   # already 1-based

                            context_parts.append(
                                f"Clause {idx + 1} [{act} {year}, page {pg}]:\n"
                                f"{doc.page_content}"
                            )
                            references_data.append({
                                "text":     doc.page_content,
                                "act_name": act,
                                "act_year": year,
                                "category": doc.metadata.get("category", ""),
                                "page":     pg,
                            })

                        joined_context = "\n\n".join(context_parts)

                        # 3. PROMPT
                        prompt = f"""You are MY-LawBot, a helpful assistant specialising in Malaysian law.

Using ONLY the legal clauses provided below, write a clear and friendly plain-English answer to the user's question.
Your response must:
- Open with a direct one-sentence answer to the question.
- State which Act (and section number if visible in the clause) the answer is based on.
- Explain the key legal points in simple language that a non-lawyer can understand.
- Use short paragraphs — no bullet lists, no raw clause numbers unless essential.
- If the clauses do not contain enough information to answer the question fully, say so honestly.
- Do NOT reproduce the raw clause text verbatim in your answer.

Retrieved legal clauses:
{joined_context}

User question: {user_query}

Plain-English answer:"""

                        # 4. CALL THE LLM (cached, not re-created each turn)
                        llm     = get_llm()
                        summary = llm.invoke(prompt).content

                        # 5. DISPLAY
                        st.markdown(summary)

                        with st.expander("⚖️ View source clauses"):
                            for ref in references_data:
                                st.markdown(
                                    f"**📄 {ref['act_name']} ({ref['act_year']})** "
                                    f"— Page {ref['page']} · *{ref['category']}*"
                                )
                                st.info(ref["text"])
                                st.divider()

                        # 6. SAVE TO HISTORY
                        st.session_state.chat_history.append({
                            "role":       "assistant",
                            "content":    summary,
                            "references": references_data,
                        })

                except Exception as e:
                    error_msg = (
                        "⚠️ Something went wrong while generating the answer. "
                        "Please try again in a moment."
                    )
                    st.error(error_msg)
                    st.caption(f"Technical detail: {e}")
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_msg}
                    )