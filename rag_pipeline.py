"""
RAG pipeline for Chiq IQ.

Replaces Vext AI with a local LangChain + FAISS pipeline backed by OpenAI.
- Loads 7 fashion knowledge-base articles from knowledge_base/
- Embeds them with OpenAI text-embedding-3-small (fast, cheap, accurate)
- Wraps GPT-4o in a ConversationalRetrievalChain so multi-turn context is preserved
"""

import os
import glob
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ---------------------------------------------------------------------------
# System prompt — the persona and instruction layer for GPT-4o
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """You are Chiq IQ, a sophisticated personal AI fashion stylist. \
You have expert knowledge in:
- Occasion-appropriate dressing (weddings, interviews, parties, casual events, festive occasions)
- Seasonal styling and weather-appropriate outfit building for all four seasons
- Color theory: flattering combinations, undertone matching, color psychology
- Body type styling: how to dress for every shape and frame to feel your best
- Elevated minimalist aesthetics — clean lines, quality fabrics, understated polish
- Wardrobe building: capsule wardrobes, essential pieces, investment vs. budget choices

Your personality is warm, encouraging, and fashion-forward. You give specific, actionable \
recommendations — actual pieces, colors, and silhouettes — not vague platitudes.

When you don't know enough about the user's needs, ask a focused follow-up question \
(occasion details, weather/season, body type, style preference, or budget).

Use the following fashion knowledge to ground your recommendations:
{context}

Previous conversation:
{chat_history}"""

_HUMAN_TEMPLATE = "{question}"


# ---------------------------------------------------------------------------
# Vectorstore — cached so embeddings only run once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Building fashion knowledge base…")
def get_vectorstore(openai_api_key: str) -> FAISS:
    """
    Load all .txt files from knowledge_base/, chunk them, embed with OpenAI,
    and return a FAISS vectorstore.  Cached for the lifetime of the Streamlit process.
    """
    kb_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    raw_texts: list[str] = []

    for filepath in sorted(glob.glob(os.path.join(kb_dir, "*.txt"))):
        with open(filepath, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
            if content:
                raw_texts.append(content)

    if not raw_texts:
        raise FileNotFoundError(
            "No .txt files found in knowledge_base/. "
            "Make sure the knowledge_base directory exists and contains content."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.create_documents(raw_texts)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key,
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ---------------------------------------------------------------------------
# RAG chain factory — one chain per session, stored in st.session_state
# ---------------------------------------------------------------------------

def create_rag_chain(openai_api_key: str, vectorstore: FAISS) -> ConversationalRetrievalChain:
    """
    Build a ConversationalRetrievalChain with:
    - GPT-4o as the LLM
    - Custom Chiq IQ system prompt
    - 4 retrieved chunks per query
    - No internal memory object — history is passed explicitly from session_state
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=openai_api_key,
        temperature=0.7,
        max_tokens=700,
    )

    messages = [
        SystemMessagePromptTemplate.from_template(_SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.from_template(_HUMAN_TEMPLATE),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False,
    )
    return chain
