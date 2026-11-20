"""
RAG pipeline for Chiq IQ — LangChain LCEL + FAISS + NVIDIA NIM.
Includes closet-aware prompting, garment selection, and Aritzia product search.
"""

import base64
import json
import os
import re
import glob
from urllib.parse import quote, urlparse, parse_qs, unquote

import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """You are ChiqIQ, a sophisticated personal AI fashion stylist — \
think of yourself as a knowledgeable Aritzia personal shopper combined with a stylist.

YOUR CONVERSATION STYLE — this is critical:
- You are dialogue-driven. You build the perfect recommendation through conversation, not essays.
- When a user first describes an occasion or need: acknowledge warmly in one sentence, \
  then ask exactly ONE focused follow-up question. Ask about whatever is most relevant \
  and not yet answered: indoor vs outdoor, weather/temperature, formality level \
  (casual / smart-casual / formal), vibe (cozy, chic, edgy, romantic, festive, minimalist), \
  color preferences, fit preferences, or budget.
- Keep early responses SHORT — 2-3 sentences maximum plus one question.
- After 2-3 exchanges where you have enough context, give a specific complete outfit recommendation.
- Never write a long outfit essay on the very first response.
- After 2-3 exchanges where you have enough context, give a specific complete outfit recommendation \
  with 3-5 specific pieces. End the recommendation with a brief styling note, not a question.
- When a user asks to find or shop for items, do NOT say you cannot browse or search Aritzia. \
  Simply acknowledge their request and describe what to look for — links will be provided automatically.

YOUR EXPERTISE:
- Occasion-appropriate dressing (weddings, interviews, parties, casual, festive events)
- Seasonal styling and weather-appropriate outfit building for all four seasons
- Color theory: flattering combinations, undertone matching, color psychology
- Body type styling: how to dress for every shape and frame
- Elevated minimalist aesthetics — clean lines, quality fabrics, understated polish
- Wardrobe building: capsule wardrobes, essential pieces, investment vs. budget choices
{closet_section}
Use the following fashion knowledge to ground your recommendations:
{context}"""

_CLOSET_SECTION_TEMPLATE = """
The user has uploaded their personal wardrobe. Here are the items they own:
{closet_names}

When making recommendations, prioritise pieces from their actual wardrobe when suitable. \
If none of their items fit the occasion, let them know and suggest what to look for.
After delivering a complete outfit recommendation (not while asking clarifying questions), \
add this line once at the very end: \
"Would you like to see how some of your closet pieces would look styled for this occasion?"
"""


def _pairs_to_messages(pairs: list[tuple[str, str]]) -> list:
    messages = []
    for human, ai in pairs:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))
    return messages


# ---------------------------------------------------------------------------
# Vectorstore
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Building fashion knowledge base…")
def get_vectorstore(nvidia_api_key: str) -> FAISS:
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
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": True},
    )
    return FAISS.from_documents(chunks, embeddings)


# ---------------------------------------------------------------------------
# RAG chain factory
# ---------------------------------------------------------------------------

def create_rag_chain(nvidia_api_key: str, vectorstore: FAISS):
    llm = ChatNVIDIA(
        model="nvidia/nemotron-3.5-bolt-e4b",
        api_key=nvidia_api_key,
        temperature=0.7,
        max_tokens=700,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def invoke_chain(
    chain,
    question: str,
    history_pairs: list[tuple[str, str]],
    closet_names: list[str] | None = None,
    photo_available: bool = False,
) -> str:
    closet_section = ""
    if closet_names and photo_available:
        closet_section = _CLOSET_SECTION_TEMPLATE.format(
            closet_names=", ".join(closet_names)
        )

    return chain.invoke({
        "question": question,
        "chat_history": _pairs_to_messages(history_pairs),
        "closet_section": closet_section,
    })


# ---------------------------------------------------------------------------
# Vision: auto-describe a garment image
# ---------------------------------------------------------------------------

def describe_garment(image_bytes: bytes, nvidia_api_key: str) -> str:
    """Send a garment image to NVIDIA vision LLM and return a short description."""
    b64 = base64.b64encode(image_bytes).decode()

    llm = ChatNVIDIA(
        model="meta/llama-3.2-11b-vision-instruct",
        api_key=nvidia_api_key,
        max_tokens=60,
        temperature=0.1,
    )

    msg = HumanMessage(content=[
        {
            "type": "text",
            "text": (
                "Look carefully at this clothing item image. "
                "Step 1: Identify the exact garment type — is it a cardigan, blazer, jacket, coat, top, "
                "blouse, shirt, dress, skirt, pants, jeans, shorts, or something else? "
                "Step 2: Describe it in under 15 words including color and key style details. "
                "Examples:\n"
                "- 'Burgundy ribbed cardigan, button-front, long sleeves.'\n"
                "- 'Navy wide-leg trousers, high-waist, tailored cut.'\n"
                "- 'Cream oversized blazer, structured shoulders, single button.'\n"
                "- 'Black pleated mini skirt, A-line silhouette.'\n"
                "Reply with ONLY the description — no extra words, no preamble."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
    ])

    response = llm.invoke([msg])
    return response.content.strip()


# ---------------------------------------------------------------------------
# Garment selection: pick the N most occasion-appropriate closet items
# ---------------------------------------------------------------------------

def pick_garments(
    nvidia_api_key: str,
    conversation_context: str,
    closet: list[dict],
    n: int = 3,
    offset: int = 0,
) -> list[dict]:
    """Rank closet items by occasion suitability; return n items starting at offset."""
    names = [item["name"] for item in closet]

    llm = ChatNVIDIA(
        model="nvidia/nemotron-3.5-bolt-e4b",
        api_key=nvidia_api_key,
        max_tokens=300,
        temperature=0,
    )

    prompt = (
        "You are a fashion stylist. Rank these wardrobe items from most to least suitable "
        "for the occasion discussed in this conversation.\n\n"
        f"Conversation context:\n{conversation_context}\n\n"
        f"Wardrobe items: {json.dumps(names)}\n\n"
        "Reply with ONLY a JSON array of the item names in order of suitability (best first). "
        "Include every item. Example: [\"item1\", \"item2\", \"item3\"]"
    )

    try:
        response = llm.invoke(prompt)
        match = re.search(r'\[.*?\]', response.content, re.DOTALL)
        if match:
            ranked_names = json.loads(match.group())
            by_name = {item["name"]: item for item in closet}
            ordered = [by_name[n] for n in ranked_names if n in by_name]
            if ordered:
                return ordered[offset: offset + n]
    except Exception:
        pass

    # Fallback: return by upload order
    return closet[offset: offset + n]


# ---------------------------------------------------------------------------
# Aritzia product search via DuckDuckGo
# ---------------------------------------------------------------------------

def search_aritzia(query: str) -> dict:
    """Search DuckDuckGo for site:aritzia.com and return the best product result."""
    import requests as req
    from bs4 import BeautifulSoup

    search_url = f"https://html.duckduckgo.com/html/?q={quote('site:aritzia.com ' + query)}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    def _decode_href(href: str) -> str:
        if "uddg=" in href:
            try:
                params = parse_qs(urlparse(href).query)
                return unquote(params.get("uddg", [""])[0])
            except Exception:
                pass
        return href

    _product_paths = ("/clothing/", "/accessories/", "/shoes/", "/tna/",
                      "/wilfred/", "/sunday-best/", "/babaton/", "/contiguous/")

    try:
        resp = req.get(search_url, headers=headers, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")
        anchors = soup.select("a.result__a")

        # First pass: prefer product-page URLs
        for a in anchors:
            url = _decode_href(a.get("href", ""))
            if "aritzia.com" in url and any(p in url for p in _product_paths):
                return {"title": a.get_text(strip=True), "url": url}

        # Second pass: any aritzia.com URL
        for a in anchors:
            url = _decode_href(a.get("href", ""))
            if "aritzia.com/en/" in url:
                return {"title": a.get_text(strip=True), "url": url}

    except Exception:
        pass

    # Fallback: Aritzia search page
    return {
        "title": f"Search '{query}' on Aritzia",
        "url": f"https://www.aritzia.com/en/search?q={quote(query)}",
    }


def extract_aritzia_queries(nvidia_api_key: str, recommendation: str) -> list[str]:
    """Ask the LLM to pull 2-3 searchable item names from a recommendation."""
    llm = ChatNVIDIA(
        model="nvidia/nemotron-3.5-bolt-e4b",
        api_key=nvidia_api_key,
        max_tokens=80,
        temperature=0,
    )

    prompt = (
        "Extract 2-3 specific clothing or accessory items from this fashion recommendation "
        "that a customer could search for on Aritzia's website. "
        "Return ONLY a JSON array of short search phrases (2-5 words each). "
        "Example: [\"oversized blazer\", \"wide leg trousers\"]\n\n"
        f"Recommendation:\n{recommendation}"
    )

    try:
        response = llm.invoke(prompt)
        match = re.search(r'\[.*?\]', response.content, re.DOTALL)
        if match:
            queries = json.loads(match.group())
            return [q for q in queries if isinstance(q, str)][:3]
    except Exception:
        pass

    return []
