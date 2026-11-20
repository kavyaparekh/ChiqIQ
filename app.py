"""
Chiq IQ — AI Fashion Stylist
Two-tab Streamlit app:
  Tab 1: Conversational RAG stylist with closet-aware try-on
  Tab 2: My Closet & Try-On (upload photo + wardrobe, auto-describe with vision AI)
"""

import os
import re
import tempfile

import streamlit as st
from gradio_client import Client, handle_file

from rag_pipeline import (
    create_rag_chain, get_vectorstore, invoke_chain,
    describe_garment, pick_garments,
    search_aritzia, extract_aritzia_queries,
)

# ---------------------------------------------------------------------------
# Intent detection helpers
# ---------------------------------------------------------------------------

_YES = {"yes","yeah","yep","sure","go","show","please","definitely","absolutely",
        "course","ok","okay","yea","ya","do","love","want","let's","lets"}
_NO  = {"no","nah","nope","not","later","skip","never","don't","dont","maybe","pass"}
_MORE = {"more","other","else","different","another","next","others","alternatives",
         "rest","remaining","extra","additional"}
_SHOP_WORDS = {"aritzia","shop","buy","find","link","links","store","purchase",
               "where","get","available","stock","online"}

def _user_bubble(text: str):
    """Render a user chat message with the olive green bubble style."""
    st.markdown(f'<div class="msg-user">{text}</div>', unsafe_allow_html=True)


def _detect_intent(text: str) -> str:
    words = set(text.lower().split())
    if words & _MORE:  return "more"
    if words & _YES:   return "yes"
    if words & _NO:    return "no"
    return "unclear"


# ---------------------------------------------------------------------------
# Try-on helper (single garment → output image bytes)
# ---------------------------------------------------------------------------

def _run_tryon(user_photo_bytes: bytes, garment_bytes: bytes, garment_name: str, hf_tokens: list[str] | None = None) -> bytes:
    person_path = garment_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as pf:
            pf.write(user_photo_bytes)
            person_path = pf.name
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as gf:
            gf.write(garment_bytes)
            garment_path = gf.name

        tokens = hf_tokens or [""]
        spaces = [
            ("yisol/IDM-VTON", True),
            ("Nymbo/Virtual-Try-On", False),
        ]
        errors = []
        for i, token in enumerate(tokens):
            if token:
                os.environ["HF_TOKEN"] = token
            for space, crop in spaces:
                try:
                    client = Client(space)
                    result = client.predict(
                        dict={"background": handle_file(person_path), "layers": [], "composite": None},
                        garm_img=handle_file(garment_path),
                        garment_des=garment_name,
                        is_checked=True,
                        is_checked_crop=crop,
                        denoise_steps=30,
                        seed=42,
                        api_name="/tryon",
                    )
                    if result and result[0]:
                        with open(result[0], "rb") as f:
                            return f.read()
                    errors.append(f"token{i+1}/{space}: empty result")
                except Exception as e:
                    errors.append(f"token{i+1}/{space}: {e}")
                    continue

        raise RuntimeError("Try-on failed — " + " | ".join(errors))
    finally:
        for p in [person_path, garment_path]:
            if p:
                try: os.unlink(p)
                except OSError: pass

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ChiqIQ — AI Fashion Stylist",
    page_icon="✨",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS — clean light theme, fixed chat input, hidden Streamlit chrome
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500&family=Inter:wght@300;400;500;600&display=swap');

    /* ── Aritzia palette ── */
    :root {
        --green-deep:   #2C4A38;
        --green-mid:    #3D6150;
        --green-olive:  #6B8B74;
        --green-sage:   #8FAF98;
        --green-light:  #EEF2EF;
        --cream:        #F7F6F3;
        --warm-white:   #FDFCFA;
        --blush:        #F0EBE3;
        --taupe-light:  #E5E3DD;
        --taupe-mid:    #A89F91;
        --warm-gray:    #7A766F;
        --ink:          #222222;
        --white:        #FFFFFF;
    }

    /* ── Hide Streamlit chrome ── */
    header[data-testid="stHeader"] { display: none !important; }
    #MainMenu { visibility: hidden !important; }
    .stDeployButton { display: none !important; }
    footer { display: none !important; }

    /* ── Base ── */
    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"], section.main {
        background: linear-gradient(160deg, var(--warm-white) 0%, var(--cream) 55%, var(--blush) 100%) !important;
        background-attachment: fixed !important;
        color: var(--ink) !important;
        font-family: 'Inter', sans-serif !important;
    }
    p, span, label, li,
    .stMarkdown, .stMarkdown * {
        color: var(--ink) !important;
        font-family: 'Inter', sans-serif !important;
    }
    /* Exclude header divs from ink override */
    .chiq-title, .chiq-subtitle { font-family: inherit; }

    /* ════════════════════════════
       FIXED HEADER — Aritzia deep green
    ════════════════════════════ */
    .chiq-header {
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 1000;
        background: var(--green-deep);
        padding: clamp(0.8rem, 2vw, 1.2rem) clamp(1rem, 5vw, 3rem) 0;
    }
    .chiq-title {
        font-family: 'Cormorant Garamond', serif;
        font-weight: 700;
        font-size: clamp(1.5rem, 4vw, 2rem);
        letter-spacing: 0.06em;
        color: var(--white) !important;
        margin: 0 0 2px 0;
        line-height: 1;
    }
    .chiq-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: clamp(0.55rem, 1.2vw, 0.65rem);
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.65) !important;
        margin: 0 0 0.8rem 0;
    }

    /* ── Layout ── */
    .block-container {
        padding-top: clamp(90px, 12vw, 110px) !important;
        padding-bottom: 110px !important;
        padding-left: clamp(1rem, 4vw, 3rem) !important;
        padding-right: clamp(1rem, 4vw, 3rem) !important;
        max-width: 860px !important;
        margin: 0 auto !important;
    }

    /* ── Tabs — sticky just below header ── */
    .stTabs [data-baseweb="tab-list"] {
        position: sticky !important;
        top: clamp(72px, 10vw, 86px) !important;
        z-index: 999 !important;
        background-color: var(--cream) !important;
        gap: clamp(1.2rem, 3vw, 2.5rem) !important;
        border-bottom: 1px solid var(--taupe-light) !important;
        padding-top: 0.5rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-size: clamp(0.62rem, 1.3vw, 0.7rem) !important;
        letter-spacing: 0.16em !important;
        text-transform: uppercase !important;
        color: var(--warm-gray) !important;
        font-weight: 500 !important;
        background: transparent !important;
        padding-bottom: 0.65rem !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--green-deep) !important;
        background: transparent !important;
    }
    /* Sliding indicator — override Streamlit's default red/accent */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--green-deep) !important;
        height: 2px !important;
    }
    /* Bottom border line under all tabs */
    .stTabs [data-baseweb="tab-border"] {
        background-color: var(--taupe-light) !important;
        height: 1px !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: transparent !important;
        padding-top: 1rem !important;
    }

    /* ── Chat messages — hide all avatars ── */
    [data-testid="stChatMessageAvatar"],
    [data-testid="stChatMessage"] > div:first-child,
    .stChatMessageAvatar { display: none !important; }

    /* Assistant messages — blush */
    [data-testid="stChatMessage"] {
        background-color: var(--blush) !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 1rem 1.2rem !important;
        margin: 0 0 0.75rem 0 !important;
        box-shadow: 0 2px 10px rgba(44, 74, 56, 0.07), 0 1px 3px rgba(44, 74, 56, 0.04) !important;
    }

    /* User messages — olive green, strip outer container */
    [data-testid="stChatMessage"]:has(.msg-user) {
        background-color: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        border: none !important;
    }
    .msg-user {
        background-color: var(--green-sage) !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
        padding: 1rem 1.2rem !important;
        width: 100%;
        box-sizing: border-box;
        line-height: 1.72;
        font-family: 'Inter', sans-serif;
        font-size: clamp(0.82rem, 1.8vw, 0.9rem);
    }
    .msg-user * { color: #FFFFFF !important; }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] strong {
        color: var(--ink) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: clamp(0.82rem, 1.8vw, 0.9rem) !important;
        line-height: 1.72 !important;
    }

    /* ════════════════════════════
       FIXED INPUT BAR — blends into page, no visible footer band
    ════════════════════════════ */
    [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 0 !important; left: 0 !important; right: 0 !important;
        z-index: 9999 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: clamp(0.6rem, 2vw, 1rem) clamp(1rem, 5vw, 3rem) clamp(0.8rem, 2vw, 1.1rem) !important;
    }
    /* Strip every inner Streamlit wrapper */
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="base-input"],
    [data-testid="stChatInput"] [class*="InputContainer"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        border-radius: 0 !important;
    }
    /* Textarea — pill-shaped, olive green */
    [data-testid="stChatInput"] textarea {
        background-color: var(--green-olive) !important;
        color: #FFFFFF !important;
        caret-color: #FFFFFF !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.75rem 1.4rem !important;
        font-family: 'Inter', sans-serif !important;
        font-size: clamp(0.8rem, 1.8vw, 0.88rem) !important;
        font-weight: 300 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: rgba(255,255,255,0.55) !important;
        font-style: italic !important;
        font-weight: 300 !important;
    }

    /* ── Clear chat button — minimal underline style ── */
    :not([data-testid="stFileUploaderDropzone"]) > button[kind="secondary"],
    :not([data-testid="stFileUploaderDropzone"]) > div > button[kind="secondary"] {
        background-color: transparent !important;
        color: var(--warm-gray) !important;
        border: none !important;
        border-bottom: 1px solid var(--taupe-light) !important;
        border-radius: 0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.68rem !important;
        letter-spacing: 0.16em !important;
        text-transform: uppercase !important;
        padding: 0.2rem 0 !important;
        font-weight: 400 !important;
        box-shadow: none !important;
    }
    :not([data-testid="stFileUploaderDropzone"]) > button[kind="secondary"]:hover {
        background-color: transparent !important;
        color: var(--green-deep) !important;
        border-bottom-color: var(--green-deep) !important;
    }

    /* ── Try-On page — file uploader ── */
    [data-testid="stFileUploader"] {
        background-color: transparent !important;
        border: none !important;
    }
    /* Outer dropzone box — same blush as the text input below */
    [data-testid="stFileUploaderDropzone"] {
        background-color: var(--blush) !important;
        border: 1px solid var(--taupe-light) !important;
        border-radius: 4px !important;
        color: var(--ink) !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] * {
        color: var(--warm-gray) !important;
        font-size: 0.82rem !important;
    }
    /* "Browse files" button — olive green pill */
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzone"] [data-testid="baseButton-secondary"] {
        background-color: var(--green-olive) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 4px !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.08em !important;
        padding: 0.4rem 1.2rem !important;
    }

    /* ── Text input (garment description) ── */
    .stTextInput input,
    .stTextInput > div > div {
        background-color: var(--blush) !important;
        border: 1px solid var(--taupe-light) !important;
        border-radius: 4px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        color: var(--ink) !important;
    }
    .stTextInput input::placeholder { color: var(--taupe-mid) !important; }

    /* ── "How does this work?" info block ── */
    .how-it-works {
        margin-top: 1.4rem;
        background-color: var(--blush);
        border: 1px solid var(--taupe-light);
        border-radius: 4px;
        padding: 1rem 1.2rem;
        font-size: 0.83rem;
        line-height: 1.7;
        color: var(--warm-gray) !important;
    }
    .how-it-works strong { color: var(--ink) !important; }
    .hiw-title {
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--green-deep) !important;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }

    /* ── Info banner ── */
    .api-notice {
        border-left: 2px solid var(--green-olive);
        padding: 0.6rem 1rem;
        font-size: 0.8rem;
        color: var(--warm-gray) !important;
        margin-bottom: 1.5rem;
    }
    .api-notice a { color: var(--green-deep) !important; }

    /* ════════════════════════════
       RESPONSIVE BREAKPOINTS
    ════════════════════════════ */

    /* Tablet */
    @media (max-width: 768px) {
        .chiq-header { padding: 0.8rem 1.2rem 0; }
        .chiq-title  { font-size: 1.5rem; }
        .block-container {
            padding-top: 90px !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            top: 68px !important;
            gap: 1.4rem !important;
        }
        [data-testid="stChatInput"] {
            padding: 0.6rem 1rem !important;
        }
        /* Stack try-on columns */
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
        [data-testid="stHorizontalBlock"] > div {
            width: 100% !important;
            min-width: unset !important;
        }
    }

    /* Mobile */
    @media (max-width: 480px) {
        .chiq-title  { font-size: 1.3rem; }
        .chiq-subtitle { font-size: 0.55rem; }
        .block-container {
            padding-top: 80px !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-bottom: 90px !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            top: 60px !important;
            gap: 1rem !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.6rem !important;
            letter-spacing: 0.1em !important;
        }
        [data-testid="stChatMessage"] p { font-size: 0.82rem !important; }
        [data-testid="stChatInput"] { padding: 0.5rem 0.75rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="chiq-header">
        <div class="chiq-title">ChiqIQ</div>
        <div class="chiq-subtitle">Your Personal AI Fashion Stylist</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

NVIDIA_KEY = st.secrets.get("nvidia_api_key", "")
REPLICATE_KEY = st.secrets.get("replicate_api_key", "")
HF_TOKENS = [
    t for t in [
        st.secrets.get("hf_token1", ""),
        st.secrets.get("hf_token2", ""),
        st.secrets.get("hf_token3", ""),
    ] if t
]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_chat, tab_tryon = st.tabs(["💬  Style Chat", "👗  My Closet & Try-On"])


# ===========================================================================
# TAB 1 — CONVERSATIONAL RAG STYLIST
# ===========================================================================

with tab_chat:
    if not NVIDIA_KEY:
        st.markdown(
            """
            <div class="api-notice">
            <strong>NVIDIA API key not set.</strong><br>
            Add <code>nvidia_api_key = "nvapi-..."</code> to <code>.streamlit/secrets.toml</code>,
            then restart the app.<br>
            Get a free key at <a href="https://build.nvidia.com" target="_blank">build.nvidia.com</a>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    # --- Initialize vectorstore and RAG chain (once per session) ---
    if "vectorstore" not in st.session_state:
        try:
            st.session_state.vectorstore = get_vectorstore(NVIDIA_KEY)
        except Exception as e:
            st.error(f"Failed to build knowledge base: {e}")
            st.stop()

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = create_rag_chain(
            NVIDIA_KEY, st.session_state.vectorstore
        )

    # --- Session state init ---
    _WELCOME = (
        "Hi! I'm ChiqIQ — your personal AI stylist. ✨\n\n"
        "Tell me about an occasion, the weather where you are, "
        "or anything about your style and I'll put together the perfect outfit for you."
    )
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": _WELCOME, "type": "text"}]
    if "chat_history_pairs" not in st.session_state:
        st.session_state.chat_history_pairs = []
    if "awaiting_tryon" not in st.session_state:
        st.session_state.awaiting_tryon = False
    if "tryon_offset" not in st.session_state:
        st.session_state.tryon_offset = 0
    if "awaiting_shop_details" not in st.session_state:
        st.session_state.awaiting_shop_details = False

    closet = st.session_state.get("closet", [])
    user_photo = st.session_state.get("user_photo_bytes")
    closet_names = [item["name"] for item in closet]

    # --- Clear button ---
    if len(st.session_state.messages) > 1:
        _, btn_col = st.columns([6, 1])
        with btn_col:
            if st.button("Clear", key="clear_btn"):
                st.session_state.messages = [{"role": "assistant", "content": _WELCOME, "type": "text"}]
                st.session_state.chat_history_pairs = []
                st.session_state.awaiting_tryon = False
                st.session_state.tryon_offset = 0
                st.session_state.rag_chain = create_rag_chain(NVIDIA_KEY, st.session_state.vectorstore)
                st.rerun()

    # --- Try-on waiting hint ---
    if st.session_state.awaiting_tryon:
        if closet and user_photo:
            st.markdown(
                "<div style='font-size:0.75rem;color:#6B8B74;letter-spacing:0.08em;margin-bottom:0.5rem'>"
                "✦ Say <strong>yes</strong> to generate try-on visuals from your closet, "
                "or <strong>show me more</strong> for the next batch."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='font-size:0.75rem;color:#A89F91;letter-spacing:0.08em;margin-bottom:0.5rem'>"
                "✦ Upload your photo &amp; wardrobe in <strong>My Closet &amp; Try-On</strong>, "
                "then say <strong>yes</strong> to visualize."
                "</div>",
                unsafe_allow_html=True,
            )

    # --- Render chat history ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                _user_bubble(msg["content"])
            else:
                st.markdown(msg["content"])
            if msg.get("type") == "tryon" and msg.get("images"):
                img_cols = st.columns(len(msg["images"]))
                for j, (name, img_bytes) in enumerate(msg["images"]):
                    with img_cols[j]:
                        st.image(img_bytes, caption=name, use_container_width=True)
            if msg.get("aritzia_links"):
                st.markdown("---\n**Shop these looks at Aritzia:**")
                for link in msg["aritzia_links"]:
                    st.markdown(f"- [{link['title']}]({link['url']})")

    # --- Chat input ---
    if user_input := st.chat_input("e.g., Christmas party, cold outside, I want to look elegant"):
        handled = False

        # ── Try-on intent flow ──────────────────────────────────────────────
        # Only check try-on intent for short replies (≤5 words) — anything longer
        # is regular conversation, not a yes/no response to the try-on offer.
        if st.session_state.awaiting_tryon and len(user_input.split()) <= 5:
            intent = _detect_intent(user_input)

            if intent in ("yes", "more"):
                handled = True
                offset = st.session_state.tryon_offset if intent == "more" else 0

                # Show user message
                st.session_state.messages.append({"role": "user", "content": user_input, "type": "text"})
                with st.chat_message("user"):
                    _user_bubble(user_input)

                if not closet or not user_photo:
                    missing = []
                    if not user_photo: missing.append("your photo")
                    if not closet: missing.append("your wardrobe items")
                    reply = (
                        f"To generate try-on visuals I need {' and '.join(missing)} — "
                        "head to the **My Closet & Try-On** tab to upload them, "
                        "then come back and say **yes** again!"
                    )
                    st.session_state.messages.append({"role": "assistant", "content": reply, "type": "text"})
                    with st.chat_message("assistant"):
                        st.markdown(reply)
                    # Keep awaiting_tryon = True so user can upload and say yes again
                else:
                    # Pick garments
                    context_window = st.session_state.chat_history_pairs[-4:]
                    context_text = " ".join(f"{h}" for h, _ in context_window)

                    with st.spinner("Selecting the best looks from your closet…"):
                        selected = pick_garments(NVIDIA_KEY, context_text, closet, n=3, offset=offset)

                    # Generate try-ons
                    generated: list[tuple[str, bytes]] = []
                    last_error = ""
                    for i, garment in enumerate(selected):
                        with st.spinner(f"Generating look {i+1} of {len(selected)}: {garment['name']}…"):
                            try:
                                img_bytes = _run_tryon(user_photo, garment["bytes"], garment["name"], HF_TOKENS)
                                generated.append((garment["name"], img_bytes))
                            except Exception as e:
                                last_error = str(e)

                    if not selected:
                        st.session_state.messages.append({"role": "assistant", "content": "No closet items found — please re-upload your wardrobe in the **My Closet & Try-On** tab.", "type": "text"})
                        with st.chat_message("assistant"):
                            st.markdown("No closet items found — please re-upload your wardrobe in the **My Closet & Try-On** tab.")
                        st.rerun()

                    remaining = len(closet) - (offset + 3)
                    if generated:
                        caption = f"Here are {len(generated)} looks from your closet for this occasion:"
                        if remaining > 0:
                            caption += f"\n\n*Say 'show me more' to see {min(remaining, 3)} more options.*"
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": caption,
                            "type": "tryon",
                            "images": generated,
                        })
                        with st.chat_message("assistant"):
                            st.markdown(caption)
                            img_cols = st.columns(len(generated))
                            for j, (name, img_bytes) in enumerate(generated):
                                with img_cols[j]:
                                    st.image(img_bytes, caption=name, use_container_width=True)
                    else:
                        err = f"Try-on generation failed: {last_error}" if last_error else "Couldn't connect to the HuggingFace Space — try again in a moment."
                        st.session_state.messages.append({"role": "assistant", "content": err, "type": "text"})
                        with st.chat_message("assistant"):
                            st.markdown(err)

                    st.session_state.tryon_offset = offset + 3

            elif intent == "no":
                st.session_state.awaiting_tryon = False
                # fall through to normal chat

        # ── Normal chat ─────────────────────────────────────────────────────
        if not handled:
            st.session_state.messages.append({"role": "user", "content": user_input, "type": "text"})
            with st.chat_message("user"):
                _user_bubble(user_input)

            # Detect shopping intent
            _user_words = set(user_input.lower().split())
            _wants_links = bool(_user_words & _SHOP_WORDS)

            # Resuming from "need size/budget" prompt
            _run_shop_search = st.session_state.awaiting_shop_details
            if _run_shop_search:
                st.session_state.awaiting_shop_details = False
                _wants_links = False  # don't re-trigger the size check

            # Check if size is already known in conversation history
            _all_history = " ".join(h + " " + a for h, a in st.session_state.chat_history_pairs)
            _all_history += " " + user_input
            _has_size = bool(re.search(
                r'\b(xs|x-small|s\b|small\b|m\b|medium\b|l\b|large\b|xl\b|xxl\b'
                r'|size\s+\d+|\b[02468]\b)\b',
                _all_history.lower()
            ))

            # When user asks for links: ask refinement questions first (via LLM), then search
            _ask_for_details = _wants_links and not _run_shop_search

            aritzia_links = []
            with st.chat_message("assistant"):
                if _ask_for_details:
                    # Inject a note so LLM asks 1-2 focused refinement questions
                    _shop_q = (
                        user_input + "\n\n"
                        "[Note: The user wants to shop for specific items. Before searching, "
                        "ask them 1-2 short focused questions about only what hasn't been discussed: "
                        "colour preference, style or length (mini/midi/maxi/crop/long), size, or budget. "
                        "Do not provide links yet. Ask naturally and conversationally.]"
                    )
                    with st.spinner("Styling your look…"):
                        try:
                            answer = invoke_chain(
                                st.session_state.rag_chain,
                                _shop_q,
                                st.session_state.chat_history_pairs,
                                closet_names=closet_names if closet_names else None,
                                photo_available=bool(user_photo),
                            )
                        except Exception as e:
                            answer = f"Something went wrong: {e}"
                    st.markdown(answer)
                    st.session_state.awaiting_shop_details = True
                else:
                    with st.spinner("Styling your look…"):
                        try:
                            answer = invoke_chain(
                                st.session_state.rag_chain,
                                user_input,
                                st.session_state.chat_history_pairs,
                                closet_names=closet_names if closet_names else None,
                                photo_available=bool(user_photo),
                            )
                        except Exception as e:
                            answer = f"Something went wrong: {e}"
                    st.markdown(answer)

                _is_recommendation = (
                    len(answer) > 120
                    and not answer.rstrip().endswith("?")
                    and not _ask_for_details
                )

                # Nudge to upload closet after a recommendation
                if _is_recommendation and not (closet and user_photo):
                    st.markdown(
                        "<div style='font-size:0.8rem;color:#6B8B74;margin-top:0.75rem;font-style:italic'>"
                        "✦ Upload your photo and wardrobe in <strong>My Closet &amp; Try-On</strong> "
                        "to see how these looks would look on you."
                        "</div>",
                        unsafe_allow_html=True,
                    )

                # Aritzia search — only when user has asked and size is known, or resuming
                if _run_shop_search:
                    with st.spinner("Finding these at Aritzia…"):
                        try:
                            # Use full recent conversation so refinement details (colour, size etc) are included
                            context_text = " ".join(
                                h + " " + a for h, a in st.session_state.chat_history_pairs[-4:]
                            ) + " " + user_input + " " + answer
                            queries = extract_aritzia_queries(NVIDIA_KEY, context_text)
                            for q in queries:
                                result = search_aritzia(q)
                                if result:
                                    aritzia_links.append(result)
                        except Exception:
                            pass

                if aritzia_links:
                    st.markdown("---\n**Shop at Aritzia:**")
                    for link in aritzia_links:
                        st.markdown(f"- [{link['title']}]({link['url']})")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "type": "text",
                "aritzia_links": aritzia_links,
            })
            st.session_state.chat_history_pairs.append((user_input, answer))

            # Set try-on flag only when bot actually asked the try-on question
            # (which only happens when closet is uploaded, via _CLOSET_SECTION_TEMPLATE)
            _offer_phrases = ("would you like to see", "closet pieces would look", "see how some of your")
            if any(p in answer.lower() for p in _offer_phrases):
                st.session_state.awaiting_tryon = True
                st.session_state.tryon_offset = 0


# ===========================================================================
# TAB 2 — MY CLOSET & TRY-ON
# ===========================================================================

with tab_tryon:

    # ── SECTION 1: My Photo ─────────────────────────────────────────────────
    st.subheader("My Photo")
    st.caption("Upload a clear, front-facing photo. Shared with the Style Chat for try-on visuals.")

    photo_col, photo_preview = st.columns([2, 1], gap="large")
    with photo_col:
        photo_file = st.file_uploader(
            "Upload your photo",
            type=["jpg", "jpeg", "png", "webp"],
            key="photo_upload",
            label_visibility="collapsed",
        )
        if photo_file:
            st.session_state.user_photo_bytes = photo_file.read()

    with photo_preview:
        if st.session_state.get("user_photo_bytes"):
            st.image(st.session_state.user_photo_bytes, width=160)
            st.caption("✓ Photo saved")

    st.divider()

    # ── SECTION 2: My Closet ────────────────────────────────────────────────
    st.subheader("My Closet")
    st.caption(
        "Upload your clothing items. The AI will auto-describe each one using vision AI — "
        "these descriptions are used by the stylist to recommend and generate looks."
    )

    closet_files = st.file_uploader(
        "Upload garment images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="closet_upload",
        label_visibility="collapsed",
    )

    if closet_files:
        if st.button("Save & Auto-describe Closet", type="primary", key="save_closet_btn"):
            new_closet = []
            prog = st.progress(0, text="Describing your wardrobe…")
            for i, f in enumerate(closet_files):
                img_bytes = f.read()
                prog.progress((i) / len(closet_files), text=f"Describing item {i+1} of {len(closet_files)}…")
                try:
                    desc = describe_garment(img_bytes, NVIDIA_KEY)
                except Exception:
                    desc = f.name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
                new_closet.append({"name": desc, "bytes": img_bytes})
            prog.progress(1.0, text="Done!")
            st.session_state.closet = new_closet
            st.success(f"✓ {len(new_closet)} items saved to your closet.")

    # Show closet grid
    if st.session_state.get("closet"):
        closet = st.session_state.closet
        st.markdown(f"**{len(closet)} items in your closet:**")
        grid_cols = st.columns(4)
        for i, item in enumerate(closet):
            with grid_cols[i % 4]:
                st.image(item["bytes"], use_container_width=True)
                st.caption(item["name"])

        if st.button("Clear closet", key="clear_closet_btn"):
            st.session_state.closet = []
            st.rerun()

    st.divider()

    # ── SECTION 3: One-off Try-On ───────────────────────────────────────────
    st.subheader("Try On a Specific Item")
    st.caption("Pick any single garment and generate a try-on preview instantly.")

    if not st.session_state.get("user_photo_bytes"):
        st.markdown(
            "<div class='api-notice'>Upload your photo above first.</div>",
            unsafe_allow_html=True,
        )
    else:
        closet_now = st.session_state.get("closet", [])
        if closet_now:
            garment_choice = st.selectbox(
                "Choose a garment from your closet",
                options=range(len(closet_now)),
                format_func=lambda i: closet_now[i]["name"],
                key="garment_select",
            )
            selected_garment = closet_now[garment_choice]
            st.image(selected_garment["bytes"], width=200)
        else:
            # Allow direct upload if closet is empty
            single_garment_file = st.file_uploader(
                "Upload a garment image",
                type=["jpg", "jpeg", "png", "webp"],
                key="single_garment_upload",
                label_visibility="collapsed",
            )
            if single_garment_file:
                selected_garment = {"name": "uploaded garment", "bytes": single_garment_file.read()}
                st.image(selected_garment["bytes"], width=200)
            else:
                selected_garment = None

        tryon_btn = st.button(
            "✨  Try it on",
            type="primary",
            disabled=(not st.session_state.get("user_photo_bytes") or not locals().get("selected_garment")),
            key="tryon_btn",
        )

        if tryon_btn and selected_garment:
            with st.spinner("Generating try-on… this takes 1–2 minutes"):
                try:
                    result_bytes = _run_tryon(
                        st.session_state.user_photo_bytes,
                        selected_garment["bytes"],
                        selected_garment["name"],
                        HF_TOKENS,
                    )
                    st.success("Here's how it looks on you!")
                    st.image(result_bytes, use_container_width=True)
                    st.download_button(
                        label="Download",
                        data=result_bytes,
                        file_name="chiqiq_tryon.jpg",
                        mime="image/jpeg",
                    )
                except Exception as e:
                    st.error(f"Try-on failed: {e}\n\nThe HuggingFace Space may be busy — try again in a moment.")

    st.markdown(
        """
        <div class="how-it-works">
            <div class="hiw-title">How does this work?</div>
            Upload your photo and wardrobe items above. The AI vision model automatically describes
            each garment so the stylist can recommend your actual clothes. When chatting, the bot will
            offer to generate try-on visuals — it picks the 3 most occasion-appropriate items from
            your closet and shows how they'd look on you, right inside the chat.<br><br>
            <strong>Tips for best results:</strong><br>
            · Use a clear, well-lit, front-facing photo of yourself<br>
            · Use garment photos with a plain or white background<br>
            · Full-body or upper-body shots work better than close-ups<br>
            · The model works best with tops, dresses, and outerwear
        </div>
        """,
        unsafe_allow_html=True,
    )
