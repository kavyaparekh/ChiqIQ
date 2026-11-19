"""
Chiq IQ — AI Fashion Stylist
Two-tab Streamlit app:
  Tab 1: Conversational RAG stylist (LangChain + GPT-4o + FAISS)
  Tab 2: Virtual Try-On (Replicate IDM-VTON diffusion model)
"""

import io
import os

import replicate
import streamlit as st
from PIL import Image

from rag_pipeline import create_rag_chain, get_vectorstore

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Chiq IQ — AI Fashion Stylist",
    page_icon="✨",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — Aritzia-inspired minimal aesthetic
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Background */
    .stApp { background-color: #FAFAF8; }

    /* Hide Streamlit default top padding */
    .block-container { padding-top: 2rem; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 1px solid #E8E5E0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #888;
        font-weight: 500;
        padding-bottom: 0.75rem;
    }
    .stTabs [aria-selected="true"] {
        color: #1A1A1A !important;
        border-bottom: 2px solid #1A1A1A !important;
    }

    /* Chat bubble overrides */
    .stChatMessage { background-color: transparent; }

    /* Buttons */
    .stButton > button {
        background-color: #1A1A1A;
        color: #FAFAF8;
        border: none;
        border-radius: 2px;
        font-size: 0.8rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.6rem 1.5rem;
    }
    .stButton > button:hover { background-color: #333; }

    /* Info banner */
    .api-notice {
        background-color: #F0EDE8;
        border-left: 3px solid #C8A882;
        padding: 0.75rem 1rem;
        border-radius: 2px;
        font-size: 0.82rem;
        color: #5A4A3A;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Chiq IQ ✨")
st.caption("Your Personal AI Fashion Stylist")

# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------

OPENAI_KEY = st.secrets.get("openai_api_key", "")
REPLICATE_KEY = st.secrets.get("replicate_api_key", "")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_chat, tab_tryon = st.tabs(["💬  Style Chat", "👗  Virtual Try-On"])


# ===========================================================================
# TAB 1 — CONVERSATIONAL RAG STYLIST
# ===========================================================================

with tab_chat:
    if not OPENAI_KEY:
        st.markdown(
            """
            <div class="api-notice">
            <strong>OpenAI API key not set.</strong><br>
            Add <code>openai_api_key = "sk-..."</code> to <code>.streamlit/secrets.toml</code>,
            then restart the app.<br>
            Get a key at <a href="https://platform.openai.com/api-keys" target="_blank">platform.openai.com/api-keys</a>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    # --- Initialize vectorstore and RAG chain (once per session) ---
    if "vectorstore" not in st.session_state:
        try:
            st.session_state.vectorstore = get_vectorstore(OPENAI_KEY)
        except Exception as e:
            st.error(f"Failed to build knowledge base: {e}")
            st.stop()

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = create_rag_chain(
            OPENAI_KEY, st.session_state.vectorstore
        )

    # --- Initialize chat history ---
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi! I'm Chiq IQ — your personal AI stylist. ✨\n\n"
                    "Tell me about an occasion, the weather where you are, "
                    "or anything about your style and I'll put together the perfect outfit for you."
                ),
            }
        ]

    if "chat_history_pairs" not in st.session_state:
        # LangChain expects [(human, ai), ...] tuples
        st.session_state.chat_history_pairs = []

    # --- Render chat history ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # --- Handle new input ---
    if user_input := st.chat_input("e.g., Christmas party, cold outside, I want to look elegant"):
        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Run RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Styling your look…"):
                try:
                    result = st.session_state.rag_chain.invoke(
                        {
                            "question": user_input,
                            "chat_history": st.session_state.chat_history_pairs,
                        }
                    )
                    answer = result["answer"]
                except Exception as e:
                    answer = f"Something went wrong: {e}"

                st.write(answer)

        # Persist to session state
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history_pairs.append((user_input, answer))

    # --- Clear chat button ---
    if len(st.session_state.messages) > 1:
        if st.button("Clear conversation"):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": (
                        "Hi! I'm Chiq IQ — your personal AI stylist. ✨\n\n"
                        "Tell me about an occasion, the weather where you are, "
                        "or anything about your style and I'll put together the perfect outfit for you."
                    ),
                }
            ]
            st.session_state.chat_history_pairs = []
            st.session_state.rag_chain = create_rag_chain(
                OPENAI_KEY, st.session_state.vectorstore
            )
            st.rerun()


# ===========================================================================
# TAB 2 — VIRTUAL TRY-ON
# ===========================================================================

with tab_tryon:
    st.subheader("See the outfit on you")
    st.caption(
        "Upload your photo and a garment image. "
        "The AI will generate a realistic preview of how the outfit looks on you."
    )

    if not REPLICATE_KEY:
        st.markdown(
            """
            <div class="api-notice">
            <strong>Replicate API key not set.</strong><br>
            Add <code>replicate_api_key = "r8_..."</code> to <code>.streamlit/secrets.toml</code>.<br>
            Get a free key at <a href="https://replicate.com/account/api-tokens" target="_blank">replicate.com</a>
            — no credit card needed to start.
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_person, col_garment = st.columns(2, gap="large")

    with col_person:
        st.markdown("**Your photo**")
        st.caption("Full-body or upper-body portrait works best.")
        person_file = st.file_uploader(
            "Upload your photo",
            type=["jpg", "jpeg", "png", "webp"],
            key="person_upload",
            label_visibility="collapsed",
        )
        if person_file:
            st.image(person_file, use_container_width=True)

    with col_garment:
        st.markdown("**Garment image**")
        st.caption("A flat-lay or model photo of the clothing item.")
        garment_file = st.file_uploader(
            "Upload garment",
            type=["jpg", "jpeg", "png", "webp"],
            key="garment_upload",
            label_visibility="collapsed",
        )
        if garment_file:
            st.image(garment_file, use_container_width=True)

    garment_desc = st.text_input(
        "Describe the garment (optional — helps the model)",
        placeholder="e.g., emerald green velvet midi dress with long sleeves",
    )

    st.divider()

    run_btn = st.button(
        "✨  Try it on",
        disabled=(not person_file or not garment_file or not REPLICATE_KEY),
    )

    if run_btn:
        with st.spinner("Generating your virtual try-on… this takes ~20–30 seconds"):
            try:
                os.environ["REPLICATE_API_TOKEN"] = REPLICATE_KEY

                person_bytes = io.BytesIO(person_file.read())
                garment_bytes = io.BytesIO(garment_file.read())

                output = replicate.run(
                    "cuuupid/idm-vton",
                    input={
                        "human_img": person_bytes,
                        "garm_img": garment_bytes,
                        "garment_des": garment_desc or "stylish outfit",
                        "is_checked": True,
                        "is_checked_crop": False,
                        "denoise_steps": 30,
                        "seed": 42,
                    },
                )

                # output is a URL string or a list; normalise to URL
                result_url = output[0] if isinstance(output, list) else output

                st.success("Here's how it looks on you! ✨")
                st.image(result_url, use_container_width=True)

                # Download button
                import requests as req

                img_data = req.get(result_url).content
                st.download_button(
                    label="Download image",
                    data=img_data,
                    file_name="chiqiq_tryon.jpg",
                    mime="image/jpeg",
                )

            except Exception as e:
                st.error(
                    f"Virtual try-on failed: {e}\n\n"
                    "Make sure your Replicate API key is valid and both images are clear, "
                    "front-facing photos."
                )

    # --- How it works note ---
    with st.expander("How does this work?"):
        st.write(
            """
            This feature uses **IDM-VTON** (Improving Diffusion Models for Authentic Virtual Try-on
            in the Wild), a state-of-the-art garment transfer model running on Replicate's cloud GPU.

            It takes your portrait and a garment photo, and generates a photorealistic image
            of you wearing that garment — preserving your face, body, and pose while accurately
            rendering the clothing's fabric, texture, and fit.

            **Tips for best results:**
            - Use a clear, well-lit, front-facing photo of yourself
            - Use a garment photo with a plain or white background
            - Full-body or upper-body shots work better than close-up face shots
            - The model works best with tops, dresses, and outerwear
            """
        )
