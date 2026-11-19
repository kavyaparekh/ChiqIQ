# Chiq IQ ✨ — AI Fashion Stylist

A conversational AI personal stylist and virtual try-on app built with Python, Streamlit, LangChain, GPT-4o, and a Replicate-powered diffusion model.

---

## Features

**Style Chat (Tab 1)**
- RAG pipeline grounded in a curated fashion knowledge base: seasonal styling, occasion dressing, color theory, body type guides, and Aritzia-inspired elevated minimalism
- Multi-turn conversation memory — the stylist remembers context across the session
- Powered by GPT-4o via LangChain's `ConversationalRetrievalChain` with FAISS vector search

**Virtual Try-On (Tab 2)**
- Upload your portrait and a garment image
- Generates a photorealistic preview of you wearing the outfit
- Powered by IDM-VTON (Improving Diffusion Models for Authentic Virtual Try-on in the Wild) on Replicate

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-3-small |
| Vector store | FAISS (local, no server needed) |
| RAG orchestration | LangChain ConversationalRetrievalChain |
| Virtual try-on | Replicate — cuuupid/idm-vton |
| Language | Python 3.11+ |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/kavyaparekh/ChiqIQ.git
cd ChiqIQ
pip install -r requirements.txt
```

### 2. Add API keys

Create `.streamlit/secrets.toml`:

```toml
openai_api_key = "sk-..."        # https://platform.openai.com/api-keys
replicate_api_key = "r8_..."     # https://replicate.com/account/api-tokens
```

### 3. Run

```bash
streamlit run app.py
```

---

## Knowledge Base

The RAG pipeline indexes 7 fashion articles stored as plain text in `knowledge_base/`:

| File | Content |
|---|---|
| `01_seasonal_styling.txt` | Spring/Summer/Fall/Winter outfit guides |
| `02_occasion_dressing.txt` | Dress codes, weddings, interviews, events |
| `03_color_theory.txt` | Color wheel, skin tone matching, combinations |
| `04_body_type_styling.txt` | All body shapes and frame types |
| `05_holiday_festive.txt` | Holiday party and NYE styling |
| `06_wardrobe_essentials.txt` | Capsule wardrobe, essential pieces, shopping strategy |
| `07_aritzia_elevated_style.txt` | Elevated minimalism and Aritzia aesthetic |

The knowledge base is designed to be extended — drop in new `.txt` files and restart the app.
