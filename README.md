# Chiq IQ ✨ — AI Fashion Stylist

A conversational AI personal stylist and virtual try-on app built with Python, Streamlit, LangChain, NVIDIA NIM, and IDM-VTON on HuggingFace Spaces.

---

## Features

**Style Chat (Tab 1)**
- Dialogue-driven RAG pipeline — asks focused questions before giving a full outfit recommendation
- Grounded in a curated fashion knowledge base: seasonal styling, occasion dressing, color theory, body type guides, and elevated minimalist aesthetics
- Multi-turn conversation memory across the session
- Closet-aware recommendations — if you've uploaded your wardrobe, the bot prioritises your actual pieces
- Aritzia product links — triggered automatically when you ask to shop or find items online; bot asks for colour, style, size, and budget before searching
- Try-on offer — when closet is uploaded, bot asks if you'd like to see your closet pieces styled for the occasion
- Powered by `nvidia/nemotron-3.5-bolt-e4b` via NVIDIA NIM and LangChain LCEL with FAISS vector search

**My Closet & Try-On (Tab 2)**
- Upload a front-facing portrait (used for all try-on generations)
- Upload multiple garment images — vision AI (`meta/llama-3.2-11b-vision-instruct`) auto-describes each item
- Closet grid view with AI-generated captions
- Chat-driven try-on: bot selects the 3 most occasion-appropriate closet items and generates photorealistic try-on previews
- One-off try-on: pick any single garment from your closet and generate instantly
- Powered by IDM-VTON on HuggingFace Spaces (`yisol/IDM-VTON`) with `Nymbo/Virtual-Try-On` as fallback
- Supports up to 3 HuggingFace tokens for quota rotation

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | `nvidia/nemotron-3.5-bolt-e4b` via NVIDIA NIM |
| Vision (garment description) | `meta/llama-3.2-11b-vision-instruct` via NVIDIA NIM |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, local) |
| Vector store | FAISS (local, no server needed) |
| RAG orchestration | LangChain LCEL |
| Virtual try-on | IDM-VTON on HuggingFace Spaces (`yisol/IDM-VTON`) |
| Product search | DuckDuckGo scrape via BeautifulSoup4 |
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
nvidia_api_key = "nvapi-..."     # https://build.nvidia.com  (free tier available)

# HuggingFace tokens for virtual try-on (ZeroGPU quota)
# Add up to 3 — the app rotates through them when one hits its limit
hf_token1 = "hf_..."            # https://huggingface.co/settings/tokens  (Read access)
hf_token2 = "hf_..."            # optional second token
hf_token3 = "hf_..."            # optional third token
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
| `07_aritzia_elevated_style.txt` | Elevated minimalism and polished everyday style |

Drop in new `.txt` files and restart the app to extend the knowledge base.

---

## Notes

- **ZeroGPU quota**: each try-on uses ~60–90s of GPU time. The free HuggingFace tier resets hourly. Add multiple `hf_token` keys to rotate automatically.
- **Garment descriptions**: vision AI describes each uploaded garment automatically on save. Best results with plain-background garment photos.
- **Try-on quality**: works best with clear, front-facing full-body or upper-body portraits and garments photographed on a white/plain background.
