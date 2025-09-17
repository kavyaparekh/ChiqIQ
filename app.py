import streamlit as st
import requests

# --- Page Configuration ---
st.set_page_config(
    page_title="Chiq Iq - AI Fashion Assistant",
    page_icon="✨",
    layout="centered"
)

# --- Title and Header ---
st.title("Chiq Iq ✨")
st.caption("Your Personal AI Fashion Assistant")

# --- API Functions ---

def get_vext_ai_response(user_prompt, vext_api_url, vext_api_key):
    """
    Fetches the fashion suggestion from your Vext AI API.
    """
    if not vext_api_url or not vext_api_key:
        st.error("Vext AI API URL or Key is not configured in secrets.toml.")
        return "Sorry, my connection to the fashion brain is not set up right now."

    # Updated headers based on the Vext AI screenshot
    headers = {
        "Content-Type": "application/json",
        "Apikey": f"Api-Key {vext_api_key}"
    }
    payload = {"payload": user_prompt}

    try:
        response = requests.post(vext_api_url, headers=headers, json=payload)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        # Based on your screenshots, the response text is in the 'text' key
        return data.get("text", "I'm not sure what to say. The API response was unexpected.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to Vext AI: {e}")
        return "I'm having trouble coming up with ideas right now. Please try again later."

# --- Streamlit Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hi! What's the occasion you're dressing for? I'll help you find the perfect outfit."
    }]

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("e.g., A casual beach day"):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get API keys from Streamlit secrets
    VEXT_API_URL = st.secrets.get("vext_api_url", "")
    VEXT_API_KEY = st.secrets.get("vext_api_key", "")

    # Display assistant's thinking message and generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            text_response = get_vext_ai_response(prompt, VEXT_API_URL, VEXT_API_KEY)
            st.write(text_response)
    
    # Add the complete assistant response to the session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": text_response
    })

