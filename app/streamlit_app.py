# app/streamlit_app.py
import os
import time

import requests
import streamlit as st

# FastAPI backend URL
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")


st.set_page_config(
    page_title="MentalLLM - Your AI Mental Health Assistant", page_icon="🧠"
)
st.title("🧠 MentalLLM")
st.markdown("Your local AI companion for mental well-being, tuned with care.")
st.markdown("---")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Share your thoughts or ask for advice..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare request to FastAPI
    api_url = f"{FASTAPI_URL}/chat"
    payload = {"prompt": prompt}

    # Display assistant response placeholder
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Send request to FastAPI backend
            with st.spinner("Thinking..."):
                response = requests.post(api_url, json=payload, timeout=120)

            if response.status_code == 200:
                data = response.json()
                full_response = data.get("response", "Sorry, I couldn't process that.")
                # Simulate streaming word by word for better UX (optional, since FastAPI doesn't stream here)
                for word in full_response.split():
                    full_response += word + " "
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.05)
                message_placeholder.markdown(full_response)
            else:
                full_response = (
                    f"Error: API returned status code {response.status_code}"
                )
                try:
                    error_detail = response.json().get("detail", "Unknown error")
                    full_response += f" - {error_detail}"
                except Exception:
                    full_response += f" - {response.text}"
                message_placeholder.error(full_response)
                st.error(f"Full response: {response.text}")

        except requests.exceptions.RequestException as e:
            full_response = f"Error connecting to the MentalLLM API: {e}"
            message_placeholder.error(full_response)
            st.error(f"Exception details: {e}")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar for information
with st.sidebar:
    st.header("About MentalLLM")
    st.markdown(
        """
    This demo connects to a locally running FastAPI backend which serves a fine-tuned LLM for mental health support.

    **Tech Stack:**
    - **Fine-tuning:** Unsloth, QLoRA
    - **Model:** Phi-3 Mini (GGUF)
    - **Inference:** Ollama
    - **Backend:** FastAPI
    - **Frontend:** Streamlit
    - **Deployment:** Docker Compose
    """
    )
    st.markdown("---")
    st.subheader("How it Works")
    st.markdown(
        """
    1. You type a message related to mental well-being.
    2. Streamlit sends the message to the FastAPI backend.
    3. FastAPI calls the Ollama service (running the GGUF model).
    4. The model generates a response.
    5. The response travels back: Ollama -> FastAPI -> Streamlit -> You!
    """
    )
    st.markdown("---")
    st.subheader("Disclaimer")
    st.markdown(
        """
    This AI assistant is for informational and supportive purposes only. It is not a substitute for professional mental health advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. If you are in a crisis or contemplating self-harm, please call a local emergency number or go to the nearest emergency room.
    """
    )
    st.markdown("---")
    st.markdown(f"**Backend URL:** `{FASTAPI_URL}`")
