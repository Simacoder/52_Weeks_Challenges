import streamlit as st
import os
from dotenv import load_dotenv

def get_secret(key):
    """
    Get a secret from Streamlit or fallback to .env for local development.

    This allows the app to run both on Streamlit Cloud and locally.
    """
    try:
        return st.secrets[key]
    except Exception:
        load_dotenv()
        return os.getenv(key)
    
def reset_chat():
    """
    Reset the Streamlit chat session state.
    """
    st.session_state.chat_history = []
    st.session_state.example = False # Add others if needed