import streamlit as st
import google.generativeai as genai
from functions import get_secret, reset_chat

# Inject custom CSS to increase font size throughout the app
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-size: 20px !important;
    }
    .stChatMessageContent {
        font-size: 20px !important;
    }
    .stTextInput > div > div > input {
        font-size: 20px !important;
    }
    .stSlider > div {
        font-size: 18px !important;
    }
    .stButton button {
        font-size: 18px !important;
        padding: 10px 20px;
    }
    .stSidebar {
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# API setup
api_key = get_secret("API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

# Sidebar configuration
temperature = st.sidebar.slider(
    label="Select the temperature",
    min_value=0.0,
    max_value=2.0,
    value=1.0
)

if st.sidebar.button("Reset chat"):
    reset_chat()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if not st.session_state.chat_history:
    st.session_state.chat_history.append(("assistant", "Hi! How can I help you?"))

# Display chat history
for role, message in st.session_state.chat_history:
    st.chat_message(role).write(message)

# User input
user_message = st.chat_input("Type your message...")

# Process input and generate response
if user_message:
    st.chat_message("user").write(user_message)
    st.session_state.chat_history.append(("user", user_message))

    system_prompt = """
    You are a friendly and a programming tutor.
    Always explain concepts in a simple and clear way, using examples when possible.
    If the user asks something unrelated to programming, politely bring the conversation back to programming topics.
    """
    full_input = f"{system_prompt}\n\nUser message:\n\"\"\"{user_message}\"\"\""

    context = [
        *[
            {"role": role, "parts": [{"text": msg}]} for role, msg in st.session_state.chat_history
        ],
        {"role": "user", "parts": [{"text": full_input}]}
    ]

    response = model.generate_content(
        context,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": 1000
        }
    )

    assistant_reply = response.text
    st.chat_message("assistant").write(assistant_reply)
    st.session_state.chat_history.append(("assistant", assistant_reply))
