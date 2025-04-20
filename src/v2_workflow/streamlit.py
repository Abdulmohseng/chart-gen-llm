# Initializing UI branch
import streamlit as st
from state import State

# set session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# First: upload file in order to start chatting
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    if st.session_state.summary:
        with st.chat_message('assistant'):
            st.write(st.session_state.summary)

        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
