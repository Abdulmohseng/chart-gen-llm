from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

llm_google = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=st.secrets['GEMINI_API_KEY'])
llm_qwen25coder = ChatOllama(model='qwen2.5-coder:14b', temperature=0.7)