import streamlit as st
import pandas as pd
import numpy as np 

from dotenv import load_dotenv
import os
load_dotenv()

import google.generativeai as genai

api_key = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content('write a short story about an old man living in the alps skiing, make it super fun')

st.markdown(response.text)