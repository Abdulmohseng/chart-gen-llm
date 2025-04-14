import streamlit as st
import plotly.express as px
import pandas as pd

from dotenv import load_dotenv
import os
load_dotenv()

# import google.generativeai as genai
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI



def select_llm(input: str):
    if input == "Google":
        api_key =  "AIzaSyDdiVNyPorh6mtoXSv7zuqvRTOMiXEMVIE" # os.getenv('GEMINI_API_KEY')
        # genai.configure(api_key=api_key)
        model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key=api_key)
    elif input == "Local (Llama3.2)":
        model = ChatOllama(model="llama3.2:latest")
    
    return model


# Function to generate plotly code from a natural language description
def generate_plot_code(description, df_columns, df, model):
    prompt = f"""
    You are an assistant that generates Python code for data visualizations using Plotly.
    Given the following description and available columns, create a chart:
    
    Description: {description}
    Columns: {', '.join(df_columns)}
    Dataframe: {df}
    The code should use Plotly's plotly.express (px) library and generate a figure object (e.g., 'fig = px.scatter(...)').

    Make sure to include the imports of all the necesseray libraries.
    Always use df as the name of the dataframe directly.
    If the column is a timeseries, order it as ascending first before building the chart.
    Make sure to do the necessary aggregations beforehand, for example: groupby().

    # çFinally, the output should only be code no comments or tags or anything!
    """

    response = model.invoke(prompt)#.text.replace('```python','').replace('```','').replace('fig.show()','')
    # print(response.content)
    return response.content

# Function to execute generated Plotly code
def execute_plotly_code(code, df):
    local_vars = {'df':df}
    try:
        exec(code, {}, local_vars)
        return local_vars.get("fig")
    except Exception as e:
        st.error(f"Error generating plot: {e}")
        return None

# Streamlit app setup
st.title("Natural Language Data Visualization with LLM")

st.markdown(
    "### Describe the plot you want and let the AI generate it!"
    "\nFor example: 'Show a bar chart of average sales by region.'"
)

# Upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Here's a preview of your data:")
    st.write(df.head())

    # Get user input for the desired plot description
    description = st.text_input("Describe the plot you'd like to create:")

    if description:
        model_name = st.selectbox(label="Choose your model",options=('Google', 'Local (Llama3.2)'))

        model = select_llm(model_name)
        # Generate plot code based on the user's description
        code = generate_plot_code(description, df.columns, df, model)
        
        st.subheader("Generated Plotly Code")
        st.code(code, language="python")

        # Try to execute the generated code and display the plot
        fig = execute_plotly_code(code, df)
        if fig:
            st.plotly_chart(fig)
