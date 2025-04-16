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
    if input == "gemini-1.5-flash":
        api_key = os.getenv('GEMINI_API_KEY')
        # genai.configure(api_key=api_key)
        model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key=api_key)
    elif input == "llama3.2:latest":
        model = ChatOllama(model="llama3.2:latest")
    elif input == "codellama:7b":
        model = ChatOllama(model="codellama:7b")
    
    return model

# Function to generate plotly code from a natural language description
def generate_plot_code(description, df_columns, file_name, model):
    prompt = f"""
    You are an assistant that generates Python code for data visualizations using Plotly.
    Given the following description and available columns, create a chart:
    
    Description: {description}
    Columns: {', '.join(df_columns)}
    
    IMPORTANT: The dataframe is already loaded as the variable 'df'. DO NOT include any code to read CSV files.
    Do not use pd.read_csv() or any file loading functions. The dataframe is already available as 'df'.

    Make sure to include the imports of plotly libraries only if needed.
    If the column is a timeseries, order it as ascending first before building the chart.
    Make sure to do the necessary aggregations beforehand, for example: groupby().

    Your code should just work with the existing 'df' variable and create a plotly figure named 'fig'.
    
    # Finally, the output should only be code no comments or tags or "```"!
    """

    response = model.invoke(prompt)
    
    # Clean up the response to ensure it's just the code
    code = response.content
    # Remove any potential code blocks if present
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]
    
    # Remove any fig.show() as we'll display with Streamlit
    code = code.replace('fig.show()', '')
    
    return code

# Function to execute generated Plotly code
def execute_plotly_code(code, df):
    # Define global variables that might be needed
    globals_dict = {
        'pd': pd,
        'px': px,
        'plotly': __import__('plotly'),
        'go': __import__('plotly.graph_objects'),
        'np': __import__('numpy')
    }
    
    # Define local variables including the dataframe
    local_vars = {'df': df}
    
    try:
        # Execute the code with the necessary context
        exec(code, globals_dict, local_vars)
        
        # Try to get the figure from locals
        if 'fig' in local_vars:
            return local_vars['fig']
        else:
            st.error("The generated code did not create a 'fig' variable.")
            st.write("Variables created:", list(local_vars.keys()))
            return None
    except Exception as e:
        st.error(f"Error executing the generated code: {str(e)}")
        st.code(code, language="python")  # Show the problematic code again
        import traceback
        st.error(traceback.format_exc())  # Show the full traceback
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
        model_name = st.selectbox(label="Choose your model", options=('gemini-1.5-flash', 'llama3.2:latest', 'codellama:7b'))
        
        if st.button("Generate Chart!"):
            model = select_llm(model_name)
            
            # Get the filename instead of passing the file object
            file_name = uploaded_file.name if hasattr(uploaded_file, 'name') else "uploaded_file.csv"
            
            # Generate plot code based on the user's description
            code = generate_plot_code(description, df.columns, file_name, model)
            
            st.subheader("Generated Plotly Code")
            st.code(code, language="python")

            # Try to execute the generated code and display the plot
            fig = execute_plotly_code(code, df)
            if fig:
                st.plotly_chart(fig)