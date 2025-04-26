# Initializing UI branch
import streamlit as st
from state import State
from main import build_graph, invoke_build_graph
from langgraph.types import Command
import plotly.io as pio
import pandas as pd
import time

st.set_page_config(layout="wide")
st.header("AI Workflow for Generating Plotly Charts", divider="gray")

# Thread configuration for langgraph
THREAD_CONFIG = {
    "configurable": {"thread_id": "id1"}
}

# Available datasets
DATASETS = {
    "China Vs Japan": "data/japanvchina.csv",
    "50 Start ups": "data/50_Startups.csv",
    "Student Performance": "data/StudentsPerformance.csv",
    "Supermarket Sales": "data/supermarket_sales - Sheet1.csv"
}

# Initialize session state variables
if "graph" not in st.session_state:
    st.session_state.graph = invoke_build_graph()
    
if "file_name" not in st.session_state:
    st.session_state.file_name = None
    
if "df" not in st.session_state:
    st.session_state.df = None

def get_state():
    """Get the current state of the graph workflow"""
    return st.session_state.graph.get_state(THREAD_CONFIG)

# Create page layout
col1, col2 = st.columns([1, 2])

# Main application flow
if get_state().next:
    # Handle dataset input step
    if get_state().next[0] == "input_dataset":
        with st.spinner("Processing file..."):
            upload_file = st.file_uploader("Please upload a dataset:")
            if upload_file:
                st.session_state.file_name = f"data/{upload_file.name}"
                try:
                    st.session_state.df = pd.read_csv(st.session_state.file_name)
                    st.session_state.graph.invoke(Command(resume=st.session_state.file_name), config=THREAD_CONFIG)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing file: {e}")
    
    # Handle all other steps
    else:
        with col1:
            if get_state().tasks[0].interrupts:
                with st.form("input_form", clear_on_submit=True):
                    # Show dataframe preview
                    if st.session_state.df is not None:
                        st.dataframe(st.session_state.df.head())
                    
                    # Create form elements
                    text_placeholder = st.empty()
                    user_input = st.text_area("", key="user_input")
                    
                    # Create button columns with better proportions
                    button_cols = st.columns([0.2, 0.025])
                    with button_cols[0]:
                        submitted = st.form_submit_button("Submit")
                    with button_cols[1]:
                        skip_to_feedback = st.form_submit_button("⏪", help="Return to previous step")
                    
                    # Display interruption message
                    if get_state().tasks[0].interrupts:
                        text_placeholder.subheader(get_state().tasks[0].interrupts[0].value)

                # Handle form submission
                if submitted:
                    with st.spinner("Processing your request..."):
                        st.session_state.graph.invoke(Command(resume=user_input), config=THREAD_CONFIG)
                        if get_state().tasks:
                            text_placeholder.subheader(get_state().tasks[0].interrupts[0].value)
                        else:
                            st.rerun()
                
                # Handle skip to feedback
                if skip_to_feedback:
                    with st.spinner("Moving to chart selection step..."):
                        current_state = get_state()
                        if current_state.values.get('summary'):
                            # Clear any existing tasks
                            if hasattr(current_state, "tasks") and current_state.tasks:
                                for interrupt in current_state.tasks[0].interrupts:
                                    st.session_state.graph.invoke(Command(resume=""), config=THREAD_CONFIG)
                            
                            # Reset and reinitialize workflow
                            st.session_state.graph = invoke_build_graph()
                            
                            if st.session_state.file_name:
                                st.session_state.graph.invoke(Command(resume=st.session_state.file_name), config=THREAD_CONFIG)
                                
                                # Advance through workflow steps
                                while get_state().next and get_state().next[0] != "user_chart_selection":
                                    if get_state().tasks and get_state().tasks[0].interrupts:
                                        st.session_state.graph.invoke(Command(resume="Auto-advancing to chart selection"), config=THREAD_CONFIG)
                                    else:
                                        break
                                
                                st.rerun()
                            else:
                                st.error("No dataset loaded. Please upload a dataset first.")
                        else:
                            st.error("Cannot skip to chart selection - no data summary available yet.")

                # Display additional information
                if get_state().values.get('chart_selected'):
                    st.subheader("User prompt:", divider="gray")
                    st.markdown(get_state().values['chart_selected'])
                    
                if get_state().values.get('business_questions'):
                    st.subheader("Business questions and charts:", divider="gray")
                    st.markdown(get_state().values['business_questions'])
                else:
                    st.subheader("Available datasets:", divider="gray")
                    st.json(DATASETS)
                
        # Right column for chart display
        with col2:
            if get_state().values.get('figures'):
                st.plotly_chart(pio.from_json(get_state().values['figures']['llm_generated_plot']))
                if get_state().values.get('code'):
                    st.code(get_state().values['code'])
            elif get_state().values.get('summary'):
                st.subheader("Input data summary:", divider="gray")
                st.markdown(get_state().values['summary'])
else:
    # End of workflow
    st.session_state.clear()
    st.title("Thank you and goodbye!")
    time.sleep(3)
    st.session_state.graph = invoke_build_graph()
    st.rerun()
