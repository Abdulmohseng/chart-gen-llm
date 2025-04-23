# Initializing UI branch
import streamlit as st
from state import State
from main import build_graph, invoke_build_graph
from langgraph.types import Command
import plotly.io as pio
import pandas as pd

st.set_page_config(layout="wide")
st.header("AI Workflow for Generating Plotly Charts")
thread_config = {
    "configurable": {"thread_id": "id1"}
}

if "graph" not in st.session_state:
    st.session_state.graph = invoke_build_graph()

def get_state():
    return st.session_state.graph.get_state(thread_config)

# if "state" not in st.session_state:
#     st.session_state.state = get_state()

datasets = {"China Vs Japan": "data/japanvchina.csv",
            "50 Start ups": "data/50_Startups.csv",
            "Student Performance": "data/StudentsPerformance.csv",
            "Supermarket Sales": "data/supermarket_sales - Sheet1.csv"
        }
# st.text(graph.get_state(thread_config))
# st.write(graph)

if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "df" not in st.session_state:
    st.session_state.df = None

# st.text(st.session_state.graph.get_state(thread_config))
# graph.invoke(Command(resume="data/japanvchina.csv"), config=thread_config)
# st.text(graph.get_state(thread_config).tasks[0].interrupts)
# graph.invoke(Command(resume="bar chart comparing tech sectors japan v china"), config=thread_config)
# st.code(graph.get_state(thread_config).values['code'])
# state_placeholder = st.empty()
# state_placeholder.write(get_state())
col1, col2 = st.columns([1,2])

if get_state().next[0] == "input_dataset":
    with st.spinner("Processing file ..."):
        upload_file = st.file_uploader("Please upload a datasets: ")
        if upload_file:
            st.session_state.file_name = "data/"+upload_file.name
            # st.write(st.session_state.file_name)
            st.session_state.df = pd.read_csv(st.session_state.file_name)
            
            st.session_state.graph.invoke(Command(resume=st.session_state.file_name), config=thread_config)
            st.rerun()
else:
    with col1:
        # if not get_state().values["summary"]:
        if get_state().tasks[0].interrupts:
            with st.form("input_form", clear_on_submit=True):
                st.dataframe(st.session_state.df.head())
                text_placeholder = st.empty()
                user_input = st.text_area("", key="user_input")
                submitted = st.form_submit_button("Submit")
                text_placeholder.markdown("### " + get_state().tasks[0].interrupts[0].value)

            if submitted:
                with st.spinner("Please wait ..."):
                    st.session_state.graph.invoke(Command(resume=user_input), config=thread_config)
                text_placeholder.text(get_state().tasks[0].interrupts[0].value)
                # state_placeholder.write(get_state())

            if get_state().values['chart_selected']:
                st.subheader("User prompt: ", divider="gray")
                st.markdown(get_state().values['chart_selected'])
            if get_state().values['business_questions']:
                st.subheader("Business questions and charts:", divider="gray")
                st.markdown(get_state().values['business_questions'])
            else:
                st.subheader("Available datasets: ", divider="gray")
                st.json(datasets)
            
    with col2:
        if get_state().values['figures']:
            st.plotly_chart(pio.from_json(get_state().values['figures']['llm_generated_plot']))
        # if get_state().values['code']:
            st.code(get_state().values['code'])
        elif get_state().values['summary']:
            st.subheader("Input data summary:", divider="gray")
            st.markdown(get_state().values['summary'])

# if get_state().values['prompt']:
#     st.markdown(get_state().values['prompt'])
# st.write(get_state())
# st.write(user_input)
# st.rerun()
# st.rerun()
# st.text(graph.get_state())
# set session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "runner" not in st.session_state:
#     # st.session_state.runner = build_graph()


# # First: upload file in order to start chatting
# uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
# if uploaded_file:
#     if st.session_state.summary:
#         with st.chat_message('assistant'):
#             st.write(st.session_state.summary)

#         if prompt := st.chat_input():
#             st.session_state.messages.append({"role": "user", "content": prompt})
#             with st.chat_message("user"):
#                 st.write(prompt)
