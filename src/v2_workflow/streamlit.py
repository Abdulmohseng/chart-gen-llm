# Initializing UI branch
import streamlit as st
from state import State
from main import build_graph, invoke_build_graph
from langgraph.types import Command
import plotly.io as pio

# initial_state = {
#         'file_path': 'your dataset here',
#         'chart_selected': '',
#         'is_applicable': False,
#         'is_valid': True,
#         'summary': [],
#         'code': '',
#         'change_request': [],
#         'prev_node': '',
#         'code_retry': 0
#     }
thread_config = {
    "configurable": {"thread_id": "id1"}
}

if "graph" not in st.session_state:
    st.session_state.graph = invoke_build_graph()

def get_state():
    return st.session_state.graph.get_state(thread_config)


# st.text(graph.get_state(thread_config))
# st.write(graph)



# st.text(st.session_state.graph.get_state(thread_config))
# graph.invoke(Command(resume="data/japanvchina.csv"), config=thread_config)
# st.text(graph.get_state(thread_config).tasks[0].interrupts)
# graph.invoke(Command(resume="bar chart comparing tech sectors japan v china"), config=thread_config)
# st.code(graph.get_state(thread_config).values['code'])

if get_state().tasks[0].interrupts:
    # st.text()
    text_placeholder = st.empty() 
    text_placeholder.text(get_state().tasks[0].interrupts[0].value)
    user_input = st.text_input("user input: ")
    if user_input:
        st.session_state.graph.invoke(Command(resume=user_input), config=thread_config)
        text_placeholder.text(get_state().tasks[0].interrupts[0].value)
        # st.rerun()
    # with st.spinner():
# st.text(get_state())
if get_state().values['code']:
    st.code(get_state().values['code'])
    # with st.echo():
    #     get_state().values['code']
if get_state().values['figures']:
    
    st.plotly_chart(pio.from_json(get_state().values['figures']['llm_generated_plot']))
st.text(get_state())
    

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
