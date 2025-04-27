# The plan is to develop a langgraph multi-step workflow that generate plotly charts from a given dataset
import random
import os
import re
import pandas as pd
from typing import Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
# from langgraph.types import Command, interrupt
# from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI


llm_google = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key='AIzaSyDdiVNyPorh6mtoXSv7zuqvRTOMiXEMVIE')
llm_qwen25coder = ChatOllama(model='qwen2.5-coder:14b', temperature=0.7)

class State(TypedDict):
    file_path: str
    chart_selected: str
    is_applicable: bool
    is_valid: bool
    summary: list
    code: str
    change_request: list[str]
    prev_node: str
    code_retry: int
    val_message: str
def input_dataset(state):
    print("---input dataset---")
    file_path = input("Enter the path to your CSV dataset: (default data/japanvchina.csv)") or "data/japanvchina.csv"
    return {"file_path": file_path}

def summarize(state):
    """
    Step 1:
    Summarize data and perform simple Exploratory Data Analysis.
    The output is designed to be descriptive enough for an LLM to reason about data,
    generate charts, and business insights. No chart recommendations are made.
    """
    print("---Step 1: Summarize---")
    try:
        df = pd.read_csv(state['file_path'])
    except Exception as e:
        print(f"Failed to read dataset, try again: {e}")
        return {'summary': [], 'is_applicable': False}

    summary = []

    summary.append(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n")
    summary.append("Column Overview:")

    for col in df.columns:
        dtype = df[col].dtype
        missing = df[col].isna().sum()
        non_missing = df.shape[0] - missing
        summary.append(f"🟦 {col} ({dtype})")

        if dtype == 'object':
            unique = df[col].nunique()
            top = df[col].value_counts().idxmax() if unique > 0 else "N/A"
            freq = df[col].value_counts().max() if unique > 0 else 0
            summary.append(f"   • Unique values: {unique}, Most frequent: '{top}' ({freq} occurrences)")
            sample_values = df[col].dropna().unique()[:3]
            summary.append(f"   • Example values: {', '.join(map(str, sample_values))}")
        
        elif pd.api.types.is_numeric_dtype(df[col]):
            mean = df[col].mean()
            std = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            skew = df[col].skew()
            summary.append(f"   • Mean: {mean:.2f}, Std: {std:.2f}, Min: {min_val}, Max: {max_val}")
            summary.append(f"   • Skewness: {skew:.2f}")

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            summary.append(f"   • Range: {df[col].min()} to {df[col].max()}")

        summary.append(f"   • Missing: {missing} ({missing / df.shape[0] * 100:.1f}%)\n")

    return {
        'summary': "\n".join(summary),
        'is_applicable': True
    }

def recommend_charts(state):
    """
    Step 2:
    Generate business question and chart suggestion based on the summarize step.
    """
    print("---Step 2: recommend_charts---")
    prompt = f"""
    Your are an experienced data visualzation developer and business analyst, you are tasked to generate at most five business question with a chart suggestion given the following summary of the datatset:
    
    Summary: {state['summary']}

    Your outputs should be structures as follows:
    1. Business question
    2. Chart recommendation
    """
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key='AIzaSyDdiVNyPorh6mtoXSv7zuqvRTOMiXEMVIE')
    
    result = llm.invoke(prompt).content
    print(result)
    pass

def user_chart_selection(state):
    """
    Step 3:
    Ask user to select a chart to visualize.
    """
    print("---Step 3: user feedback---")
    # feedback = interrupt("Please provide feedback:")
    choice = input("Choose a chart to create: ") or "Line chart with Year on the x-axis, Market Share (%) on the y-axis, and separate lines for each Tech Sector, further grouped by Country (Japan, China). A dual-axis chart could be considered if the market share scales are vastly different between countries."
    return {"chart_selected": choice}
    # pass

def generate_chart_code(state):
    """
    Step 4:
    Generate chart code then sends it for validation
    """
    print("---Step 4: code generation---")
    prompt = f"""
    You are an expert data vizualization engineer that produces charts using plotly and executes the code yourself.
    
    Given the following Summary of the dataset: {state['summary']}

    Generate a chart based on this request: {state['chart_selected']}

    Rules:
    - The output should only be python code, dont add triple quotes such as ``` or ```python at start and end.
    - Do any necessary aggregations for example: groupby() or sum() using pandas.
    - Your already have acces to variable 'df' do not define it.
    - at the the end just run fig.show()
    """
    if state['prev_node'] == 'user_change_request':
        prompt += f"""
        ** Given the above summary and instructions, please modify the following code based on the users request. **

        code: {state['code']}

        user request: {state['change_request'][-1]}
    """
    if state['val_message']:
        prompt+= f"""
        ** The code I just ran was invalid, please modify the following code based on error message. **

        error message: {state['val_message']}

        code: {state['code']}
    """
    # api_key = os.getenv('GEMINI_API_KEY')
    
    code_output = llm_google.invoke(prompt).content
    code_output = clean_llm_code(code_output)

    new_state = execute_code(code_output, state)
    return new_state

def validate_chart_code(state):
    """
    Step 5:
    Validate the chart code then either passes or fails.
    if pass --> user change requests
    if fail --> go to step 4 and generate the code again given the error message
    """
    print("---Step 5: chart validate---")
    pass

def user_change_request(state):
    """
    Step 6:
    Ask user if there are any changes they want to implement
    if yes --> generate chart code based on the instructions
    if no --> present final chart
    """
    print("---Step 6: user change request---")
    choice = input("Do you want to change the charts? ('no' to end) ") or "no"
    state['change_request'].append(choice)
    if choice == 'no':
        return {'prev_node': 'user_change_request'}

    return {'prev_node': 'user_change_request'} 


# ----- Decision nodes -----
def decide_if_applicable(state) -> Literal["input_dataset", "recommend_charts"]:
    """
    The LLM should decide if the given dataset is applicable to creating charts from (it has meaning)
    """
    prompt = f"""
    You are an evaluator that will wither decide if the dataset has potential business questions or charts that can be created based on the following summary:

    summary: {state['summary']}

    The output should either be True --> if it passes or False --> if data is bad
    """
    if not state['is_applicable']:
        return "input_dataset"
    
    output = llm_google.invoke(prompt).content
    print(f"\n\n**********\n\napplicablity: {output}\n\n**********\n\n")
    if "true" in output.lower():
        return "recommend_charts"
    return "input_dataset"

def decide_if_valid(state) -> Optional[Literal["user_change_request", "generate_chart_code"]]:
    """
    Check code and executes it and maybe look at chart (multi-modal)
    """
    print("---Decide if valid step---")
    print("state['code_retry']", state['code_retry'])
    if state['code_retry'] > 3:
        return None
    elif state['val_message']:
        return "generate_chart_code"
    else:
        return "user_change_request"

def decide_change_request(state) -> Optional[Literal['generate_chart_code']]:
    if state['change_request'][-1].lower() == 'no':
        print_state_variables(state)
        return None
    return "generate_chart_code"

# ---- Helper functions ----
def print_state_variables(state):
    for key, value in state.items():
        print("--------------")
        print(f"{key}: {value}")
        print("--------------")

def clean_llm_code(raw_output):
    """
    Strips markdown code fencing like ```python ... ```
    """
    # Remove triple backticks and optional language labels
    cleaned = re.sub(r"^```(?:python)?\n", "", raw_output.strip())
    cleaned = re.sub(r"\n```$", "", cleaned)
    return cleaned

def execute_code(code: str, state: State):
    try:
        local_env = {
            'pd': pd,
            'df': pd.read_csv(state['file_path'])
        }
        exec(code, local_env)
    except Exception as e:
        val_message = f'error executing the code generated from llm, {e}'
        state['code_retry'] = state['code_retry']+1
        # generate_chart_code(state, val_message)
        return {'code_retry': state['code_retry'], 'val_message': val_message, 'code': code}
    
    return {'code_retry': 0, 'val_message': '', 'code': code}
    



builder = StateGraph(State)
# Nodes
builder.add_node("input_dataset", input_dataset)
builder.add_node("dataset_summary", summarize)
builder.add_node("recommend_charts", recommend_charts)
builder.add_node("user_chart_selection", user_chart_selection)
builder.add_node("generate_chart_code", generate_chart_code)
builder.add_node("validate_chart_code", validate_chart_code)
builder.add_node("user_change_request", user_change_request)
# Edges
builder.add_edge(START, "input_dataset")
builder.add_edge("input_dataset", "dataset_summary")
builder.add_conditional_edges("dataset_summary", decide_if_applicable)
builder.add_edge("recommend_charts", "user_chart_selection")
builder.add_edge("user_chart_selection", "generate_chart_code")
builder.add_edge("generate_chart_code", "validate_chart_code")
builder.add_conditional_edges("validate_chart_code", decide_if_valid, {
    "user_change_request":"user_change_request",
    'generate_chart_code':'generate_chart_code',
    None: '__end__'
})
builder.add_conditional_edges("user_change_request", decide_change_request, {
    'generate_chart_code':'generate_chart_code',
    None: '__end__'
})

# Set up memory
# checkpointer = MemorySaver()

# Add
graph = builder.compile()

# View
# display(Image(graph.get_graph().draw_mermaid_png()))
# print(graph.get_graph().draw_ascii())

graph.invoke({
    'file_path': 'your dataset here',
    'chart_selected': '',
    'is_applicable': True,
    'is_valid': True,
    'summary':[],
    'code':'',
    'change_request': [],
    'prev_node': '',
    'code_retry': 0,
    'val_message': ''
})