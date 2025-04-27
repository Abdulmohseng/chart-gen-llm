# from langchain_google_genai import ChatGoogleGenerativeAI
from llm import llm_google
from utils import print_state_variables
from state import State
from typing import Literal, Optional

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

    print("*"*15)
    print(f"\n\napplicablity: {output}\n\n")
    print("*"*15)
    
    if "true" in output.lower():
        return "recommend_charts"
    return "input_dataset"

def decide_if_valid(state) -> Optional[Literal["user_change_request", "generate_chart_code"]]:
    """
    Check code and executes it and maybe look at chart (multi-modal)
    """
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