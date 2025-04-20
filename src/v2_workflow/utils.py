import re
import pandas as pd
from state import State

def print_state_variables(state: State):
    for key, value in state.items():
        print("--------------")
        print(f"{key}: {value}")
        print("--------------")

def clean_llm_code(raw_output: str) -> str:
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
        state['code_retry'] += 1
        generate_chart_code(state, val_message)