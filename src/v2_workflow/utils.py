import re
import pandas as pd
from state import State
# from chart_generation import generate_chart_code

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
        global_env = {
            'pd': pd,
            'df': pd.read_csv(state['file_path'])
        }
        local_env = {}
        exec(code, global_env, local_env)
        state['figures']['llm_generated_plot'] = local_env['fig'].to_json()
        state['code'] = code
        print(state)
        return state
    except Exception as e:
        val_message = f'error executing the code generated from llm, {e}'
        state['code_retry'] += 1
        # generate_chart_code(state, val_message)
        return state

     