from langchain_google_genai import ChatGoogleGenerativeAI
from state import State
from utils import clean_llm_code, execute_code
from llm import llm_google

def recommend_charts(state: State):
    print("---Step 2: recommend_charts---")
    prompt = f"""
    Your are an experienced data visualization developer and business analyst, you are tasked to generate at most three business questions (be creative with the last chart) with a chart suggestion given the following summary of the dataset:
    
    Summary: {state['summary']}

    Your outputs should be structured as follows:
    1.** Business question **:\n
    ** Chart recommendation **:
    """
    
    result = llm_google.invoke(prompt).content
    return {'business_questions': result}

def generate_chart_code(state: State, val_message=''):
    print("---Step 4: code generation---")
    prompt = f"""
    You are an expert data visualization engineer that produces charts using plotly and executes the code yourself.
    
    Given the following Summary of the dataset: {state['summary']}

    Generate a chart based on this request: {state['chart_selected']}

    Rules:
    - The output should only be python code, don't add triple quotes such as ``` or ```python at start and end.
    - Do any necessary aggregations for example: groupby() or sum() using pandas.
    - You already have access to variable 'df' do not define it.
    - do not display fig.show()
    """
    
    if state['prev_node'] == 'user_change_request':
        prompt += f"""
        ** Given the above summary and instructions, please modify the following code based on the user's request. **

        code: {state['code']}

        user request: {state['change_request'][-1]}
    """
    
    if state['val_message']:
        prompt += f"""
        ** The code I just ran was invalid, please modify the following code based on the error message. **

        error message: {state['val_message']}

        code: {state['code']}
    """
    state['prompt'] = prompt
    
    code_output = llm_google.invoke(prompt).content
    code_output = clean_llm_code(code_output)

    new_state = execute_code(code_output, state)
    return new_state