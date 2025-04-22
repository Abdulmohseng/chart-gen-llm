from typing_extensions import TypedDict

class State(TypedDict):
    file_path: str
    chart_selected: str
    is_applicable: bool
    is_valid: bool
    summary: list
    business_questions: str
    code: str
    change_request: list[str]
    prev_node: str
    code_retry: int
    figures: dict
    val_message: str
    # user_prompt: str