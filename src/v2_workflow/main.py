from langgraph.graph import StateGraph, START
from state import State
from data_processing import input_dataset, summarize, validate_chart_code
from chart_generation import recommend_charts, generate_chart_code
from user_interaction import user_chart_selection, user_change_request
from decision_nodes import decide_if_applicable, decide_if_valid, decide_change_request

def build_graph():
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

    return builder.compile()

if __name__ == "__main__":
    graph = build_graph()
    graph.invoke({
        'file_path': 'your dataset here',
        'chart_selected': '',
        'is_applicable': False,
        'is_valid': True,
        'summary': [],
        'code': '',
        'change_request': [],
        'prev_node': '',
        'code_retry': 0
    })