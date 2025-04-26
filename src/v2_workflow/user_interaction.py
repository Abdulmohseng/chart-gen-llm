from state import State
from langgraph.types import interrupt

def user_chart_selection(state: State):
    print("---Step 3: user feedback---")
    prompt_message = f"""
🔍 DESCRIBE YOUR IDEAL CHART:
Be as specific as possible about what you want to visualize. Include:
- Chart type (bar, line, scatter, pie, heatmap, etc.)
- X and Y axis fields
- Title and subtitle ideas
- Special features (dual axis, annotations, etc.)
"""
    choice = interrupt(prompt_message)
    return {"chart_selected": choice}

def user_change_request(state: State):
    print("---Step 6: user change request---")
    prompt_message = """
📊 HOW WOULD YOU LIKE TO MODIFY THIS CHART?
Be specific about what changes you want:
- Request specific visual changes (colors, fonts, layout, arabic)
- Add/modify labels, titles or annotations
- Adding custom hover template
"""
    choice = interrupt(prompt_message)
    state['change_request'].append(choice)
    if choice.lower() == 'no':
        return {'prev_node': 'user_change_request'}

    return {'prev_node': 'user_change_request'}