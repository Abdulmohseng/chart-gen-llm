from state import State
from langgraph.types import interrupt

def user_chart_selection(state: State):
    print("---Step 3: user feedback---")
    choice = interrupt("Choose a chart to create: ") # or "Line chart with Year on the x-axis, Market Share (%) on the y-axis, and separate lines for each Tech Sector, further grouped by Country (Japan, China). A dual-axis chart could be considered if the market share scales are vastly different between countries."
    return {"chart_selected": choice}

def user_change_request(state: State):
    print("---Step 6: user change request---")
    choice = interrupt("Do you want to change the charts? ('no' to end) ")# or "no"
    state['change_request'].append(choice)
    if choice == 'no':
        return {'prev_node': 'user_change_request'}

    return {'prev_node': 'user_change_request'}