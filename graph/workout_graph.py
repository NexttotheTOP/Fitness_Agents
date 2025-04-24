from langgraph.graph import END, StateGraph

from graph.workout_state import WorkoutState
from graph.nodes.workout_variation import generate_workout_variation

# Create the workflow
workflow = StateGraph(WorkoutState)

# Add nodes
workflow.add_node("generate", generate_workout_variation)

# Set entry point and connect directly to end
workflow.set_entry_point("generate")
workflow.add_edge("generate", END)

# Compile the graph
app = workflow.compile()

# Optional: Generate visualization
# app.get_graph().draw_mermaid_png(output_file_path="workout_graph.png") 