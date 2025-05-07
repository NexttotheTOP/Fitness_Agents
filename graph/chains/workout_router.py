from typing import Dict, Any, Literal
from graph.workout_state import StateForWorkoutApp

def route_by_workflow_type(state: StateForWorkoutApp) -> Dict[str, Any]:
    """Route to the appropriate node based on workflow_type in state."""
    print("\n---ROUTING WORKFLOW---")
    
    workflow_type = state.get("workflow_type", "create")  # Default to create if not specified
    print(f"Workflow type: {workflow_type}")
    
    # Return the state and next node
    if workflow_type == "variation":
        return {"next": "generate_variations"}
    else:
        return {"next": "create_workout"}
