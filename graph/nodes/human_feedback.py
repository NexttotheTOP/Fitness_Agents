from typing import Dict, Any, Literal
from graph.workout_state import StateForWorkoutApp
from langgraph.types import interrupt, Command

def await_human_feedback(state: StateForWorkoutApp):
    # print(f"HUMAN_FEEDBACK NODE - State ID: {id(state)}")
    # print(f"State keys: {list(state.keys())}")
    # print(f"workout_profile_analysis: {state.get('workout_profile_analysis', 'NOT_FOUND')[:100]}...")
    feedback = interrupt("Do you approve this plan? (agree/deny/custom)")
    print(f"HUMAN_FEEDBACK NODE - feedback: {feedback}")
    state["feedback"] = feedback
    

    return Command(goto="conversation_router", update={"feedback": feedback})