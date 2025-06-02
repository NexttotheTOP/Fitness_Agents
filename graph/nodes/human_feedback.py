from typing import Dict, Any, Literal
from graph.workout_state import StateForWorkoutApp

def await_human_feedback(state: StateForWorkoutApp):
    print(f"HUMAN_FEEDBACK NODE - State ID: {id(state)}")
    print(f"State keys: {list(state.keys())}")
    print(f"workout_profile_analysis: {state.get('workout_profile_analysis', 'NOT_FOUND')[:100]}...")
    
    # For now, just pass state through without yielding events
    # This tests if the async generator pattern was the issue
    
    return state