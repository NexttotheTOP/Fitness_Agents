import uuid
from datetime import datetime
from typing import Dict, Any, AsyncGenerator, AsyncIterable
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import json

from graph.workout_state import WorkoutState, AgentState, QueryType
from graph.nodes.fitness_coach import ProfileAgent, DietaryAgent, FitnessAgent, QueryAgent, HeadCoachAgent

def create_fitness_coach_workflow():
    """Create a simplified fitness coach workflow with memory and thread management"""
    workflow = StateGraph(WorkoutState)
    
    # Add nodes for each agent
    workflow.add_node("profile", ProfileAgent())
    workflow.add_node("dietary", DietaryAgent())
    workflow.add_node("fitness", FitnessAgent())
    workflow.add_node("query", QueryAgent())
    
    # Create simple linear flow for profile creation
    workflow.add_edge("profile", "dietary")
    workflow.add_edge("dietary", "fitness")
    workflow.add_edge("fitness", END)
    
    # Set entry point to profile directly
    workflow.set_entry_point("profile")
    
    # Create the compiler with memory management
    return workflow.compile(checkpointer=MemorySaver())

# Create the app instance
app = create_fitness_coach_workflow()

def get_initial_state(user_profile: dict = None, thread_id: str = None, user_id: str = None) -> WorkoutState:
    """Create initial state with the provided user ID, thread ID or new ones if not provided"""
    # Extract user_id from user_profile if it exists and not provided separately
    if user_profile and "user_id" in user_profile and user_id is None:
        user_id = user_profile.get("user_id")
    
    return WorkoutState(
        user_id=user_id or str(uuid.uuid4()),  # Generate a user_id if not provided
        thread_id=thread_id or str(uuid.uuid4()),  # Generate a thread_id if not provided
        user_profile=user_profile or {},
        dietary_state=AgentState(
            last_update=datetime.now().isoformat(),
            content="",
            is_streaming=False
        ),
        fitness_state=AgentState(
            last_update=datetime.now().isoformat(),
            content="",
            is_streaming=False
        ),
        current_query="",
        query_type=QueryType.GENERAL,
        conversation_history=[],
        original_workout=None,
        variations=[],
        analysis={},
        generation=None,
        body_analysis=None,
        complete_response=None
    )

async def stream_response(state: WorkoutState, query: str = None) -> AsyncIterable[str]:
    """Stream responses from the appropriate agents"""
    if query:
        # Query mode - use query agent directly
        state["current_query"] = query
        query_agent = QueryAgent()
        
        # Stream response through query agent
        async for chunk in query_agent.stream(state):
            if chunk:
                yield chunk
    else:
        # Process through graph first to generate the complete content
        config = {
            "configurable": {
                "thread_id": state["thread_id"],
                "checkpoint_id": f"session_{state['thread_id']}"
            }
        }
        
        # Run the graph to generate all content
        result_state = app.invoke(state, config=config)
        
        # Format the complete response
        head_coach = HeadCoachAgent()
        formatted_state = head_coach(result_state)
        
        # Return complete response in chunks
        sections = formatted_state["complete_response"].split("\n\n")
        for section in sections:
            yield section + "\n\n"

def process_query(state: WorkoutState, query: str, config: dict = None) -> WorkoutState:
    """Process a user query using the existing thread"""
    # Create a new state with the existing data plus the query
    query_state = WorkoutState(
        user_id=state["user_id"],
        thread_id=state["thread_id"],
        user_profile=state["user_profile"],
        dietary_state=state["dietary_state"],
        fitness_state=state["fitness_state"],
        current_query=query,
        query_type=QueryType.GENERAL,  # Default type
        conversation_history=state["conversation_history"],
        original_workout=state.get("original_workout"),
        variations=state.get("variations", []),
        analysis=state.get("analysis", {}),
        generation=state.get("generation"),
        body_analysis=state.get("body_analysis"),
        complete_response=state.get("complete_response")
    )
    
    if config is None:
        config = {
            "configurable": {
                "thread_id": state["thread_id"],
                "checkpoint_id": f"session_{state['thread_id']}"
            }
        }
    
    # Just use the QueryAgent directly instead of going through the graph
    query_agent = QueryAgent()
    result = query_agent(query_state)
    
    # Update the session state with the results
    state.update(result)
    return state 