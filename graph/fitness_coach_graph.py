import uuid
from datetime import datetime
from typing import Dict, Any, AsyncGenerator, AsyncIterable
from langgraph.graph import StateGraph, END
import json
import os
import logging
import traceback

from graph.workout_state import WorkoutState, AgentState, QueryType
from graph.nodes.fitness_coach import ProfileAgent, DietaryAgent, FitnessAgent, QueryAgent, HeadCoachAgent
from graph.memory_store import get_postgres_checkpointer, store_profile_overview, get_most_recent_profile_overview, setup_fitness_tables, get_structured_previous_overview

# Initialize the PostgreSQL connection pool and checkpointer
try:
    postgres_checkpointer = get_postgres_checkpointer()
    setup_fitness_tables()  # Setup tables if they don't exist
    logging.info("PostgreSQL checkpointer and fitness tables initialized successfully")
except Exception as e:
    logging.error(f"Error initializing PostgreSQL: {str(e)}")
    # Fallback to memory checkpointer if PostgreSQL initialization fails
    from langgraph.checkpoint.memory import MemorySaver
    postgres_checkpointer = MemorySaver()
    logging.warning("Using in-memory checkpointer as fallback")

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
    
    # Create the compiler with PostgreSQL checkpointer for persistence
    return workflow.compile(checkpointer=postgres_checkpointer)

# Create the app instance
app = create_fitness_coach_workflow()

def get_initial_state(user_profile: dict = None, thread_id: str = None, user_id: str = None) -> WorkoutState:
    """Create initial state with the provided user ID, thread ID or new ones if not provided"""
    # Extract user_id from user_profile if it exists and not provided separately
    if user_profile and "user_id" in user_profile and user_id is None:
        user_id = user_profile.get("user_id")
    
    # Generate IDs if not provided
    user_id = user_id or str(uuid.uuid4())
    thread_id = thread_id or str(uuid.uuid4())
    
    logging.info("\n\n==========================================")
    logging.info(f"Creating initial state for user_id: {user_id}, thread_id: {thread_id}")
    logging.info("==========================================\n\n")
    
    # Try to fetch and parse previous fitness overview data for this user
    previous_complete_response = None
    previous_sections = None
    
    try:
        logging.info("\n\n==========================================")
        logging.info(f"Retrieving previous profile overview for user {user_id}")
        logging.info("==========================================\n\n")
        
        # Get structured previous overview data
        previous_data, parsed_sections = get_structured_previous_overview(user_id)
        
        if previous_data:
            # Store the complete previous response
            previous_complete_response = previous_data.get("response")
            
            # Store the parsed sections
            previous_sections = parsed_sections
            
            logging.info("\n\n==========================================")
            logging.info(f"Retrieved previous profile overview for user {user_id}")
            logging.info(f"Previous overview length: {len(previous_complete_response)}")
            logging.info(f"From thread_id: {previous_data.get('thread_id')}")
            logging.info(f"Timestamp: {previous_data.get('timestamp')}")
            logging.info(f"Parsed sections: {list(parsed_sections.keys())}")
            logging.info("==========================================\n\n")
        else:
            logging.info("\n\n==========================================")
            logging.info(f"No previous profile overview found for user {user_id}")
            logging.info("==========================================\n\n")
    except Exception as e:
        logging.error("\n\n==========================================")
        logging.error(f"Error retrieving previous profile overview: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        logging.error("==========================================\n\n")
    
    return WorkoutState(
        user_id=user_id,
        thread_id=thread_id,
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
        complete_response=None,
        previous_complete_response=previous_complete_response,
        previous_sections=previous_sections
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
        logging.info(f"Invoking app with state for user_id: {state['user_id']}")
        result_state = app.invoke(state, config=config)
        logging.info(f"App invoke completed for user_id: {state['user_id']}")
        
        # Format the complete response
        logging.info("Calling HeadCoachAgent to format response")
        head_coach = HeadCoachAgent()
        formatted_state = head_coach(result_state)
        logging.info(f"HeadCoach formatting completed, complete_response length: {len(formatted_state.get('complete_response', 'None'))}")
        
        # Store the complete response using Supabase
        if formatted_state.get("complete_response"):
            logging.info(f"Preparing to store profile overview for user {formatted_state['user_id']}")
            try:
                metadata = {
                    "age": formatted_state.get("user_profile_data", {}).get("age"),
                    "gender": formatted_state.get("user_profile_data", {}).get("gender"),
                    "weight": formatted_state.get("user_profile_data", {}).get("weight"),
                    "height": formatted_state.get("user_profile_data", {}).get("height"),
                    "goals": formatted_state.get("user_profile_data", {}).get("fitness_goals"),
                    "timestamp": datetime.now().isoformat()
                }
                
                logging.info(f"Calling store_profile_overview with user_id: {formatted_state['user_id']}, thread_id: {formatted_state['thread_id']}")
                
                # Store complete_response from state as the overview
                store_profile_overview(
                    formatted_state["user_id"],
                    formatted_state["thread_id"],
                    formatted_state["complete_response"],
                    metadata
                )
                logging.info(f"store_profile_overview call completed for user {formatted_state['user_id']}")
            except Exception as e:
                logging.error(f"Error storing profile overview: {str(e)}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                logging.error(f"User ID: {formatted_state.get('user_id')}")
                logging.error(f"Thread ID: {formatted_state.get('thread_id')}")
                logging.error(f"Complete response length: {len(formatted_state.get('complete_response', ''))}")
        else:
            logging.warning(f"No complete_response available to store for user {formatted_state.get('user_id', 'unknown')}")
        
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
        complete_response=state.get("complete_response"),
        previous_complete_response=state.get("previous_complete_response"),
        previous_sections=state.get("previous_sections")
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