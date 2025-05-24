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
    """Stream responses from the appropriate agents, and at the end stream the progress tracking section."""
    if query:
        # Query mode - use query agent directly
        state["current_query"] = query
        query_agent = QueryAgent()
        # Stream response through query agent
        async for chunk in query_agent.stream(state):
            if chunk:
                yield chunk
    else:
        config = {
            "configurable": {
                "thread_id": state["thread_id"],
                "checkpoint_id": f"session_{state['thread_id']}"
            }
        }
        try:
            final_content = {
                "profile_assessment": "",
                "body_analysis": state.get("body_analysis", ""),
                "dietary_plan": "",
                "fitness_plan": "",
                "progress_tracking": "",
                "complete_response": ""
            }
            # --- NEW: Accumulate the full streamed overview ---
            streamed_overview = ""
            # Stream through each node in the graph
            async for chunk_type, chunk in app.astream(
                state, 
                stream_mode=["updates", "custom", "messages"],
                config=config
            ):
                if chunk_type == "custom" and isinstance(chunk, dict):
                    if "type" in chunk and "content" in chunk:
                        if chunk["type"] == "profile":
                            final_content["profile_assessment"] += chunk["content"]
                            streamed_overview += chunk["content"]
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk['content']})}\n\n"
                        elif chunk["type"] == "dietary":
                            if not hasattr(stream_response, '_yielded_dietary_newline'):
                                yield f"data: {json.dumps({'type': 'content', 'content': ''})}\n\n"
                                stream_response._yielded_dietary_newline = True
                            final_content["dietary_plan"] += chunk["content"]
                            streamed_overview += chunk["content"]
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk['content']})}\n\n"
                        elif chunk["type"] == "fitness":
                            if not hasattr(stream_response, '_yielded_fitness_newline'):
                                yield f"data: {json.dumps({'type': 'content', 'content': ''})}\n\n"
                                stream_response._yielded_fitness_newline = True
                            final_content["fitness_plan"] += chunk["content"]
                            streamed_overview += chunk["content"]
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk['content']})}\n\n"
                        elif chunk["type"] == "progress":
                            final_content["progress_tracking"] += chunk["content"]
                            streamed_overview += chunk["content"]
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk['content']})}\n\n"
                        elif chunk["type"] == "response":
                            streamed_overview += chunk["content"]
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk['content']})}\n\n"
                        elif chunk["type"] == "step":
                            logging.info(f"Step: {chunk['content']}")
                elif chunk_type == "updates" and isinstance(chunk, dict):
                    if "node" in chunk:
                        node_name = chunk.get("node")
                        logging.info(f"Completed node: {node_name}")
                        if "return_values" in chunk and isinstance(chunk["return_values"], dict):
                            if node_name == "profile" and "user_profile" in chunk["return_values"]:
                                profile_content = chunk["return_values"]["user_profile"]
                                if isinstance(profile_content, str) and profile_content.strip():
                                    final_content["profile_assessment"] = profile_content
                                    streamed_overview += profile_content
                                    yield f"data: {json.dumps({'type': 'content', 'content': profile_content})}\n\n"
                            if node_name == "dietary" and "dietary_state" in chunk["return_values"]:
                                dietary_state = chunk["return_values"]["dietary_state"]
                                if hasattr(dietary_state, "content") and dietary_state.content.strip():
                                    final_content["dietary_plan"] = dietary_state.content
                                    streamed_overview += dietary_state.content
                                    yield f"data: {json.dumps({'type': 'content', 'content':  dietary_state.content})}\n\n"
                            if node_name == "fitness" and "fitness_state" in chunk["return_values"]:
                                fitness_state = chunk["return_values"]["fitness_state"]
                                if hasattr(fitness_state, "content") and fitness_state.content.strip():
                                    final_content["fitness_plan"] = fitness_state.content
                                    streamed_overview += fitness_state.content
                                    yield f"data: {json.dumps({'type': 'content', 'content': fitness_state.content})}\n\n"
                elif chunk_type == "messages":
                    message_chunk, metadata = chunk
                    token = None
                    if hasattr(message_chunk, "content"):
                        token = message_chunk.content
                    elif isinstance(message_chunk, str):
                        token = message_chunk
                    if token:
                        node_name = metadata.get("langgraph_node") if metadata else None
                        if node_name == "profile":
                            final_content["profile_assessment"] += token
                        elif node_name == "dietary":
                            final_content["dietary_plan"] += token
                        elif node_name == "fitness":
                            final_content["fitness_plan"] += token
                        streamed_overview += token
                        yield f"data: {json.dumps({'type': 'content', 'content': token})}\n\n"
            # --- END NEW ---
            # After all main content is streamed, stream the progress tracking section
            from graph.nodes.fitness_coach import HeadCoachAgent
            head_coach = HeadCoachAgent()
            previous_overview = state.get("previous_complete_response", "")
            yield f"data: {json.dumps({'type': 'content', 'content': ''})}\n\n"
            async for token in head_coach.compare_responses(previous_overview, streamed_overview, state["user_profile"]):
                if token:
                    yield f"data: {json.dumps({'type': 'progress', 'content': token})}\n\n"
        except Exception as e:
            logging.error(f"Error during streaming: {str(e)}")
            logging.error(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

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