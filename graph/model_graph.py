from typing import Dict, List, Any, TypedDict, Annotated, Optional, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import StreamWriter
from langgraph.checkpoint.memory import MemorySaver
import uuid
from datetime import datetime

# Import our model state and agent
from graph.model_state import ModelState, Position, Camera
# Import new modular nodes
from graph.nodes.model_graph_nodes import (
    planner_node,
    tool_executor_node,
    responder_node,
    router_agent,
    muscle_control_agent,
    camera_control_agent,
    register_writer,
    clear_writer
)

# Create a memory checkpointer
memory_saver = MemorySaver()

def create_model_graph():
    """Create and return the refactored multi-node LangGraph for the 3-D model."""
    builder = StateGraph(ModelState)

    # Register nodes
    builder.add_node("planner", planner_node)
    builder.add_node("muscle_control", muscle_control_agent)
    builder.add_node("camera_control", camera_control_agent)
    builder.add_node("execute_tools", tool_executor_node)
    builder.add_node("responder", responder_node)
    # Router agent: doesn't modify state, just makes decisions
    builder.add_node("router", router_agent)

    # Entry point
    builder.add_edge(START, "planner")
    
    # After planning, check which specialized agents are needed
    builder.add_conditional_edges(
        "planner",
        lambda state: "muscle_control" if state.get("_route_muscle") else "camera_control" if state.get("_route_camera") else "responder",
        {
            "muscle_control": "muscle_control",
            "camera_control": "camera_control",
            "responder": "responder"
        }
    )
    
    # MODIFIED: Always go from muscle_control to camera_control
    builder.add_edge("muscle_control", "camera_control")
    
    # MODIFIED: Always go from camera_control to execute_tools 
    builder.add_edge("camera_control", "execute_tools")
    
    # MODIFIED: Always go from execute_tools to responder
    builder.add_edge("execute_tools", "responder")

    # Router decides next node based on the _route field it adds to state
    builder.add_conditional_edges(
        "router",
        lambda state: state.get("_route", "responder"),
        {
            "execute_tools": "execute_tools",
            "responder": "responder"
        }
    )

    # Final edge
    builder.add_edge("responder", END)

    # Compile with checkpointing
    return builder.compile(checkpointer=memory_saver)

def create_default_state(thread_id: Optional[str] = None, user_id: Optional[str] = None) -> ModelState:
    """Create a default initial state for the graph.
    
    Args:
        thread_id: Optional thread ID for the conversation
        user_id: Optional user ID for the conversation
    
    Returns:
        A ModelState dictionary with default values
    """
    # Generate IDs if not provided
    thread_id = thread_id or str(uuid.uuid4())
    user_id = user_id or str(uuid.uuid4())
    
    return {
        # Session identifiers
        "thread_id": thread_id,
        "user_id": user_id,
        
        # Conversation state
        "messages": [],
        
        # Model state
        "highlighted_muscles": {},
        "animation": {
            "frame": 0,
            "isPlaying": False
        },
        "camera": {
            "position": {"x": 0, "y": 1, "z": 7},
            "target": {"x": 0, "y": 0, "z": 0}
        },
        "current_agent": "muscle_expert",  # Placeholder, kept for backward compatibility
        "events": [],  # Empty list to collect events
        # New planner/executor fields
        "pending_tool_calls": None,
        "assistant_draft": None,
        # New fields with defaults
        "_route_camera": False,
        "_route_muscle": False,
        "_route": None,
        "_planner_iterations": 0,
        "_tool_executions": 0,
        "_just_executed_muscle_tools": False,
        "pending_muscle_tool_calls": None,
        "pending_camera_tool_calls": None,
        "_control_history": {}
    }

class ModelGraphInterface:
    """Interface class for the model graph to simplify interaction."""
    
    def __init__(self):
        self.graph = create_model_graph()
        self.active_sessions = {}  # Store active sessions by thread_id
    
    def get_or_create_state(self, thread_id: Optional[str] = None, user_id: Optional[str] = None) -> ModelState:
        """Get an existing session or create a new one if it doesn't exist.
        
        Args:
            thread_id: Optional thread ID for the conversation
            user_id: Optional user ID for the conversation
            
        Returns:
            The current or newly created state
        """
        # Generate a new thread_id if not provided
        thread_id = thread_id or str(uuid.uuid4())
        
        print(f"Getting state for thread_id: {thread_id}")
        
        # Check if we have this thread in memory
        if thread_id in self.active_sessions:
            print(f"Found state in active_sessions with {len(self.active_sessions[thread_id].get('messages', []))} messages")
            return self.active_sessions[thread_id]
        
        # Check if we have this thread in persistent storage
        try:
            checkpoint_data = memory_saver.get(thread_id)
            if checkpoint_data:
                state = checkpoint_data["state"]
                print(f"Found state in memory_saver with {len(state.get('messages', []))} messages")
                self.active_sessions[thread_id] = state
                return state
        except Exception as e:
            print(f"Error retrieving from memory_saver: {e}")
            # If retrieval fails, just create a new state
            pass
        
        # Create new state if not found
        print(f"Creating new state for thread_id: {thread_id}")
        new_state = create_default_state(thread_id, user_id)
        self.active_sessions[thread_id] = new_state
        return new_state
    
    async def process_message(self,   # may be deleted afterwards
                         message: str, 
                         thread_id: Optional[str] = None, 
                         user_id: Optional[str] = None,
                         stream_handler=  None) -> Dict[str, Any]:
        """Process a user message through the graph.
        
        Args:
            message: The user message text
            thread_id: Optional thread ID for the conversation
            user_id: Optional user ID for the conversation
            stream_handler: Optional callback function to handle streaming updates
        
        Returns:
            Dict containing updated state and response
        """
        # Get or create state for this session
        state = self.get_or_create_state(thread_id, user_id)
        thread_id = state["thread_id"]
        
        # Use provided user_id or keep existing one
        if user_id and state["user_id"] != user_id:
            state["user_id"] = user_id
            
        # Add the user message to the state
        messages = state.get("messages", []).copy()
        print(f"Before adding message, message count: {len(messages)}")
        messages.append({
            "role": "user", 
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Create input state with the new message
        input_state = state.copy()
        input_state["messages"] = messages
        
        print(f"Initial state before processing: messages={len(input_state.get('messages', []))}, events={input_state.get('events', [])}")
        print(f"Stream handler available: {stream_handler is not None}")
        
        # Configure stream mode
        stream_modes = ["custom", "updates"]
        
        # Create config with thread_id for checkpointing
        config = {
            "configurable": {
                "thread_id": state["thread_id"]
            }
        }
        
        # Process the message through the graph
        if stream_handler:
            # Register the stream handler with our writer registry
            register_writer(thread_id, stream_handler)
            print(f"Registered stream handler with writer registry for thread {thread_id}")
            
            try:
                # Use streaming and call the handler for each chunk
                print(f"Processing with default langgraph stream_handler")
                # Stream with context
                async for stream_mode, chunk in self.graph.astream(
                    input_state,
                    config=config,
                    stream_mode=stream_modes,
                ):
                    print(f"Stream mode: {stream_mode}, chunk: {chunk}")
                    if chunk is None:
                        continue
                    # Await the async stream_handler!
                    await stream_handler(stream_mode, chunk)
                # Get the final state
                final_state = await self.graph.aget_state(config=config)
            finally:
                # Clean up the registry
                clear_writer(thread_id)
                print(f"Cleared stream handler from registry for thread {thread_id}")
        else:
            # Process without streaming
            print(f"Processing without stream_handler")
            final_state = await self.graph.ainvoke(input_state, config=config)
        # --- UNPACK TUPLE IF NEEDED ---
        if isinstance(final_state, tuple):
            for item in final_state:
                if isinstance(item, dict) and "thread_id" in item:
                    final_state = item
                    break
        #print(f"Final state after processing: messages={len(final_state.get('messages', []))}")
        
        # Update our maintained state and active_sessions
        self.active_sessions[final_state["thread_id"]] = final_state
        
        # Prepare the response
        latest_messages = final_state.get("messages", [])
        response = None
        if latest_messages and latest_messages[-1]["role"] == "assistant":
            response = latest_messages[-1]["content"]
            
            # Add timestamp to assistant message if missing
            if "timestamp" not in latest_messages[-1]:
                latest_messages[-1]["timestamp"] = datetime.now().isoformat()
        
        # Extract any events that need to be sent to the frontend
        events = final_state.get("events", [])
        print(f"Extracted events for frontend: {events}")
        
        # Clear events from state after processing
        cleared_events_state = final_state.copy()
        cleared_events_state["events"] = []
        self.active_sessions[final_state["thread_id"]] = cleared_events_state
        
        # Save to memory checkpointer to persist
        try:
            memory_saver.put(final_state["thread_id"], {"state": cleared_events_state})
            print(f"Persisted state to memory_saver with {len(cleared_events_state.get('messages', []))} messages")
        except Exception as e:
            print(f"Error persisting to memory_saver: {e}")
        
        print(f"Response to user: {response}")
        print(f"Events to send to frontend: {events}")
        
        return {
            "response": response,
            "events": events,
            "thread_id": final_state["thread_id"],
            "user_id": final_state["user_id"],
            "state": final_state
        }
    
    def get_state(self, thread_id: str) -> Optional[ModelState]:
        """Get the current state for a specific thread ID.
        
        Args:
            thread_id: The thread ID to retrieve state for
            
        Returns:
            The state if found, None otherwise
        """
        # Check memory first
        if thread_id in self.active_sessions:
            print(f"Found state in active_sessions with {len(self.active_sessions[thread_id].get('messages', []))} messages")
            return self.active_sessions[thread_id]
        
        # If not in memory, try to load from persistent storage
        try:
            checkpoint_data = memory_saver.get(thread_id)
            if checkpoint_data:
                state = checkpoint_data["state"]
                print(f"Found state in memory_saver with {len(state.get('messages', []))} messages")
                self.active_sessions[thread_id] = state
                return state
        except Exception as e:
            print(f"Error retrieving from memory_saver: {e}")
            # If retrieval fails, return None
            return None
    
    def reset(self, thread_id: Optional[str] = None, user_id: Optional[str] = None) -> ModelState:
        """Reset the state to default values.
        
        Args:
            thread_id: Optional thread ID to keep for continuity
            user_id: Optional user ID to keep for continuity
            
        Returns:
            The new default state
        """
        new_state = create_default_state(thread_id, user_id)
        
        # Update memory
        if thread_id:
            self.active_sessions[thread_id] = new_state
            # Update persistent storage
            try:
                memory_saver.put(thread_id, {"state": new_state})
                print(f"Reset and persisted new state to memory_saver")
            except Exception as e:
                print(f"Error persisting reset state to memory_saver: {e}")
            
        return new_state

# Create a singleton instance for easy import
model_graph = ModelGraphInterface() 
