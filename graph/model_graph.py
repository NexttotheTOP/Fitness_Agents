from typing import Dict, List, Any, TypedDict, Annotated, Optional, Union, Callable, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import StreamWriter
from langgraph.checkpoint.memory import MemorySaver
import uuid
from datetime import datetime

# Import our model state and agent
from graph.model_state import ModelState, Position, Camera
from graph.nodes.model_agents import conversation_agent, tool_agent

# Create a memory checkpointer
memory_saver = MemorySaver()

def should_route_to_tool_agent(state: ModelState) -> Literal["conversation_agent", "tool_agent"]:
    """
    Determines if the current request should be routed to the tool agent.
    Returns the name of the next node to route to.
    """
    if state.get("agent_request"):
        print("Routing to tool agent because agent_request is present")
        return "tool_agent"
    else:
        print("Routing to conversation agent as default")
        return "conversation_agent"

def create_model_graph():
    """Create and return a proper two-agent graph with routing between agents."""
    # Initialize the state graph with our ModelState schema
    builder = StateGraph(ModelState)
    
    # Add both agent nodes
    builder.add_node("conversation_agent", conversation_agent)
    builder.add_node("tool_agent", tool_agent)
    
    # Define the edges - conversation agent is the main entry point
    builder.add_edge(START, "conversation_agent")
    
    # Conditional routing between agents based on state
    builder.add_conditional_edges(
        "conversation_agent",
        should_route_to_tool_agent,
        {
            "conversation_agent": END,  # If no tool needed, we're done
            "tool_agent": "tool_agent",  # Route to tool agent if needed
        }
    )
    
    # Tool agent always routes back to conversation agent to respond to the user
    builder.add_edge("tool_agent", "conversation_agent")
    
    # Compile the graph with the memory checkpointer
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
        
        # These fields are optional in the TypedDict, no need to initialize
        "current_agent": "conversation_agent",
        "agent_request": None,
        "user_question": None,
        "tool_agent_report": None,
        "events": []  # Empty list to collect events
    }

class ModelGraphInterface:
    """Interface class for the model graph to simplify interaction."""
    
    def __init__(self):
        self.graph = create_model_graph()
        self.active_sessions = {}  # Store active sessions by thread_id
    
    def get_or_create_state(self, thread_id: Optional[str] = None, user_id: Optional[str] = None) -> ModelState:
        """Get an existing session or create a new one if it doesn't exist."""
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
    
    def process_message(self, 
                         message: str, 
                         thread_id: Optional[str] = None, 
                         user_id: Optional[str] = None,
                         stream_handler=None) -> Dict[str, Any]:
        """Process a user message through the graph."""
        # Get or create state for this session
        state = self.get_or_create_state(thread_id, user_id)
        
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
            # Use streaming and call the handler for each chunk
            for stream_mode, chunk in self.graph.stream(
                input_state,
                config=config,
                stream_mode=stream_modes
            ):
                if chunk is None:
                    continue
                if stream_handler(stream_mode, chunk):
                    # Pass through any non-None values returned by the handler
                    pass
            
            # Get the final state
            final_state = self.graph.get_state()
        else:
            # Process without streaming
            final_state = self.graph.invoke(input_state, config=config)
        
        print(f"Final state after processing: messages={len(final_state.get('messages', []))}")
        
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
            memory_saver.put(
                final_state["thread_id"], 
                {
                    "state": cleared_events_state
                },
                {"metadata": {}, "new_versions": {}}
            )
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
        """Get the current state for a specific thread ID."""
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
        """Reset the state to default values."""
        new_state = create_default_state(thread_id, user_id)
        
        # Update memory
        if thread_id:
            self.active_sessions[thread_id] = new_state
            # Update persistent storage
            try:
                memory_saver.put(
                    thread_id, 
                    {"state": new_state},
                    {"metadata": {}, "new_versions": {}}
                )
                print(f"Reset and persisted new state to memory_saver")
            except Exception as e:
                print(f"Error persisting reset state to memory_saver: {e}")
            
        return new_state

# Create a singleton instance for easy import
model_graph = ModelGraphInterface() 

def stream_handler(stream_mode, chunk):
    if chunk is None:
        return None
    # ... rest of your code ... 