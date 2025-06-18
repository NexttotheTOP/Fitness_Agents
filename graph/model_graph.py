from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import uuid
from datetime import datetime
from graph.model_state import ModelState, Position, Camera
from graph.nodes.model_graph_nodes import (
    planner_node,
    tool_executor_node,
    responder_node,
    #router_agent,
    muscle_control_agent,
    camera_control_agent,
    register_writer,
    clear_writer
)

# Create a memory checkpointer
memory_saver = MemorySaver()

def create_model_graph():
    builder = StateGraph(ModelState)

    builder.add_node("planner", planner_node)
    builder.add_node("muscle_control", muscle_control_agent)
    builder.add_node("camera_control", camera_control_agent)
    builder.add_node("execute_tools", tool_executor_node)
    builder.add_node("responder", responder_node)
    #builder.add_node("router", router_agent)

    builder.add_edge(START, "planner")
    
    # After planning, check which specialized agents are needed
    builder.add_conditional_edges(
        "planner",
        lambda state: state.get("_route", "responder"),
        {
            "muscle_control": "muscle_control",
            "responder": "responder"
        }
    )
    
    builder.add_edge("muscle_control", "camera_control")
    
    builder.add_edge("camera_control", "execute_tools")
    
    builder.add_edge("execute_tools", "responder")

    # builder.add_conditional_edges(
    #     "router",
    #     lambda state: state.get("_route", "responder"),
    #     {
    #         "execute_tools": "execute_tools",
    #         "responder": "responder"
    #     }
    # )

    builder.add_edge("responder", END)

    return builder.compile(checkpointer=memory_saver)

def create_default_state(thread_id: Optional[str] = None, user_id: Optional[str] = None) -> ModelState:
    thread_id = thread_id or str(uuid.uuid4())
    user_id = user_id or str(uuid.uuid4())
    
    return {
        "thread_id": thread_id,
        "user_id": user_id,
        "messages": [],
        "highlighted_muscles": {},
        "animation": {
            "frame": 0,
            "isPlaying": False
        },
        "camera": {
            "position": {"x": 0, "y": 1, "z": 7},
            "target": {"x": 0, "y": 0, "z": 0}
        },
        "current_agent": "muscle_expert",
        "events": [],
        "pending_tool_calls": None,
        "assistant_draft": None,
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
    def __init__(self):
        self.graph = create_model_graph()
        self.active_sessions = {}
    
    def get_or_create_state(self, thread_id: Optional[str] = None, user_id: Optional[str] = None) -> ModelState:
        thread_id = thread_id or str(uuid.uuid4())
        print(f"Getting state for thread_id: {thread_id}")

        if thread_id in self.active_sessions:
            print(f"Found state in active_sessions with {len(self.active_sessions[thread_id].get('messages', []))} messages")
            return self.active_sessions[thread_id]
        
        try:
            checkpoint_data = memory_saver.get(thread_id)
            if checkpoint_data:
                state = checkpoint_data["state"]
                print(f"Found state in memory_saver with {len(state.get('messages', []))} messages")
                self.active_sessions[thread_id] = state
                return state
        except Exception as e:
            print(f"Error retrieving from memory_saver: {e}")
            pass
        
        print(f"Creating new state for thread_id: {thread_id}")
        new_state = create_default_state(thread_id, user_id)
        self.active_sessions[thread_id] = new_state
        return new_state
    
    async def process_message(self,
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
        state = self.get_or_create_state(thread_id, user_id)
        thread_id = state["thread_id"]
        
        if user_id and state["user_id"] != user_id:
            state["user_id"] = user_id
            
        messages = state.get("messages", []).copy()
        print(f"Before adding message, message count: {len(messages)}")
        messages.append({
            "role": "user", 
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        input_state = state.copy()
        input_state["messages"] = messages
        
        print(f"Initial state before processing: messages={len(input_state.get('messages', []))}, events={input_state.get('events', [])}")
        print(f"Stream handler available: {stream_handler is not None}")
        
        stream_modes = ["custom", "updates"]
        
        config = {
            "configurable": {
                "thread_id": state["thread_id"]
            }
        }
        
        if stream_handler:
            register_writer(thread_id, stream_handler)
            print(f"Registered stream handler with writer registry for thread {thread_id}")
            
            try:
                print(f"Processing with default langgraph stream_handler")
                async for stream_mode, chunk in self.graph.astream(
                    input_state,
                    config=config,
                    stream_mode=stream_modes,
                ):
                    print(f"Stream mode: {stream_mode}, chunk: {chunk}")
                    if chunk is None:
                        continue
                    await stream_handler(stream_mode, chunk)
                final_state = await self.graph.aget_state(config=config)
            finally:
                clear_writer(thread_id)
                print(f"Cleared stream handler from registry for thread {thread_id}")
        else:
            print(f"Processing without stream_handler")
            final_state = await self.graph.ainvoke(input_state, config=config)
        # --- UNPACK TUPLE IF NEEDED ---
        if isinstance(final_state, tuple):
            for item in final_state:
                if isinstance(item, dict) and "thread_id" in item:
                    final_state = item
                    break
        #print(f"Final state after processing: messages={len(final_state.get('messages', []))}")
        
        self.active_sessions[final_state["thread_id"]] = final_state
        
        latest_messages = final_state.get("messages", [])
        response = None
        if latest_messages and latest_messages[-1]["role"] == "assistant":
            response = latest_messages[-1]["content"]
            
            if "timestamp" not in latest_messages[-1]:
                latest_messages[-1]["timestamp"] = datetime.now().isoformat()
        
        events = final_state.get("events", [])
        print(f"Extracted events for frontend: {events}")
        
        cleared_events_state = final_state.copy()
        cleared_events_state["events"] = []
        self.active_sessions[final_state["thread_id"]] = cleared_events_state
        
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
        if thread_id in self.active_sessions:
            print(f"Found state in active_sessions with {len(self.active_sessions[thread_id].get('messages', []))} messages")
            return self.active_sessions[thread_id]
        
        try:
            checkpoint_data = memory_saver.get(thread_id)
            if checkpoint_data:
                state = checkpoint_data["state"]
                print(f"Found state in memory_saver with {len(state.get('messages', []))} messages")
                self.active_sessions[thread_id] = state
                return state
        except Exception as e:
            print(f"Error retrieving from memory_saver: {e}")
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
        
        if thread_id:
            self.active_sessions[thread_id] = new_state
            try:
                memory_saver.put(thread_id, {"state": new_state})
                print(f"Reset and persisted new state to memory_saver")
            except Exception as e:
                print(f"Error persisting reset state to memory_saver: {e}")
            
        return new_state

model_graph = ModelGraphInterface() 
