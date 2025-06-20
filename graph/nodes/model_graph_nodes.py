from __future__ import annotations

"""Shared nodes used in the refactored 3-D model LangGraph.

This file contains multiple nodes:
1. planner_node – determines which specialized agents to call based on user request
2. muscle_control_agent - specialized agent for handling muscle highlighting  
3. camera_control_agent - specialized agent for handling camera positioning
4. tool_executor_node – executes the pending tool calls deterministically
5. responder_node – produces the final conversational reply
6. router_agent - makes routing decisions to break loops

The nodes are designed to be imported by graph/model_graph.py when building the
LangGraph state machine.
"""

from typing import Any, Dict, List, Optional, Literal
import copy
from langgraph.types import StreamWriter
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json
from datetime import datetime

from graph.model_state import ModelState
from graph.nodes.model_agents import (
    RESPONSE_PROMPT,
    TOOL_MAP,
    MUSCLE_MAPPING_STR,
    MUSCLE_PAIRING_RULES,
    MUSCLE_NAMING_RULES,
    FUNCTIONAL_GROUPS_STR,
    select_muscles_tool,
    toggle_muscle_tool,
    set_camera_view_tool,
    reset_camera_tool,
)

# Import shared state for Socket.IO integration - use emit_event instead of our custom function
from graph.shared_state import thread_to_sid, get_sid, emit_event

# Define constants for specialized tools
MUSCLE_CONTROL_TOOLS = [
    select_muscles_tool,
    toggle_muscle_tool,
]

CAMERA_CONTROL_TOOLS = [
    set_camera_view_tool,  # New combined camera tool
    reset_camera_tool
]

# Constants for specialized agent prompts
MUSCLE_AGENT_PROMPT = """
[Persona]
You are a specialized muscle control agent for a 3D anatomy model. Your role is to control the tool that highlights muscles in the frontend in response to user and agent requests.
You get excited about using the muscle highlighting tools and camera controls to create the perfect demonstration!

[Core Principles]
- Use the tools to control the 3D model to show muscles in the context of workouts, training, or general fitness questions
- Focus on highlighting muscles that are most relevant to the user's query, whether specific or general
- By default, only highlight the LEFT side of muscles unless specifically asked to show both sides
- Use different colors for each muscle to help users distinguish between them

[Task]
- Proactively demonstrate relevant muscles and anatomy based on user queries using the 3D model
- For specific muscle questions, highlight those exact muscles
- For general area questions (e.g., "back"), highlight the primary muscles in that area
- For complex questions requiring multiple steps, first explain your approach
- Use the [Available Muscles] section as a reference for muscle names and groupings, but prioritize relevance to the query
- When users mention common names (e.g., "bicep"), use your knowledge to map to the appropriate muscle(s)
- Use distinct, visually clear colors for each muscle (unless the user requests a specific color)

[Naming Instructions]
- Use PascalCase with underscores for all muscle names (e.g., Zygomaticus_Major, Pectoralis_Major_01_Clavicular)
- For right-side muscles, append _R (e.g., Gluteus_Maximus_R)
- For left-side muscles, use the base name with NO suffix (e.g., Gluteus_Maximus)
- Do NOT use _L, spaces, lowercase, or any other formats

[Available Muscles Reference]
The following is a reference of muscles in the 3D model, organized by region. Use this as a guide for naming conventions and available muscles:
{MUSCLE_MAPPING_STR}

[Functional Muscle Groups Reference]
These groups can help identify muscles involved in specific movements or exercises:
{FUNCTIONAL_GROUPS_STR}

[Muscle Pairing Rules]
{MUSCLE_PAIRING_RULES}

[Naming Rules]
{MUSCLE_NAMING_RULES}

[Color Assignment Guidelines]
- UNIQUE COLORS: Assign a DIFFERENT color to each individual muscle to help users distinguish them
- HIGH CONTRAST: Use vibrant, distinct colors that stand out against each other
- COLOR SUGGESTIONS (but feel free to use any visually clear hex colors):
  • "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", 
  • "#FF8800", "#8800FF", "#00FF88", "#FF0088", "#8888FF", "#FFFF88"

[Tool Usage Instructions]
- **select_muscles(muscle_names: list, colors: dict)**: Highlight specific muscles. The `colors` argument MUST be a dictionary mapping each muscle name to a hex color.
info: Each time you sent a new list of muscles to highlight, the frontend will automatically deselect all muscles that are not in the new list.

[REQUIRED OUTPUT FORMAT]
For the select_muscles tool, your call MUST include:
1. A list of muscle names that are most relevant to the user's query (defaulting to left side)
2. A colors dictionary mapping EACH muscle name to a UNIQUE hex color code

Remember: The goal is to help users visually understand the muscles relevant to their query.

You MUST use the select_muscles tool to highlight muscles related to the user's request.
Do not generate text-only responses - always include the appropriate tool call.
Even for simple information requests, if muscles are relevant, you must highlight them.
"""

CAMERA_AGENT_PROMPT = """
You are a specialized camera control agent for a 3D anatomy model. Your ONLY role is to position the camera to provide the best view of muscles.

[Camera Control Guidelines]
For the clearest demonstrations, use these specific presets based on the muscle group:

Upper Body Front View (for chest, biceps, abs):
- Position: x: -0.03, y: 0.83, z: 3.48
- Target: x: -0.03, y: 0.83, z: ~0

Upper Body Back View (for back, shoulders):
- Position: x: 0.20, y: 1.53, z: -3.70
- Target: x: 0.07, y: 0.77, z: 0.16

Lower Body Front View (for quads, calves):
- Position: x: -0.0007, y: -0.50, z: 4.45
- Target: x: 0.0006, y: -0.50, z: 0.00006

Lower Body Back View (for glutes, hamstrings):
- Position: x: 0.20, y: 0.26, z: -4.21
- Target: x: 0.06, y: -0.56, z: -0.11

[Tools]
- `set_camera_view(position_x, position_y, position_z, target_x, target_y, target_z)`: Set both camera position and look-at target in one call
- `reset_camera()`: Reset to default view

Study the highlighted muscles and position the camera to provide the best view of those muscles.
"""

# Shared LLM instance for all nodes (low-temperature for determinism)
_llm_streaming = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, streaming=True)

# Debug mode for verbose logging
DEBUG_MODE = True

# Global writer registry for temporary storage of active writers
_active_writers = {}

def register_writer(thread_id, writer):
    """Register a writer function for a specific thread_id to be used by nodes."""
    global _active_writers
    _active_writers[thread_id] = writer
    print(f"Registered writer for thread {thread_id}")
    
def get_writer(thread_id):
    """Get a previously registered writer or return None if not available."""
    global _active_writers
    writer = _active_writers.get(thread_id)
    if writer:
        print(f"Retrieved registered writer for thread {thread_id}")
    return writer
    
def clear_writer(thread_id):
    """Clear a writer after use to prevent memory leaks."""
    global _active_writers
    if thread_id in _active_writers:
        del _active_writers[thread_id]
        print(f"Cleared writer for thread {thread_id}")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _format_state_for_prompt(state: ModelState) -> str:
    """Return a short, human-readable snapshot of the current 3-D model state."""
    highlighted_muscles = state.get("highlighted_muscles", {})
    highlighted_muscles_str = (
        ", ".join(f"{m} (color: {c})" for m, c in highlighted_muscles.items())
        if highlighted_muscles
        else "None"
    )
    anim = state.get("animation", {"frame": 0, "isPlaying": False})
    camera = state.get("camera", {"position": {}, "target": {}})

    return (
        f"Highlighted muscles: {highlighted_muscles_str}\n"
        f"Animation: frame {anim.get('frame', 0)}, playing {anim.get('isPlaying', False)}\n"
        f"Camera position: {camera.get('position', {})}, target: {camera.get('target', {})}"
    )



# ROUTER_SYSTEM_PROMPT = """You are a routing agent for a 3D muscle model system. Your job is to decide the next step in processing.

# You have 2 possible routes:
# 1. "execute_tools" - Execute pending tool calls that will manipulate the 3D model
# 2. "responder" - Generate a final text response to the user

# [Request Fulfillment Analysis]
# - Your primary task is to determine if the user's request has already been fulfilled
# - Examine the current state (highlighted muscles, camera position) 
# - Compare with what the user asked for
# - Analyze pending tool calls to see if they would provide meaningful changes

# [Rules for Decision Making]
# - Choose "execute_tools" if:
#   * The user's request has NOT been fulfilled yet
#   * The pending tool calls would make meaningful changes to fulfill the request
#   * The model's current state doesn't match what the user asked for

# - Choose "responder" if:
#   * The user's request has already been fulfilled (muscles highlighted, camera positioned)
#   * The pending tool calls are redundant or would undo already completed work
#   * Further tool execution wouldn't provide additional value to the user

# [Analysis Guidelines]
# - Understand what the user actually requested (e.g., "highlight chest muscles")
# - Check if the muscles relevant to that request are already highlighted
# - Verify if the camera position is appropriate for viewing those muscles
# - Consider if the pending tool calls would improve the state or just alter it meaninglessly

# You must respond ONLY with one of these exact strings: "execute_tools" or "responder"
# """

# async def router_agent(
#     state: ModelState,
#     writer: Optional[StreamWriter] = None,
#     context: Optional[Dict[str, Any]] = None,
# ) -> ModelState:
#     """Smart router agent that decides whether to execute tools or go to responder.
    
#     This agent analyzes the current state and pending tool calls to make an intelligent
#     routing decision, preventing loops while ensuring user requests are fulfilled.
    
#     Returns:
#         Updated ModelState with a new _route field containing the routing decision.
#     """
#     # Copy the state to avoid mutations
#     new_state = copy.deepcopy(state)
    
#     # Try to get writer from context if not directly provided
#     if writer is None and context and "writer" in context:
#         writer = context["writer"]
#         print(f"[router_agent] Got writer from context!")
    
#     # Extract iteration count and pending tools info
#     iterations = new_state.get("_planner_iterations", 0)
#     pending_tools = new_state.get("pending_tool_calls", []) or []  # Handle None case
    
#     # Get the current highlighted muscles and camera state for analysis
#     highlighted_muscles = new_state.get("highlighted_muscles", {}) or {}  # Handle None case
#     camera = new_state.get("camera", {}) or {}  # Handle None case
#     muscles_str = ", ".join(f"{name}" for name in highlighted_muscles.keys()) if highlighted_muscles else "none"
    
#     # Get the last user query - this is crucial for understanding what was requested
#     last_query = ""
#     messages = new_state.get("messages", []) or []  # Handle None case
#     for msg in reversed(messages):
#         if msg.get("role") == "user":
#             last_query = msg.get("content", "")
#             break
    
#     # Get list of events to understand what has already been done
#     events = new_state.get("events", []) or []  # Handle None case
#     event_types = set(e.get("type", "") for e in events if isinstance(e, dict))
    
#     # Check if assistant_draft is present - important for correctly deciding if we need to respond
#     has_assistant_draft = bool(new_state.get("assistant_draft", "").strip())
    
#     # Fast path decisions for simple cases
    
#     # 1. No pending tools - go straight to responder
#     if not pending_tools:
#         print(f"[router_agent] No pending tools, routing to responder")
#         new_state["_route"] = "responder"
#         return new_state
    
#     # 2. Safety check for excessive iterations
#     if iterations >= 8:
#         print(f"[router_agent] High iteration count ({iterations}), forcing responder")
#         new_state["_route"] = "responder"
#         return new_state
        
#     # 3. Check if we've completed the user's request based on specific request type
    
#     # Analyze the request type
#     request_involves_muscles = any(term in last_query.lower() for term in 
#                                   ["muscle", "highlight", "show", "display", "chest", "arm", "leg", "back", "shoulder",
#                                    "bicep", "tricep", "quad", "core", "abs", "glute"])
                                   
#     request_involves_camera = any(term in last_query.lower() for term in 
#                                  ["camera", "view", "angle", "look", "position", "rotate", "turn", "face", "see"])
    
#     # For muscle requests - check if we have appropriate muscles highlighted
#     if request_involves_muscles and highlighted_muscles:
#         # Check for specific muscle groups mentioned in the query
#         if "chest" in last_query.lower() and any("pectoralis" in m.lower() for m in highlighted_muscles):
#             print(f"[router_agent] Chest muscles already highlighted, likely fulfilled request")
#             if all(call.get("name") == "select_muscles" for call in pending_tools):
#                 new_state["_route"] = "responder"
#                 return new_state
                
#         if "back" in last_query.lower() and any(m.lower() in ["latissimus_dorsi", "trapezius", "rhomboideus"] for m in highlighted_muscles):
#             print(f"[router_agent] Back muscles already highlighted, likely fulfilled request")
#             if all(call.get("name") == "select_muscles" for call in pending_tools):
#                 new_state["_route"] = "responder"
#                 return new_state
                
#         if "leg" in last_query.lower() and any("femor" in m.lower() or "tibia" in m.lower() or "gluteus" in m.lower() for m in highlighted_muscles):
#             print(f"[router_agent] Leg muscles already highlighted, likely fulfilled request")
#             if all(call.get("name") == "select_muscles" for call in pending_tools):
#                 new_state["_route"] = "responder"
#                 return new_state
    
#     # Check for redundant tool calls
    
#     # Check if we've already performed multiple muscle selection operations
#     muscle_selection_count = sum(1 for e in events if e.get("type") == "model:selectMuscles")
#     if muscle_selection_count >= 2 and request_involves_muscles:
#         print(f"[router_agent] Already performed {muscle_selection_count} muscle selections, seems redundant")
#         # Let's check pending tools - if they're also muscle selections, likely redundant
#         if any(call.get("name") == "select_muscles" for call in pending_tools):
#             new_state["_route"] = "responder"
#             return new_state
    
#     # Analyze if pending tools would toggle already highlighted muscles (redundant)
#     if all(call.get("name") == "toggle_muscle" for call in pending_tools) and highlighted_muscles:
#         toggle_targets = [call.get("args", {}).get("muscle_name") for call in pending_tools 
#                          if call.get("name") == "toggle_muscle"]
#         # If all toggle targets are already highlighted, this is redundant
#         if all(target in highlighted_muscles for target in toggle_targets if target):
#             print(f"[router_agent] Pending toggles would just unhighlight already highlighted muscles - redundant")
#             new_state["_route"] = "responder" 
#             return new_state
    
#     # Create a detailed analysis of the current state for the LLM
#     detailed_state = f"""
# [User's Request]
# {last_query}

# [Current State]
# - Currently highlighted muscles: {muscles_str}
# - Camera position: {camera.get('position', {})}
# - Camera target: {camera.get('target', {})}
# - Iteration count: {iterations}

# [Events Already Executed]
# - Types of events already run: {list(event_types)}
# - Total events: {len(events)}
# - Has muscle selection events: {any(e.get('type') == 'model:selectMuscles' for e in events)}
# - Has camera position events: {any(e.get('type') == 'model:setCameraPosition' for e in events)}
# - Has camera target events: {any(e.get('type') == 'model:setCameraTarget' for e in events)}

# [Pending Tool Calls]
# - Number of pending tools: {len(pending_tools)}
# - Tool types: {[call.get('name') for call in pending_tools]}

# [Request Analysis]
# - Request involves muscles: {request_involves_muscles}
# - Request involves camera: {request_involves_camera}
# - Has assistant draft: {has_assistant_draft}
# """
    
#     # Build the prompt
#     system = SystemMessage(content=ROUTER_SYSTEM_PROMPT)
#     human = HumanMessage(content=detailed_state)
    
#     if writer:
#         await writer({"type": "thinking", "content": "Deciding if request is fulfilled..."})
    
#     try:
#         # Call the LLM for the final decision
#         llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=10, streaming=True)
#         response = await llm.ainvoke([system, human])
        
#         # Extract the decision (only accept the exact strings we need)
#         decision_text = response.content.strip().lower() if hasattr(response, "content") else ""
        
#         # Default to responder if we can't get a clear decision
#         if "execute" in decision_text and not "do not execute" in decision_text:
#             decision = "execute_tools"
#         else:
#             decision = "responder"
#     except Exception as e:
#         print(f"[router_agent] Error calling LLM: {e}, defaulting to responder")
#         decision = "responder"
    
#     print(f"[router_agent] Decision: {decision} (iterations={iterations}, events={len(events)})")
    
#     # Add routing decision to state
#     new_state["_route"] = decision
    
#     return new_state



async def planner_node(
    state: ModelState,
    writer: Optional[StreamWriter] = None,
    context: Optional[Dict[str, Any]] = None,
) -> ModelState:

    new_state: ModelState = copy.deepcopy(state)

    if writer is None and context and "writer" in context:
        writer = context["writer"]
        print(f"[planner_node] Got writer from context!")

    messages_history = new_state.get("messages", []).copy()
    if not messages_history or messages_history[-1]["role"] != "user":
        return new_state

    user_msg = messages_history[-1]["content"]
    print(f"[planner_node] Processing user message: {user_msg}")

    muscles = new_state.get("highlighted_muscles", {})
    muscle_str = ", ".join(f"{m} ({c})" for m, c in muscles.items()) if muscles else "none"
    camera = new_state.get("camera", {})
    camera_str = f"Position: {camera.get('position', {})}, Target: {camera.get('target', {})}"
    control_history = new_state.get("_control_history", {})
    current_request_id = f"req_{len(messages_history)}"



    planner_prompt = """
You route user messages on a fitness-focused 3D muscle-model page.

[INPUTS]
**CURRENT 3D MODEL STATE**  
- Highlighted muscles
- Camera position and target

**CONVERSATION HISTORY**  
   The full back-and-forth of your ongoing chat including the user's latest message.

[Actions]
1. Muscle Control – highlight / select / toggle muscles (camera auto-frames).
2. Direct Response – text only, **no model changes** (use only for pure small-talk/far off-topic).

[Routing Policy]
- The page's purpose is interactive anatomy for fitness; show muscles ~99 % of the time.
- Default to **Muscle Control** whenever a muscle, body part, exercise, or anatomy concept is mentioned or implied.

you should basically always route to muscle control unless the muscles are already highlighted and still talking about the exact same muscles or greets without anything else.

[Return JSON exactly]
{
  "muscle_control": true | false,
  "reason": "short rationale"
}
"""

    # Build conversation history as Human/Assistant messages
    conversation_messages = []
    for msg in messages_history[:-1]:
        role = msg.get("role")
        if role == "user":
            conversation_messages.append(HumanMessage(content=f"{msg['content']}"))
        elif role == "assistant":
            conversation_messages.append(SystemMessage(content=f"{msg['content']}"))

    # Append the latest user message as a HumanMessage (like responder_node)
    if messages_history and messages_history[-1]["role"] == "user":
        conversation_messages.append(HumanMessage(content=messages_history[-1]["content"]))

    # Compose the system messages for the LLM
    system_msgs = [
        SystemMessage(content=planner_prompt),
        SystemMessage(content="[CURRENT 3D MODEL STATE]\n" +
            f"- Highlighted muscles:\n {muscle_str}\n" +
            f"- Camera position and target:\n {camera_str}\n" 

        ),
        SystemMessage(content="[CONVERSATION HISTORY]"),
        *conversation_messages,
    ]

    if writer:
        await writer({"type": "thinking", "content": "Analyzing your request to determine what's needed..."})

    routing_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=500, streaming=True)

    try:
        response = await routing_llm.ainvoke(system_msgs)
        response_text = response.content.strip()
        print(f"[planner_node] Router LLM response: {response_text}")
        import json
        if '{' in response_text and '}' in response_text:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_str = response_text[json_start:json_end]
            routing_decision = json.loads(json_str)
            muscle_control = routing_decision.get("muscle_control", False)
            print(f"[planner_node] Routing decision: muscle_control={muscle_control}")
        else:
            print(f"[planner_node] Failed to parse JSON from LLM response, defaulting to muscle_control")
            muscle_control = True
    except Exception as e:
        print(f"[planner_node] Error in routing LLM: {e}, defaulting to muscle_control")
        muscle_control = True

    if muscle_control:
        new_state["_route"] = "muscle_control"
    else:
        new_state["_route"] = "responder"

    new_state["assistant_draft"] = f"I'm analyzing your question about {user_msg}."
    return new_state


def get_color_for_muscle(muscle_name: str) -> str:
    """Generate a consistent color for a muscle based on its name."""
    # Simple mapping of muscle groups to colors
    if any(term in muscle_name.lower() for term in ["pector", "delt", "bicep", "chest"]):
        return "#FF5555"  # Red for chest/shoulders/biceps
    elif any(term in muscle_name.lower() for term in ["back", "lat", "trapezius", "rhomb"]):
        return "#5555FF"  # Blue for back
    elif any(term in muscle_name.lower() for term in ["gluteus", "glute", "hamstring", "quad", "leg"]):
        return "#FFFF55"  # Yellow for lower body
    elif any(term in muscle_name.lower() for term in ["abs", "rectus"]):
        return "#55FF55"  # Green for abs
    else:
        # Generate a color based on hash of muscle name
        hash_val = hash(muscle_name)
        r = (hash_val & 0xFF0000) >> 16
        g = (hash_val & 0x00FF00) >> 8
        b = hash_val & 0x0000FF
        return f"#{r:02x}{g:02x}{b:02x}"

async def tool_executor_node(
    state: ModelState,
    writer: Optional[StreamWriter] = None,
    context: Optional[Dict[str, Any]] = None,
) -> ModelState:
    """Execute pending tool calls from specialized agents and handle state updates.
    
    The tool executor node serves several critical functions:
    1. Aggregates tool calls from both muscle and camera agents
    2. Executes them in a deterministic, consistent order
    3. Manages state updates (highlighted_muscles, camera positions)
    4. Deduplicates events to prevent redundant frontend updates
    5. Streams events to the frontend
    6. Tracks execution history to prevent loops
    
    While specialized agents could execute their own tools, centralizing execution:
    - Ensures consistent state management
    - Prevents race conditions between agents
    - Provides a single place for event deduplication and tracking
    - Makes debugging and logging simpler
    
    Args:
        state: The current state dictionary
        writer: Optional streaming writer for token-by-token output
        context: Optional context with additional parameters
        
    Returns:
        Updated state with tool results and events
    """
    new_state: ModelState = copy.deepcopy(state)
    
    # Get thread_id upfront for logging and socket.io
    thread_id = new_state.get("thread_id")
    
    # Try to get writer from context if not directly provided
    if writer is None and context and "writer" in context:
        writer = context["writer"]
        print(f"[tool_executor_node] Got writer from context for thread {thread_id}!")
    
    # If still no writer, try to get from global registry using thread_id
    if writer is None and "thread_id" in new_state:
        writer = get_writer(new_state["thread_id"])
        if writer:
            print(f"[tool_executor_node] Got writer from global registry for thread {thread_id}")
        else:
            print(f"[tool_executor_node] WARNING: No writer found in registry for thread {thread_id}")
    
    # Check active writers for debugging
    print(f"[tool_executor_node] Active writers: {list(_active_writers.keys())}")
    print(f"[tool_executor_node] Current thread_id: {thread_id}")
    
    # Check if this is token-only stream mode
    is_token_only = new_state.get("_token_only_stream", False)
    
    # Merge tool calls from both specialized agents
    pending_muscle_calls = new_state.get("pending_muscle_tool_calls") or []
    pending_camera_calls = new_state.get("pending_camera_tool_calls") or []
    
    # Also check for generic pending_tool_calls
    generic_pending_calls = new_state.get("pending_tool_calls") or []
    
    # If we have specialized calls, use those
    # Otherwise fall back to generic pending_tool_calls
    if pending_muscle_calls or pending_camera_calls:
        # Combine all pending calls - always process muscle calls before camera calls
        pending = pending_muscle_calls + pending_camera_calls
    else:
        # Use generic pending_tool_calls if no specialized calls
        pending = generic_pending_calls
    
    if DEBUG_MODE:
        print(f"[tool_executor_node] Processing {len(pending)} pending tool calls "
              f"({len(pending_muscle_calls)} muscle, {len(pending_camera_calls)} camera, {len(generic_pending_calls)} generic)")
        print(f"[tool_executor_node] Current state: highlighted_muscles={new_state.get('highlighted_muscles')}")
        
        # Print actual tool calls for debugging
        for i, call in enumerate(pending):
            print(f"[tool_executor_node] Call {i+1}: {call.get('name')} - {json.dumps(call.get('args', {}))}")
    
    # If no tools to execute, just return the state unchanged
    if not pending:
        print("[tool_executor_node] No tools to execute")
        # Still route to responder
        new_state["_route"] = "responder"
        return new_state

    # Collect events
    events_collected: List[Dict[str, Any]] = new_state.get("events", []).copy()
    
    # Track event types to avoid duplicates
    tracked_events = set()
    
    # Track tool execution count for debugging/monitoring
    tool_execution_count = new_state.get("_tool_executions", 0)
    new_state["_tool_executions"] = tool_execution_count + len(pending)
    
    # IMPORTANT: Pre-process and merge select_muscles calls
    # This ensures we don't send multiple select_muscles events with different muscles
    # which would cause the frontend to override previous selections
    all_muscle_selections = []
    merged_muscle_names = []
    merged_colors = {}
    
    # First scan for all select_muscles calls and collect them
    for call in pending:
        if call.get("name") == "select_muscles":
            all_muscle_selections.append(call)
            muscle_names = call.get("args", {}).get("muscle_names", [])
            colors_arg = call.get("args", {}).get("colors", {})
            
            # Add these muscle names to our merged list
            merged_muscle_names.extend(muscle_names)
            
            # Handle different color formats and ensure every muscle gets a color
            if colors_arg is None:
                # No colors provided, assign based on muscle groups
                print(f"[tool_executor_node] No colors provided, assigning defaults to {len(muscle_names)} muscles")
                for name in muscle_names:
                    merged_colors[name] = get_color_for_muscle(name)
            elif isinstance(colors_arg, str):
                # String color applies to all muscles in this call
                print(f"[tool_executor_node] Using string color '{colors_arg}' for all {len(muscle_names)} muscles")
                for name in muscle_names:
                    merged_colors[name] = colors_arg
            elif isinstance(colors_arg, dict):
                # Dictionary maps specific muscles to colors
                print(f"[tool_executor_node] Using color dictionary with {len(colors_arg)} entries")
                for name in muscle_names:
                    if name in colors_arg and colors_arg[name]:
                        merged_colors[name] = colors_arg[name]
                    else:
                        # Muscle not in dict or has null color, assign based on group
                        merged_colors[name] = get_color_for_muscle(name)
            else:
                # Unexpected color format, use defaults
                print(f"[tool_executor_node] Unexpected color format: {type(colors_arg)}, using defaults")
                for name in muscle_names:
                    merged_colors[name] = get_color_for_muscle(name)
    
    # If we found multiple select_muscles calls, we'll replace them with a single merged call
    if len(all_muscle_selections) > 1:
        print(f"[tool_executor_node] Merging {len(all_muscle_selections)} select_muscles calls into one")
        print(f"[tool_executor_node] Combined muscle names: {merged_muscle_names}")
        print(f"[tool_executor_node] Combined colors: {merged_colors}")
        # Create a single merged event
        merged_event = {
            "type": "model:selectMuscles",
            "payload": {
                "muscleNames": merged_muscle_names,
                "colors": merged_colors
            }
        }
        
        # Add the merged event
        events_collected.append(merged_event)
        
        # Update the highlighted_muscles state
        new_state["highlighted_muscles"] = merged_colors
        
        # Remove the muscle selection calls from pending
        pending = [call for call in pending if call.get("name") != "select_muscles"]
        
        print(f"[tool_executor_node] Created merged select_muscles event with {len(merged_muscle_names)} muscles")
            
    # Process each remaining tool call (non-select_muscles ones)
    for i, call in enumerate(pending):
        try:
            if not isinstance(call, dict):
                print(f"[tool_executor_node] WARNING: Tool call {i} is not a dictionary: {type(call)}")
                continue
                
            tool_name: str = call.get("name", "")
            if not tool_name:
                print(f"[tool_executor_node] WARNING: Tool call {i} has no name: {call}")
                continue
                
            # Skip select_muscles only if we've already merged multiple such calls earlier
            # We detect this by checking if a merged select_muscles event was created above
            # and therefore the current pending list no longer contains the original calls.
            # If there is still a select_muscles call present in `pending`, it means we did
            # NOT merge (because there was only one), so we should process it normally.
            if tool_name == "select_muscles" and len(all_muscle_selections) > 1:
                # Already handled via the merged event – skip individual calls to avoid duplicates
                continue
                
            tool_args: Dict[str, Any] = call.get("args", {})
            tool_obj = TOOL_MAP.get(tool_name)
            if not tool_obj:
                print(f"[tool_executor_node] Unknown tool: {tool_name}")
                continue
            
            print(f"[tool_executor_node] Processing tool call: {tool_name} with args: {json.dumps(tool_args)}")
            
            # Execute tool based on its type
            tool_result: Dict[str, Any] = {}
            
            # MUSCLE TOOLS
            if tool_name == "select_muscles":
                muscle_names = tool_args.get("muscle_names", [])
                colors = tool_args.get("colors", {})
                
                print(f"[tool_executor_node] select_muscles with {len(muscle_names)} muscles and colors: {colors}")
                print(f"[tool_executor_node] Muscle names: {muscle_names[:5]}{'...' if len(muscle_names) > 5 else ''}")
                
                # Handle colors as either string or dictionary using same logic as preprocessing
                muscle_color_map = {}
                
                if colors is None:
                    print(f"[tool_executor_node] No colors provided for direct tool call, assigning defaults")
                    for name in muscle_names:
                        muscle_color_map[name] = get_color_for_muscle(name)
                elif isinstance(colors, str):
                    print(f"[tool_executor_node] Colors is a string: {colors}")
                    # Use the string color for all muscles
                    for name in muscle_names:
                        muscle_color_map[name] = colors
                elif isinstance(colors, dict):
                    print(f"[tool_executor_node] Colors is a dictionary with {len(colors)} mappings")
                    # Use the dictionary with fallbacks for missing entries
                    for name in muscle_names:
                        if name in colors and colors[name]:
                            muscle_color_map[name] = colors[name]
                        else:
                            muscle_color_map[name] = get_color_for_muscle(name)
                else:
                    print(f"[tool_executor_node] Colors has unexpected type: {type(colors)}")
                    # Default color based on muscle groups
                    for name in muscle_names:
                        muscle_color_map[name] = get_color_for_muscle(name)
                
                # Create event
                event = {
                    "type": "model:selectMuscles",
                    "payload": {"muscleNames": muscle_names, "colors": muscle_color_map}
                }
                
                print(f"[tool_executor_node] Created select_muscles event: {json.dumps(event)[:200]}")
                
                # Add to events
                events_collected.append(event)
                
                # Update state directly
                new_state["highlighted_muscles"] = muscle_color_map
                
                # Set result for event collection
                tool_result = {
                    "events": [event],
                }
                
                print(f"[tool_executor_node] Executed select_muscles with {len(muscle_names)} muscles")
                
            # CAMERA TOOLS
            elif tool_name == "set_camera_view":
                # Extract position coordinates
                position_x = float(tool_args.get("position_x", 0))
                position_y = float(tool_args.get("position_y", 0))
                position_z = float(tool_args.get("position_z", 0))
                
                # Extract target coordinates
                target_x = float(tool_args.get("target_x", 0))
                target_y = float(tool_args.get("target_y", 0))
                target_z = float(tool_args.get("target_z", 0))
                
                # Ensure position values are within reasonable ranges
                position_x = max(-7, min(7, position_x))
                position_y = max(-1, min(2, position_y))
                position_z = max(-7, min(7, position_z))
                
                # Ensure target values are within reasonable ranges
                target_x = max(-0.2, min(0.2, target_x))
                target_y = max(-0.6, min(1, target_y))
                target_z = max(-0.2, min(0.2, target_z))
                
                # Create position and target objects
                position = {"x": position_x, "y": position_y, "z": position_z}
                target = {"x": target_x, "y": target_y, "z": target_z}
                
                # Create a single combined event
                event = {
                    "type": "model:setCameraView",
                    "payload": {"position": position, "target": target}
                }
                
                # Add to events
                events_collected.append(event)
                
                # Update camera state
                camera = new_state.get("camera", {"position": {}, "target": {}}).copy()
                camera["position"] = position
                camera["target"] = target
                
                # Update state directly
                new_state["camera"] = camera
                
                # Set result for event collection
                tool_result = {
                    "events": [event],
                }
                
                print(f"[tool_executor_node] Executed set_camera_view with position:({position_x}, {position_y}, {position_z}), target:({target_x}, {target_y}, {target_z})")
                
            elif tool_name == "reset_camera":
                # Create event
                event = {
                    "type": "model:resetCamera",
                    "payload": {}
                }
                
                # Add to events
                events_collected.append(event)
                
                # Use the optimal Full Body Front view
                default_position = {"x": 0, "y": 1, "z": 6.5}
                default_target = {"x": 0, "y": 0, "z": 0}
                camera = {
                    "position": default_position,
                    "target": default_target
                }
                
                # Update state directly
                new_state["camera"] = camera
                
                # Set result for event collection
                tool_result = {
                    "events": [event],
                }
                
                print("[tool_executor_node] Executed reset_camera")
            
            else:
                # Unknown tool
                print(f"[tool_executor_node] Unknown tool: {tool_name}")
                continue

            # Check if the tool_result has events and stream them immediately
            if writer and "events" in tool_result:
                print(f"[tool_executor_node] Streaming {len(tool_result['events'])} events to frontend")
                for ev in tool_result["events"]:
                    # Create a stable hash for this event to check for duplicates
                    event_hash = f"{ev['type']}:{json.dumps(ev['payload'])}"
                    if event_hash not in tracked_events:
                        tracked_events.add(event_hash)
                        print(f"[tool_executor_node] Streaming event: {ev['type']} with payload: {json.dumps(ev['payload'])[:100]}")
                        # Ensure we use "event" as the type for the writer to correctly route in main.py
                        await writer({"type": "event", "content": ev})
                        print(f"[tool_executor_node] WRITER EVENT EMITTED WITH TYPE: event, EVENT TYPE: {ev['type']}")
                    else:
                        print(f"[tool_executor_node] Skipping duplicate event: {ev['type']}")
            
        except Exception as err:
            print(f"[tool_executor_node] Error executing {call}: {err}")
            import traceback
            traceback.print_exc()

    # Take only unique events by type+payload
    unique_events = []
    event_hashes = set()
    
    for ev in events_collected:
        # Create a hash of event type and payload
        ev_hash = f"{ev['type']}:{json.dumps(ev['payload'])}"
        if ev_hash not in event_hashes:
            event_hashes.add(ev_hash)
            unique_events.append(ev)
    
    # Update the state
    new_state["events"] = unique_events
    print(f"[tool_executor_node] Updated state: highlighted_muscles={new_state.get('highlighted_muscles', {})} camera={new_state.get('camera', {})}")
    print(f"[tool_executor_node] Collected {len(events_collected)} events, {len(unique_events)} unique")
    
    # Verify if any events were lost or filtered
    if len(events_collected) > len(unique_events):
        print(f"[tool_executor_node] WARNING: {len(events_collected) - len(unique_events)} events were filtered as duplicates")
    
    # IMPORTANT: Check our reliable flags BEFORE clearing pending calls
    just_executed_muscle_tools = new_state.get("_just_executed_muscle_tools", False)
    print(f"[tool_executor_node] Did we execute muscle tools? {just_executed_muscle_tools}")

    # Get pending_camera_tool_calls before clearing it
    pending_camera_tool_calls = new_state.get("pending_camera_tool_calls", [])

    # Clear pending calls so the planner can stop looping if none are added again
    new_state["pending_tool_calls"] = None
    new_state["pending_muscle_tool_calls"] = None
    new_state["pending_camera_tool_calls"] = None

    # MODIFIED: Verify and log highlighted_muscles state to ensure it carries through
    highlighted_muscles = new_state.get("highlighted_muscles", {})
    print(f"[tool_executor_node] After execution, highlighted_muscles has {len(highlighted_muscles)} muscles")
    if highlighted_muscles:
        print(f"[tool_executor_node] First few highlighted muscles: {list(highlighted_muscles.keys())[:3]}")

    # MODIFIED: With the new graph flow, we always go to responder after tool execution
    # Remove conditional routing and always set route to responder
    new_state["_route"] = "responder"
    new_state["_route_camera"] = False
    new_state["_just_executed_muscle_tools"] = False
    print(f"[tool_executor_node] Setting route to responder after tool execution")

    # Optionally provide a quick state-update chunk
    if writer:
        if DEBUG_MODE:
            print(f"[tool_executor_node] Writer available! Sending {len(unique_events)} events to frontend")
            print(f"[tool_executor_node] Final state has {len(unique_events)} events")
            for i, ev in enumerate(unique_events):
                print(f"[tool_executor_node] Event {i+1}: {ev['type']} - {json.dumps(ev['payload'])[:100]}")
        
        # Send a final update for all events
        await writer({"type": "updates", "content": {"events": unique_events}})
    else:
        print(f"[tool_executor_node] NO WRITER AVAILABLE! Cannot stream {len(unique_events)} events to frontend!")

    # IMPORTANT: Regardless of writer availability, always try to emit events via Socket.IO
    # This ensures events reach the frontend even if the streaming writer isn't available
    if thread_id:
        for event in unique_events:
            # Use the shared emit_event function
            await emit_event(event, thread_id)
        
        # Also send a summary
        if unique_events:
            summary_event = {"type": "events_summary", "count": len(unique_events)}
            await emit_event(summary_event, thread_id)
    
    # Return the fully updated state so LangGraph captures it.
    return new_state

# ---------------------------------------------------------------------------
# 3. Responder node
# ---------------------------------------------------------------------------

async def responder_node(
    state: ModelState,
    writer: Optional[StreamWriter] = None,
    context: Optional[Dict[str, Any]] = None,
):
    """Generate the final assistant message after all tools have run."""
    new_state: ModelState = copy.deepcopy(state)
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Check iteration count to prevent infinite loops
    iteration_count = new_state.get("_planner_iterations", 0)
    print(f"[responder_node] Iteration count: {iteration_count}, Events: {len(new_state.get('events', []))}")
    
    # Reset iteration counter for future messages
    new_state["_planner_iterations"] = 0
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    final_text = ""
    # Extract conversation history
    messages_history = new_state.get("messages", []).copy()
    
    # Get the last user question
    last_user_question = ""
    for msg in reversed(messages_history):
        if msg["role"] == "user":
            last_user_question = msg["content"]
            break

    initial_assessment = new_state.get("assistant_draft", "") or ""

    # Build a human-readable summary of model changes from events.
    # Only get unique events by type to avoid repetition
    unique_event_types = set()
    filtered_events = []
    for ev in new_state.get("events", []):
        ev_type = ev.get("type")
        if ev_type not in unique_event_types:
            unique_event_types.add(ev_type)
            filtered_events.append(ev)
    
    model_changes_lines = []
    for ev in filtered_events:
        ev_type = ev.get("type")
        if ev_type == "model:selectMuscles":
            muscles = ev["payload"].get("muscleNames", [])
            if muscles: 
                muscle_str = ", ".join([m.split("_R")[0] for m in muscles])
                model_changes_lines.append(f"Highlighted {muscle_str}")
        elif ev_type == "model:setCameraView":
            model_changes_lines.append("Adjusted camera position for better view")
        elif ev_type == "model:resetCamera":
            model_changes_lines.append("Reset camera view")
    
    # If we have muscle information in state, make sure to mention it even if no events
    current_muscles = new_state.get("highlighted_muscles", {})
    if current_muscles and not any("Highlighted" in line for line in model_changes_lines):
        muscles_str = ", ".join(m.split("_R")[0] for m in current_muscles.keys())
        model_changes_lines.append(f"Currently showing {muscles_str}")
            
    model_changes = "; ".join(model_changes_lines) if model_changes_lines else "No model changes."

    # Convert conversation history to LangChain message format
    conversation_messages = []
    for msg in messages_history[:-1]:  # Exclude the last user message as we'll add it separately
        role = msg.get("role")
        if role == "user":
            conversation_messages.append(HumanMessage(content=msg["content"]))
        elif role == "assistant":
            conversation_messages.append(AIMessage(content=msg["content"]))
    # Append the latest user message as a HumanMessage
    if messages_history and messages_history[-1]["role"] == "user":
        conversation_messages.append(HumanMessage(content=messages_history[-1]["content"]))

    # Create full prompt chain with conversation history
    prompt_chain = [
        SystemMessage(content=RESPONSE_PROMPT),
        SystemMessage(content=f"Model changes:\n{model_changes}"),
        SystemMessage(content=f"Current model state:\n{_format_state_for_prompt(new_state)}"),
        SystemMessage(content="[CONVERSATION HISTORY]"),
        *conversation_messages,
    ]
    
    
    try:
        # Always use streaming
        async for chunk in _llm_streaming.astream(prompt_chain):
            token = chunk.content if hasattr(chunk, "content") else chunk
            if token:
                # print(f"[responder_node] Directly streaming token: {token}")
                new_state["generation"] = new_state.get("generation", "") + token
                final_text += token
                
                # Yield for streaming
                yield {"type": "response", "content": token}
        
        print(f"[responder_node] Finished streaming. Total response length: {len(final_text)}")
    except Exception as e:
        print(f"[responder_node] Error during streaming: {e}")
        # Fall back to non-streaming
        resp = await _llm_streaming.ainvoke(prompt_chain)
        final_text = resp.content
        print(f"[responder_node] Used fallback non-streaming. Response length: {len(final_text)}")
        yield {"type": "response", "content": final_text}

    # Update state
    messages = new_state.get("messages", []).copy()
    messages.append({
        "role": "assistant", 
        "content": final_text,
        "timestamp": datetime.now().isoformat()
    })
    new_state["messages"] = messages

    # Clean up internal fields
    new_state["assistant_draft"] = None
    new_state["events"] = []
    new_state["_route_camera"] = False
    new_state["_route_muscle"] = False
    new_state["_route"] = None
    new_state["_planner_iterations"] = 0
    new_state["_just_executed_muscle_tools"] = False
    new_state["_tool_executions"] = 0

    #new_state["final_state"] = True
    print(f"[responder_node] Returning final state with {len(new_state['messages'])} messages. Last message: {new_state['messages'][-1]}")

    # Yield the fully updated state so LangGraph can capture it.
    yield new_state

async def muscle_control_agent(
    state: ModelState,
    writer: Optional[StreamWriter] = None,
    context: Optional[Dict[str, Any]] = None,
) -> ModelState:
    """Specialized agent for controlling muscle highlighting.
    
    This agent focuses exclusively on deciding which muscles to highlight
    based on the user's request and the current state of the model.
    
    Args:
        state: The current state dictionary
        writer: Optional streaming writer for token-by-token output
        context: Optional context with additional parameters
        
    Returns:
        state with pending_muscle_tool_calls containing the muscle-related tool calls
    """
    # Ensure we do not mutate the incoming dict directly
    new_state: ModelState = copy.deepcopy(state)
    
    # Try to get writer from context if not directly provided
    if writer is None and context and "writer" in context:
        writer = context["writer"]
        print(f"[muscle_control_agent] Got writer from context!")
    
    # Get current state of muscles
    highlighted_muscles = new_state.get("highlighted_muscles", {})
    highlighted_muscles_str = (
        ", ".join(f"{m} (color: {c})" for m, c in highlighted_muscles.items())
        if highlighted_muscles
        else "None"
    )
    
    print(f"[muscle_control_agent] Current highlighted muscles: {highlighted_muscles_str}")
    
    # Extract conversation history and last user message
    messages_history = new_state.get("messages", []).copy()
    if not messages_history or messages_history[-1]["role"] != "user":
        # Nothing to do; just return state unchanged
        return new_state
    
    user_msg = messages_history[-1]["content"]
    print(f"[muscle_control_agent] Processing user message: {user_msg}")
    
    # Build conversation history for context
    conversation_messages = []
    for msg in messages_history[:-1]:
        role = msg.get("role")
        if role == "user":
            conversation_messages.append(HumanMessage(content=msg["content"]))
        elif role == "assistant":
            conversation_messages.append(AIMessage(content=msg["content"]))
    # Append the latest user message as a HumanMessage
    if messages_history and messages_history[-1]["role"] == "user":
        conversation_messages.append(HumanMessage(content=messages_history[-1]["content"]))

    # Compose the prompt chain: system prompt, conversation history, then user message and state
    prompt_chain = [
        SystemMessage(content=MUSCLE_AGENT_PROMPT.format(
            MUSCLE_MAPPING_STR=MUSCLE_MAPPING_STR,
            MUSCLE_PAIRING_RULES=MUSCLE_PAIRING_RULES,
            MUSCLE_NAMING_RULES=MUSCLE_NAMING_RULES,
            FUNCTIONAL_GROUPS_STR=FUNCTIONAL_GROUPS_STR
        )),
        SystemMessage(content=f"[CURRENT 3D MODEL STATE]\nHighlighted muscles: {highlighted_muscles_str}"),
        SystemMessage(content="[CONVERSATION HISTORY]"),
        *conversation_messages,
    ]

    if writer:
        await writer({"type": "thinking", "content": "Planning muscle highlighting..."})

    # Call the LLM with muscle control tools - force tool usage
    llm_response = await _llm_streaming.bind_tools(
        MUSCLE_CONTROL_TOOLS,
        tool_choice="required",  # Force tool usage
    ).ainvoke(prompt_chain)
    
    # Check if tool calls were proposed
    pending_muscle_tool_calls = []
    if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
        pending_muscle_tool_calls = llm_response.tool_calls
        print(f"[muscle_control_agent] Generated {len(pending_muscle_tool_calls)} muscle tool calls")
        for i, call in enumerate(pending_muscle_tool_calls):
            print(f"[muscle_control_agent] Tool {i+1}: {call.get('name')} - {call.get('args')}")
            
            # MODIFIED: Update highlighted_muscles state directly for the camera_control_agent
            if call.get("name") == "select_muscles":
                # Extract muscle names and colors from the tool call
                muscle_names = call.get("args", {}).get("muscle_names", [])
                colors_arg = call.get("args", {}).get("colors", {})
                
                # Process colors in the same way as tool_executor_node
                muscle_color_map = {}
                
                if not colors_arg:
                    # No colors provided, assign based on muscle groups
                    print(f"[muscle_control_agent] No colors provided, assigning defaults to {len(muscle_names)} muscles")
                    for name in muscle_names:
                        muscle_color_map[name] = get_color_for_muscle(name)
                elif isinstance(colors_arg, str):
                    # String color applies to all muscles in this call
                    print(f"[muscle_control_agent] Using string color '{colors_arg}' for all {len(muscle_names)} muscles")
                    for name in muscle_names:
                        muscle_color_map[name] = colors_arg
                elif isinstance(colors_arg, dict):
                    # Dictionary maps specific muscles to colors
                    print(f"[muscle_control_agent] Using color dictionary with {len(colors_arg)} entries")
                    for name in muscle_names:
                        if name in colors_arg and colors_arg[name]:
                            muscle_color_map[name] = colors_arg[name]
                        else:
                            # Muscle not in dict or has null color, assign based on group
                            muscle_color_map[name] = get_color_for_muscle(name)
                
                # Update state directly so camera_control_agent can use it
                new_state["highlighted_muscles"] = muscle_color_map
                print(f"[muscle_control_agent] Updated highlighted_muscles state with {len(muscle_color_map)} muscles")
            
            elif call.get("name") == "toggle_muscle":
                # Handle toggle_muscle tool calls
                muscle_name = call.get("args", {}).get("muscle_name", "")
                color = call.get("args", {}).get("color") or "#FFD700"  # Default gold color
                
                # Update state directly
                current_muscles = new_state.get("highlighted_muscles", {}).copy()
                if muscle_name in current_muscles:
                    current_muscles.pop(muscle_name)
                else:
                    current_muscles[muscle_name] = color
                
                new_state["highlighted_muscles"] = current_muscles
                print(f"[muscle_control_agent] Updated highlighted_muscles state after toggle")
    else:
        print(f"[muscle_control_agent] WARNING: No tool calls generated despite setting tool_choice='required'")
        print(f"[muscle_control_agent] LLM response content: {getattr(llm_response, 'content', '')}")
        
        # Fallback to another LLM call with an even more explicit prompt
        fallback_prompt = """
You are a fitness expert controlling a 3D muscle model. 
Based on this user question: "{user_msg}"

You need to specify EXACTLY which muscles should be highlighted on the model.
Return your answer in this JSON format:
{{
  "muscle_names": ["Muscle1", "Muscle2", ...],
  "colors": {{"Muscle1": "#hexcolor1", "Muscle2": "#hexcolor2", ...}}
}}

Include both left and right versions of muscles (with _R suffix for right side).
Be comprehensive - include all relevant muscles the user would want to see.

IMPORTANT: You MUST provide a color hex code for EACH muscle in the muscle_names list.
Use consistent coloring based on muscle groups:
- Chest/abs: "#FF5555" (red) 
- Back: "#5555FF" (blue)
- Arms: "#55FF55" (green)
- Legs: "#FFFF55" (yellow)
- Shoulders: "#FF55FF" (purple)
"""
        
        try:
            # Create the fallback prompt instance
            fallback_system = SystemMessage(content=fallback_prompt.format(user_msg=user_msg))
            fallback_user = HumanMessage(content="Select the muscles that should be highlighted for this request.")
            
            # Call a structured LLM for the fallback
            fallback_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
            fallback_response = await fallback_llm.ainvoke([fallback_system, fallback_user])
            fallback_text = fallback_response.content
            
            print(f"[muscle_control_agent] Fallback LLM response: {fallback_text}")
            
            # Try to parse JSON from the response
            if '{' in fallback_text and '}' in fallback_text:
                json_start = fallback_text.find('{')
                json_end = fallback_text.rfind('}') + 1
                json_content = fallback_text[json_start:json_end]
                
                fallback_data = json.loads(json_content)
                
                if isinstance(fallback_data, dict) and "muscle_names" in fallback_data:
                    muscle_names = fallback_data.get("muscle_names", [])
                    colors = fallback_data.get("colors", {})
                    
                    # Create default colors if missing
                    if not colors and muscle_names:
                        colors = {}
                        for muscle in muscle_names:
                            colors[muscle] = "#FFD700"  # Gold default color
                    
                    # Create a tool call
                    pending_muscle_tool_calls = [{
                        "name": "select_muscles",
                        "args": {
                            "muscle_names": muscle_names,
                            "colors": colors
                        }
                    }]
                    
                    print(f"[muscle_control_agent] Created fallback muscle selection with {len(muscle_names)} muscles")
                    
                    # MODIFIED: Also directly update state for camera agent
                    if colors and muscle_names:
                        new_state["highlighted_muscles"] = colors
                        print(f"[muscle_control_agent] Updated highlighted_muscles with fallback data: {len(colors)} muscles")
                    
                else:
                    print(f"[muscle_control_agent] Failed to extract muscle names from fallback JSON")
            else:
                print(f"[muscle_control_agent] Fallback did not return valid JSON")
                
        except Exception as e:
            print(f"[muscle_control_agent] Error in fallback muscle selection: {e}")
            # No further fallbacks - will need to be handled by responder
    
    # Mark muscle control as completed in the control history
    control_history = new_state.get("_control_history", {})
    current_request_id = f"req_{len(messages_history)}"
    control_history[f"{current_request_id}_muscle_done"] = True
    new_state["_control_history"] = control_history
    
    # Add text explanation to assistant draft
    assistant_draft = new_state.get("assistant_draft", "")
    assistant_draft += "\n\nI'm highlighting the relevant muscles for your request."
    new_state["assistant_draft"] = assistant_draft
    
    # Store the tool calls in state
    new_state["pending_muscle_tool_calls"] = pending_muscle_tool_calls
    
    # IMPORTANT: Also set generic pending_tool_calls to ensure they're picked up by the tool executor
    new_state["pending_tool_calls"] = pending_muscle_tool_calls
    
    # CRITICAL: Set a clear flag indicating we just generated muscle tools
    # This will be more reliable than checking for pending_muscle_tool_calls
    new_state["_just_executed_muscle_tools"] = True
    print(f"[muscle_control_agent] Setting _just_executed_muscle_tools = True")
    
    # MODIFIED: Since we're now going directly to camera_control, we should NOT set _route
    # Remove the _route field to ensure proper flow
    if "_route" in new_state:
        del new_state["_route"]
    
    return new_state

async def camera_control_agent(
    state: ModelState,
    writer: Optional[StreamWriter] = None,
    context: Optional[Dict[str, Any]] = None,
) -> ModelState:
    """Specialized agent for controlling camera positioning.
    
    This agent focuses exclusively on positioning the camera
    to provide the best view of the currently highlighted muscles.
    
    Args:
        state: The current state dictionary
        writer: Optional streaming writer for token-by-token output
        context: Optional context with additional parameters
        
    Returns:
        state with pending_camera_tool_calls containing the camera-related tool calls
    """
    # Ensure we do not mutate the incoming dict directly
    new_state: ModelState = copy.deepcopy(state)
    
    # Try to get writer from context if not directly provided
    if writer is None and context and "writer" in context:
        writer = context["writer"]
        print(f"[camera_control_agent] Got writer from context!")
    
    # Get current state of muscles and camera
    highlighted_muscles = new_state.get("highlighted_muscles", {})
    highlighted_muscles_str = (
        ", ".join(f"{m}" for m in highlighted_muscles.keys())
        if highlighted_muscles
        else "None"
    )
    
    # MODIFIED: Add additional detailed logging for debugging the muscle state passing
    print(f"[camera_control_agent] Received highlighted_muscles with {len(highlighted_muscles)} muscles")
    if highlighted_muscles:
        print(f"[camera_control_agent] First few muscles: {list(highlighted_muscles.keys())[:3]}")
    
    camera = new_state.get("camera", {"position": {}, "target": {}})
    camera_str = f"Position: {camera.get('position', {})}, Target: {camera.get('target', {})}"
    
    print(f"[camera_control_agent] Highlighted muscles: {highlighted_muscles_str}")
    print(f"[camera_control_agent] Current camera: {camera_str}")
    
    # Extract conversation history and last user message
    messages_history = new_state.get("messages", []).copy()
    if not messages_history or messages_history[-1]["role"] != "user":
        # Nothing to do; just return state unchanged
        return new_state
    
    user_msg = messages_history[-1]["content"]
    print(f"[camera_control_agent] Processing user message: {user_msg}")
    
    # Enhanced camera prompt with more AI-driven decision making
    enhanced_camera_prompt = f"""
You are a specialized camera control agent for a 3D anatomy model. Your ONLY job is to position the camera for optimal viewing.

[Currently Highlighted Muscles]
{highlighted_muscles_str}

[Current Camera Position]
{camera_str}

[Camera Control Guidelines]
For the clearest demonstrations, use these specific presets based on the highlighted muscles:

Upper Body Front View (for chest, biceps, abs):
- Position: x: -0.03, y: 0.83, z: 3.48
- Target: x: -0.03, y: 0.83, z: 0.0

Upper Body Back View (for back, shoulders, lats):
- Position: x: 0.20, y: 1.53, z: -3.70
- Target: x: 0.07, y: 0.77, z: 0.16

Lower Body Front View (for quads, calves):
- Position: x: -0.0007, y: -0.50, z: 4.45
- Target: x: 0.0006, y: -0.50, z: 0.0

Lower Body Back View (for glutes, hamstrings):
- Position: x: 0.20, y: 0.26, z: -4.21
- Target: x: 0.06, y: -0.56, z: -0.11

[CRITICAL INSTRUCTIONS]
You MUST use at least one camera control tool in your response.
Analyze the highlighted muscles and determine the optimal camera position.
- For back muscles (Latissimus, Trapezius, Rhomboideus): use Upper Body Back View
- For chest muscles (Pectoralis): use Upper Body Front View
- For hamstrings or glutes: use Lower Body Back View
- For quadriceps or calves: use Lower Body Front View
- If no muscles you don't need to move the camera

DO NOT return a response without using a tool.
"""
    
    # System message with specialized camera agent prompt
    system = SystemMessage(content=enhanced_camera_prompt)
    
    model_snapshot = SystemMessage(content=f"""[Current Model State]
Highlighted muscles: {highlighted_muscles_str}
Camera: {camera_str}
""")
    
    # Build conversation history
    conversation_messages = []
    for msg in messages_history[:-1]:
        role = msg.get("role")
        if role == "user":
            conversation_messages.append(HumanMessage(content=msg["content"]))
        elif role == "assistant":
            conversation_messages.append(AIMessage(content=msg["content"]))
    
    # Append the latest user message as a HumanMessage
    if messages_history and messages_history[-1]["role"] == "user":
        conversation_messages.append(HumanMessage(content=messages_history[-1]["content"]))

    # Create full prompt
    prompt_chain = [system, model_snapshot] + conversation_messages
    
    if writer:
        await writer({"type": "thinking", "content": "Planning camera positioning based on highlighted muscles..."})
    
    # Call the LLM with camera control tools - force tool usage
    llm_response = await _llm_streaming.bind_tools(
        CAMERA_CONTROL_TOOLS,
        tool_choice="required",  # Force tool usage
    ).ainvoke(prompt_chain)
    
    # Check if tool calls were proposed
    pending_camera_tool_calls = []
    if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
        pending_camera_tool_calls = llm_response.tool_calls
        print(f"[camera_control_agent] Generated {len(pending_camera_tool_calls)} camera control tools")
        for i, call in enumerate(pending_camera_tool_calls):
            print(f"[camera_control_agent] Tool {i+1}: {call.get('name')} - {call.get('args')}")
    else:
        # [Keep your fallback logic for when no tool calls are generated]
        print(f"[camera_control_agent] WARNING: No tool calls generated despite setting tool_choice='required'")
        # [Keep your fallback tool creation code...]
    
    # Mark camera control as completed in the control history
    control_history = new_state.get("_control_history", {})
    current_request_id = f"req_{len(messages_history)}"
    control_history[f"{current_request_id}_camera_done"] = True
    new_state["_control_history"] = control_history
    
    # Add text explanation to assistant draft
    assistant_draft = new_state.get("assistant_draft", "")
    assistant_draft += "\n\nI'm adjusting the camera to give you the best view of the highlighted muscles."
    new_state["assistant_draft"] = assistant_draft
        
    # Store the tool calls in state
    new_state["pending_camera_tool_calls"] = pending_camera_tool_calls
    
    # MODIFIED: We now always go to execute_tools, so we need to ensure it has the correct tools
    # If we have both muscle and camera tools, merge them for execute_tools
    pending_muscle_tool_calls = new_state.get("pending_muscle_tool_calls", [])
    if pending_muscle_tool_calls:
        # Combine tool calls from both agents - always process muscle tools first
        combined_tools = pending_muscle_tool_calls + pending_camera_tool_calls
        new_state["pending_tool_calls"] = combined_tools
        print(f"[camera_control_agent] Combined {len(pending_muscle_tool_calls)} muscle tools with {len(pending_camera_tool_calls)} camera tools")
    else:
        # Just use camera tools
        new_state["pending_tool_calls"] = pending_camera_tool_calls
        print(f"[camera_control_agent] Using only {len(pending_camera_tool_calls)} camera tools")
    
    # MODIFIED: Clear the routing flags to ensure normal flow (just in case)
    if "_route" in new_state:
        del new_state["_route"]
    
    return new_state