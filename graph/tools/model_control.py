from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
import json

# These tools will be used to control the 3D model through socket events
# Each tool corresponds to a specific socket event from the frontend

class SelectMusclesInput(BaseModel):
    """Select multiple muscles on the 3D model, with optional colors."""
    muscle_names: List[str] = Field(
        ..., 
        description="List of muscle names to select. Use proper muscle naming (as provided in the instructions) like 'Biceps_Brachii'."
    )
    colors: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional mapping of muscle names to highlight colors (hex)."
    )

@tool("select_muscles", args_schema=SelectMusclesInput, return_direct=False)
def select_muscles_tool(state: Annotated[AgentState, InjectedState], muscle_names: List[str], colors: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Execute the muscle selection and update the state, with optional colors."""
    print(f"select_muscles_tool called with muscles: {muscle_names}, colors: {colors}")
    # Default color if not specified
    default_color = "#FFD600"
    muscle_color_map = {name: (colors[name] if colors and name in colors else default_color) for name in muscle_names}
    event = {
        "type": "model:selectMuscles",
        "payload": {"muscleNames": muscle_names, "colors": muscle_color_map}
    }
    events = state.get("events", []).copy()
    events.append(event)
    
    # Deduplicate events
    unique_events = _dedup_events(events)
    
    print(f"Added event: {event}")
    result = {
        "events": unique_events,
        "highlighted_muscles": muscle_color_map
    }
    print(f"Returning result with {len(unique_events)} events")
    return result

class ToggleMuscleInput(BaseModel):
    """Toggle selection of a single muscle on the 3D model, with optional color."""
    muscle_name: str = Field(
        ..., 
        description="Name of muscle to toggle. Use proper muscle naming like 'Biceps_Brachii'."
    )
    color: Optional[str] = Field(
        default=None,
        description="Highlight color (hex) for the muscle."
    )

@tool("toggle_muscle", args_schema=ToggleMuscleInput, return_direct=False)
def toggle_muscle_tool(state: Annotated[AgentState, InjectedState], muscle_name: str, color: Optional[str] = None) -> Dict[str, Any]:
    """Execute the muscle toggle and update the state, with optional color."""
    print(f"toggle_muscle_tool called with muscle: {muscle_name}, color: {color}")
    event = {
        "type": "model:toggleMuscle",
        "payload": {"muscleName": muscle_name, "color": color or '#FFD600'}
    }
    events = state.get("events", []).copy()
    events.append(event)
    # Update highlighted muscles
    highlighted_muscles = state.get("highlighted_muscles", {}).copy()
    if muscle_name in highlighted_muscles:
        highlighted_muscles.pop(muscle_name)
    else:
        highlighted_muscles[muscle_name] = color or '#FFD600'
    # Deduplicate events
    unique_events = _dedup_events(events)
    print(f"Added event: {event}")
    result = {
        "events": unique_events,
        "highlighted_muscles": highlighted_muscles
    }
    print(f"Returning result with {len(unique_events)} events")
    return result

class SetAnimationFrameInput(BaseModel):
    """Set the animation frame for the 3D model."""
    frame: int = Field(
        ..., 
        description="Animation frame number (0-50).",
        ge=0, 
        le=50
    )

@tool("set_animation_frame", args_schema=SetAnimationFrameInput, return_direct=False)
def set_animation_frame_tool(frame: int, state: Annotated[AgentState, InjectedState]) -> Dict[str, Any]:
    """Execute the animation frame change and update the state."""
    print(f"set_animation_frame_tool called with frame: {frame}")
    event = {
        "type": "model:setAnimationFrame",
        "payload": {"frame": frame}
    }
    events = state.get("events", []).copy()
    events.append(event)
    
    # Update animation state
    animation = state.get("animation", {}).copy()
    animation["frame"] = frame
    
    # Deduplicate events
    unique_events = _dedup_events(events)
    
    print(f"Added event: {event}")
    result = {
        "events": unique_events,
        "animation": animation
    }
    print(f"Returning result with {len(unique_events)} events")
    return result

class ToggleAnimationInput(BaseModel):
    """Start or stop the animation of the 3D model."""
    is_playing: bool = Field(
        ..., 
        description="Whether the animation should be playing (true) or paused (false)."
    )

@tool("toggle_animation", args_schema=ToggleAnimationInput, return_direct=False)
def toggle_animation_tool(is_playing: bool, state: Annotated[AgentState, InjectedState]) -> Dict[str, Any]:
    """Execute the animation toggle and update the state."""
    print(f"toggle_animation_tool called with is_playing: {is_playing}")
    event = {
        "type": "model:toggleAnimation",
        "payload": {"isPlaying": is_playing}
    }
    events = state.get("events", []).copy()
    events.append(event)
    
    # Update animation state
    animation = state.get("animation", {}).copy()
    animation["isPlaying"] = is_playing
    
    # Deduplicate events
    unique_events = _dedup_events(events)
    
    print(f"Added event: {event}")
    result = {
        "events": unique_events,
        "animation": animation
    }
    print(f"Returning result with {len(unique_events)} events")
    return result

class SetCameraPositionInput(BaseModel):
    """Set the camera position for the 3D model view."""
    x: float = Field(..., description="X coordinate of camera position.")
    y: float = Field(..., description="Y coordinate of camera position.")
    z: float = Field(..., description="Z coordinate of camera position.")

@tool(
    "set_camera_position",
    args_schema=SetCameraPositionInput,
    return_direct=False,
    description="""Set the camera position for the 3D model view.
    
    Common view positions:
    - Upper Body Front View (chest, biceps, abs): x: -0.03, y: 0.83, z: 3.48
    - Upper Body Back View (back, shoulders): x: 0.20, y: 1.53, z: -3.70
    - Lower Body Front View (quads, calves): x: -0.0007, y: -0.50, z: 4.45
    - Lower Body Back View (glutes, hamstrings): x: 0.20, y: 0.26, z: -4.21
    """
)
def set_camera_position_tool(x: float, y: float, z: float, state: Annotated[AgentState, InjectedState]) -> Dict[str, Any]:
    """Execute the camera position change and update the state."""
    print(f"set_camera_position_tool called with position: ({x}, {y}, {z})")
    position = {"x": x, "y": y, "z": z}
    event = {
        "type": "model:setCameraPosition",
        "payload": {"position": position}
    }
    events = state.get("events", []).copy()
    events.append(event)
    
    # Update camera state
    camera = state.get("camera", {"position": {}, "target": {}}).copy()
    camera["position"] = position
    
    # Deduplicate events
    unique_events = _dedup_events(events)
    
    print(f"Added event: {event}")
    result = {
        "events": unique_events,
        "camera": camera
    }
    print(f"Returning result with {len(unique_events)} events")
    return result

class SetCameraTargetInput(BaseModel):
    """Set the camera target (look-at point) for the 3D model view."""
    x: float = Field(..., description="X coordinate of camera target.")
    y: float = Field(..., description="Y coordinate of camera target.")
    z: float = Field(..., description="Z coordinate of camera target.")

@tool(
    "set_camera_target",
    args_schema=SetCameraTargetInput,
    return_direct=False,
    description="""Set the camera target (look-at point) for the 3D model view."""
)
def set_camera_target_tool(x: float, y: float, z: float, state: Annotated[AgentState, InjectedState]) -> Dict[str, Any]:
    """Execute the camera target change and update the state."""
    print(f"set_camera_target_tool called with target: ({x}, {y}, {z})")
    target = {"x": x, "y": y, "z": z}
    event = {
        "type": "model:setCameraTarget",
        "payload": {"target": target}
    }
    events = state.get("events", []).copy()
    events.append(event)
    
    # Update camera state
    camera = state.get("camera", {"position": {}, "target": {}}).copy()
    camera["target"] = target
    
    # Deduplicate events
    unique_events = _dedup_events(events)
    
    print(f"Added event: {event}")
    result = {
        "events": unique_events,
        "camera": camera
    }
    print(f"Returning result with {len(unique_events)} events")
    return result

class SetCameraViewInput(BaseModel):
    """Set both camera position and target in a single call."""
    position_x: float = Field(..., description="X coordinate of camera position.")
    position_y: float = Field(..., description="Y coordinate of camera position.")
    position_z: float = Field(..., description="Z coordinate of camera position.")
    target_x: float = Field(..., description="X coordinate of camera target (look-at point).")
    target_y: float = Field(..., description="Y coordinate of camera target (look-at point).")
    target_z: float = Field(..., description="Z coordinate of camera target (look-at point).")

@tool(
    "set_camera_view",
    args_schema=SetCameraViewInput,
    return_direct=False,
    description="""Set both camera position and target in a single call.
    
    Common view presets:
    - Upper Body Front View (chest, biceps, abs): 
      position: x: -0.03, y: 0.83, z: 3.48, target: x: -0.03, y: 0.83, z: 0.0
    - Upper Body Back View (back, shoulders): 
      position: x: 0.20, y: 1.53, z: -3.70, target: x: 0.07, y: 0.77, z: 0.16
    - Lower Body Front View (quads, calves): 
      position: x: -0.0007, y: -0.50, z: 4.45, target: x: 0.0006, y: -0.50, z: 0.0
    - Lower Body Back View (glutes, hamstrings): 
      position: x: 0.20, y: 0.26, z: -4.21, target: x: 0.06, y: -0.56, z: -0.11
    """
)
def set_camera_view_tool(
    position_x: float, position_y: float, position_z: float,
    target_x: float, target_y: float, target_z: float,
    state: Annotated[AgentState, InjectedState]
) -> Dict[str, Any]:
    """Execute camera position and target change in a single operation."""
    print(f"set_camera_view_tool called with position: ({position_x}, {position_y}, {position_z}), target: ({target_x}, {target_y}, {target_z})")
    
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
    events = state.get("events", []).copy()
    events.append(event)
    
    # Update camera state
    camera = state.get("camera", {"position": {}, "target": {}}).copy()
    camera["position"] = position
    camera["target"] = target
    
    # Deduplicate events
    unique_events = _dedup_events(events)
    
    print(f"Added event: {event}")
    result = {
        "events": unique_events,
        "camera": camera
    }
    print(f"Returning result with {len(unique_events)} events")
    return result

@tool("reset_camera", return_direct=False)
def reset_camera_tool(state: Annotated[AgentState, InjectedState]) -> Dict[str, Any]:
    """Reset the camera to the default position and target."""
    print("reset_camera_tool called")
    event = {
        "type": "model:resetCamera",
        "payload": {}
    }
    events = state.get("events", []).copy()
    events.append(event)
    
    # Set default values
    default_position = {"x": 0, "y": 1, "z": 7}
    default_target = {"x": 0, "y": 0, "z": 0}
    camera = {
        "position": default_position,
        "target": default_target
    }
    
    # Deduplicate events
    unique_events = _dedup_events(events)
    
    print(f"Added event: {event}")
    result = {
        "events": unique_events,
        "camera": camera
    }
    print(f"Returning result with {len(unique_events)} events")
    return result

def _dedup_events(events):
    """Helper to deduplicate events by type and payload.
    
    This prevents the same event from being sent multiple times.
    """
    if not events:
        return []
        
    unique_events = []
    event_hashes = set()
    
    for event in events:
        # Create a hash of event type and payload
        event_hash = f"{event['type']}:{json.dumps(event['payload'])}"
        if event_hash not in event_hashes:
            event_hashes.add(event_hash)
            unique_events.append(event)
    
    return unique_events

# Export the tools as a list of callables
MODEL_CONTROL_TOOL_FUNCTIONS = [
    select_muscles_tool,
    toggle_muscle_tool,
    set_animation_frame_tool,
    toggle_animation_tool,
    set_camera_view_tool,
    reset_camera_tool,
] 