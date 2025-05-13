from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState

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
    events = state.get("events", [])
    events.append(event)
    print(f"Added event: {event}")
    result = {
        "events": events,
        "highlighted_muscles": muscle_color_map
    }
    print(f"Returning result with events: {events}")
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
    events = state.get("events", [])
    events.append(event)
    highlighted_muscles = state.get("highlighted_muscles", {}).copy()
    if muscle_name in highlighted_muscles:
        highlighted_muscles.pop(muscle_name)
    else:
        highlighted_muscles[muscle_name] = color or '#FFD600'
    print(f"Added event: {event}")
    result = {
        "events": events,
        "highlighted_muscles": highlighted_muscles
    }
    print(f"Returning result with events: {events}")
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
    events = state.get("events", [])
    events.append(event)
    animation = state.get("animation", {}).copy()
    animation["frame"] = frame
    print(f"Added event: {event}")
    result = {
        "events": events,
        "animation": animation
    }
    print(f"Returning result with events: {events}")
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
    events = state.get("events", [])
    events.append(event)
    animation = state.get("animation", {}).copy()
    animation["isPlaying"] = is_playing
    print(f"Added event: {event}")
    result = {
        "events": events,
        "animation": animation
    }
    print(f"Returning result with events: {events}")
    return result

class SetCameraPositionInput(BaseModel):
    """Set the camera position for the 3D model view."""
    x: float = Field(..., description="X coordinate of camera position.")
    y: float = Field(..., description="Y coordinate of camera position.")
    z: float = Field(..., description="Z coordinate of camera position.")

@tool("set_camera_position", args_schema=SetCameraPositionInput, return_direct=False)
def set_camera_position_tool(x: float, y: float, z: float, state: Annotated[AgentState, InjectedState]) -> Dict[str, Any]:
    """Execute the camera position change and update the state."""
    print(f"set_camera_position_tool called with position: ({x}, {y}, {z})")
    position = {"x": x, "y": y, "z": z}
    event = {
        "type": "model:setCameraPosition",
        "payload": {"position": position}
    }
    events = state.get("events", [])
    events.append(event)
    camera = state.get("camera", {"position": {}, "target": {}}).copy()
    camera["position"] = position
    print(f"Added event: {event}")
    result = {
        "events": events,
        "camera": camera
    }
    print(f"Returning result with events: {events}")
    return result

class SetCameraTargetInput(BaseModel):
    """Set the camera target (look-at point) for the 3D model view."""
    x: float = Field(..., description="X coordinate of camera target.")
    y: float = Field(..., description="Y coordinate of camera target.")
    z: float = Field(..., description="Z coordinate of camera target.")

@tool("set_camera_target", args_schema=SetCameraTargetInput, return_direct=False)
def set_camera_target_tool(x: float, y: float, z: float, state: Annotated[AgentState, InjectedState]) -> Dict[str, Any]:
    """Execute the camera target change and update the state."""
    print(f"set_camera_target_tool called with target: ({x}, {y}, {z})")
    target = {"x": x, "y": y, "z": z}
    event = {
        "type": "model:setCameraTarget",
        "payload": {"target": target}
    }
    events = state.get("events", [])
    events.append(event)
    camera = state.get("camera", {"position": {}, "target": {}}).copy()
    camera["target"] = target
    print(f"Added event: {event}")
    result = {
        "events": events,
        "camera": camera
    }
    print(f"Returning result with events: {events}")
    return result

@tool("reset_camera", return_direct=False)
def reset_camera_tool(state: Annotated[AgentState, InjectedState]) -> Dict[str, Any]:
    """Reset the camera to the default position and target."""
    print("reset_camera_tool called")
    event = {
        "type": "model:resetCamera",
        "payload": {}
    }
    events = state.get("events", [])
    events.append(event)
    default_position = {"x": 0, "y": 1, "z": 7}
    default_target = {"x": 0, "y": 0, "z": 0}
    camera = {
        "position": default_position,
        "target": default_target
    }
    print(f"Added event: {event}")
    result = {
        "events": events,
        "camera": camera
    }
    print(f"Returning result with events: {events}")
    return result

# Export the tools as a list of callables
MODEL_CONTROL_TOOL_FUNCTIONS = [
    select_muscles_tool,
    toggle_muscle_tool,
    set_animation_frame_tool,
    toggle_animation_tool,
    set_camera_position_tool,
    set_camera_target_tool,
    reset_camera_tool,
] 