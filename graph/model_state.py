from typing import Dict, List, Optional, TypedDict, Any, Literal

class MuscleInfo(TypedDict):
    name: str
    description: Optional[str]
    functions: Optional[List[str]]
    muscles_worked: Optional[List[str]]

class Position(TypedDict):
    x: float
    y: float
    z: float

class Camera(TypedDict):
    position: Position
    target: Position

class ModelState(TypedDict):
    # Session identifiers
    thread_id: str
    user_id: str
    # Conversation state
    messages: List[Dict[str, Any]]
    # Model state
    highlighted_muscles: Dict[str, str]  # mapping of muscle name to color
    animation: Dict[str, Any]  # includes frame, isPlaying
    camera: Camera
    # Agent state
    current_agent: Literal["muscle_expert", "animation_expert", "camera_expert"]
    # Debugging/tracking
    events: List[Dict[str, Any]]  # To track model control events 