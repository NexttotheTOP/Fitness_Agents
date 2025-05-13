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
    current_agent: Literal["conversation_agent", "tool_agent"]
    # Communication between agents
    agent_request: Optional[str]  # Request from conversation agent to tool agent
    user_question: Optional[str]  # Original user question for context
    tool_agent_report: Optional[str]  # Report from tool agent back to conversation agent
    # Debugging/tracking
    events: List[Dict[str, Any]]  # To track model control events 