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
    thread_id: str
    user_id: str
    messages: List[Dict[str, Any]]
    highlighted_muscles: Dict[str, str]
    animation: Dict[str, Any]
    camera: Camera
    current_agent: Literal["muscle_expert", "animation_expert", "camera_expert"]
    events: List[Dict[str, Any]]

    pending_tool_calls: List[Dict] | None
    assistant_draft: Optional[str]
    _route_camera: Optional[bool]
    _route_muscle: Optional[bool]
    _route: Optional[str]
    _planner_iterations: Optional[int]
    _tool_executions: Optional[int]
    _just_executed_muscle_tools: Optional[bool]
    pending_muscle_tool_calls: Optional[List[Dict[str, Any]]]
    pending_camera_tool_calls: Optional[List[Dict[str, Any]]]
    _control_history: Optional[Dict[str, Any]]