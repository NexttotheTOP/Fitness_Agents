from typing import TypedDict, List, Optional, Any, Dict, Literal
from pydantic import BaseModel

class ExerciseDetails(BaseModel):
    description: str
    category: str
    muscle_groups: List[str]
    difficulty: str
    equipment_needed: str
    
    class Config:
        extra = "allow"

class Exercise(BaseModel):
    name: str
    sets: int
    reps: int
    notes: Optional[str] = None
    details: ExerciseDetails
    
    class Config:
        extra = "allow"

class Workout(BaseModel):
    name: str
    description: str
    exercises: List[Exercise]
    
    class Config:
        extra = "allow"

# This matches the frontend request structure
class WorkoutRequest(BaseModel):
    timestamp: str
    requestData: Workout
    
    class Config:
        extra = "allow"

class UserProfile(BaseModel):
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    age: int = 30
    gender: str = "unspecified"
    height: str = "170cm"
    weight: str = "70kg"
    activity_level: str = "moderate"
    fitness_goals: List[str] = []
    dietary_preferences: List[str] = []
    health_restrictions: List[str] = []
    body_photos: Optional[List[str]] = None 
    body_type: Optional[str] = None  # Analysis result of body type
    imagePaths: Optional[Dict[str, List[str]]] = None  # Structured image paths with front, side, back views
    
    class Config:
        extra = "allow"

class QueryType(str):
    DIETARY = "dietary"
    FITNESS = "fitness"
    GENERAL = "general"
    PROFILE = "profile"

class AgentState(BaseModel):
    """State specific to each agent"""
    last_update: str = ""
    content: str = ""
    is_streaming: bool = False
    
    class Config:
        extra = "allow"

class WorkoutState(TypedDict):
    """Extended state that includes both workout and fitness coach data"""
    # Original workout fields
    original_workout: Workout
    variations: List[Workout]
    analysis: dict
    generation: Optional[str]
    
    # New fitness coach fields
    user_id: str  # Primary identifier for the user
    thread_id: str  # Used for LangGraph state management
    user_profile: Dict[str, Any]
    dietary_state: AgentState
    fitness_state: AgentState
    current_query: str
    query_type: str  # "dietary", "fitness", "general", or "profile"
    conversation_history: List[Dict[str, str]]
    
    # Body analysis fields
    body_analysis: Optional[str]  # Result of vision model body analysis
    
    # Complete response field
    complete_response: Optional[str]  # Final formatted response from HeadCoach
    
    # Progress tracking fields
    previous_complete_response: Optional[str]  # Previous response for comparison
    progress_comparison: Optional[str]  # Generated comparison of progress 