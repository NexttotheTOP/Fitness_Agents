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
    reps: Optional[int] = None  # For rep-based exercises
    duration: Optional[str] = None  # For time-based exercises (e.g. "30 seconds", "1 minute")
    notes: Optional[str] = None
    details: ExerciseDetails
    
    class Config:
        extra = "allow"

class Workout(BaseModel):
    name: str
    description: str
    exercises: List[Exercise]
    difficulty_level: Optional[str] = None  # beginner, intermediate, advanced
    estimated_duration: Optional[str] = None  # e.g. "45 minutes"
    target_muscle_groups: Optional[List[str]] = None
    equipment_required: Optional[List[str]] = None
    
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
    
    # Structured previous overview sections
    previous_sections: Optional[Dict[str, str]]  # Previous overview parsed into sections 


class StateForWorkoutApp(TypedDict):
    """State for the workout creation app with streamlined fields"""
    # User identification
    user_id: str  # Primary identifier for the user
    thread_id: Optional[str]  # Used for LangGraph state management
    
    # Core workout fields
    workout_prompt: str  # NLQ prompt (will contain different content based on operation)
    workflow_type: str  # "create" or "variation" to explicitly state the intention
    original_workout: Optional[Workout]  # Original workout (required for variations)
    created_workouts: List[Workout]  # List of newly created workouts
    created_exercises: List[Exercise]  # Standalone exercises generated (optional)
    variations: List[Workout]  # Workout variations if requested
    
    # Context from frontend
    context: Optional[Dict[str, List[Dict[str, Any]]]]  # Referenced exercises and workouts from frontend
    workout_profile_analysis: Optional[str]  # Full text output from the analysis/proposal agent
    
    # User profile context fields  
    user_profile: Dict[str, Any]  # User profile data
    profile_assessment: Optional[str]  # Extracted from profile overview
    body_analysis: Optional[str]  # Body analysis if available
    
    # Reference data
    previous_complete_response: Optional[str]  # Previous complete response
    previous_sections: Optional[Dict[str, str]]  # Previous sections
    reasoning: Optional[str]  # Reasoning for workout creation
    plan_proposal_markdown: Optional[str]  # Plan proposal for workout creation
    
    # Conversation history for HITL
    analysis_conversation_history: Optional[List[Dict[str, str]]]  # Conversation during analysis phase
    pending_user_input: Optional[str]  # Question waiting for user response
    needs_user_input: Optional[bool]  # Flag indicating if user input is needed



class WorkoutsResponse(BaseModel):
    """Top-level container for a list of full workout definitions."""

    workouts: List[Workout]

    class Config:
        extra = "allow"


class ExercisesResponse(BaseModel):
    """Top-level container for a list of individual exercises that are **not**
    grouped into workouts.  This enables the system to flexibly return a set
    of exercises when generating a complete workout is unnecessary (e.g., the
    user only requests exercise ideas or substitutes)."""

    exercises: List[Exercise]

    class Config:
        extra = "allow"