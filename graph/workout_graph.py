from langgraph.graph import END, StateGraph

from graph.workout_state import StateForWorkoutApp
from graph.nodes.workout_variation import generate_workout_variation
from graph.nodes.workout_creation import create_workout_from_nlq
from graph.nodes.workout_analysis import WorkoutAnalysisAgent
from graph.nodes.workout_proposal import propose_workout_plan
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from graph.memory_store import get_most_recent_profile_overview
from graph.chains.workout_router import route_by_workflow_type

def create_workout_graph():
    """Create the workout graph."""
    # Create a new graph
    workflow = StateGraph(StateForWorkoutApp)

    # Add the nodes
    workflow.add_node("analyze_profile", WorkoutAnalysisAgent().analyze_user_profile)
    workflow.add_node("route", route_by_workflow_type)
    workflow.add_node("propose_plan", propose_workout_plan)
    workflow.add_node("create_workout", create_workout_from_nlq)
    workflow.add_node("generate_variations", generate_workout_variation)

    # Add the edges
    # First analyze profile, then route
    workflow.add_edge("analyze_profile", "route")
    
    # Route to appropriate node based on "next" key
    workflow.add_conditional_edges(
        "route",
        lambda x: x["next"],
        {
            "create_workout": "propose_plan",
            "generate_variations": "generate_variations"
        }
    )
    # Proposal node only before create_workout
    workflow.add_edge("propose_plan", "create_workout")
    
    # Finally end
    workflow.add_edge("create_workout", END)
    workflow.add_edge("generate_variations", END)

    # Set the entry point to profile analysis
    workflow.set_entry_point("analyze_profile")

    return workflow.compile()

app = create_workout_graph()

def initialize_workout_state(
    user_id: str,
    workout_prompt: str,
    workflow_type: str = "create",
    original_workout: Optional[Dict] = None,
    thread_id: Optional[str] = None
) -> StateForWorkoutApp:
    """
    Initialize state for workout app
    
    Args:
        user_id: User identifier
        workout_prompt: Natural language query for workout
        workflow_type: "create" or "variation"
        original_workout: Original workout for variations (required if workflow_type="variation")
        thread_id: Optional thread ID, will be generated if not provided
    """
    print("\n---INITIALIZING WORKOUT STATE---")
    print(f"User ID: {user_id}")
    print(f"Workflow Type: {workflow_type}")
    print(f"Thread ID: {thread_id}")
    
    # Generate thread_id if not provided
    if not thread_id:
        thread_id = str(uuid.uuid4())
        print(f"Generated new thread ID: {thread_id}")
        
    # Fetch user profile data from database
    print("\n---FETCHING BACKEND PROFILE DATA---")
    profile_data = get_most_recent_profile_overview(user_id)
    
    if profile_data:
        print(f"Found profile data in backend: {profile_data}")
        print(f"- Response length: {len(profile_data.get('response', ''))}")
        print(f"- Metadata keys: {list(profile_data.get('metadata', {}).keys())}")
    else:
        print("No profile data found in backend")
    
    # Initialize variables for profile data
    profile_sections = {}
    profile_assessment = ""
    body_analysis = ""
    
    if profile_data:
        # Get the full response
        previous_response = profile_data.get("response", "")
        print(f"\n---PARSING BACKEND PROFILE SECTIONS---")
        print(f"Previous response length: {len(previous_response)}")
        
        # Parse sections from the response
        profile_sections = parse_markdown_sections(previous_response)
        print(f"Found sections: {list(profile_sections.keys())}")
        
        # Get specific sections we need
        profile_assessment = profile_sections.get("profile_assessment", "")
        body_analysis = profile_sections.get("body_composition_analysis", "")
        
        print(f"\nExtracted sections:")
        print(f"- Profile assessment length: {len(profile_assessment)}")
        print(f"- Profile assessment preview: {profile_assessment[:200]}...")
        print(f"- Body analysis length: {len(body_analysis)}")
        print(f"- Body analysis preview: {body_analysis[:200]}...")
    
    # Create the initial state
    print("\n---CREATING INITIAL STATE---")
    state = {
        # User identification
        "user_id": user_id,
        "thread_id": thread_id,
        
        # Core workout fields
        "workout_prompt": workout_prompt,
        "workflow_type": workflow_type,
        "original_workout": None,  # Will be populated below if provided
        "created_workouts": [],
        "variations": [],
        
        # Analysis fields
        "analysis": {},
        "generation_status": "pending",
        
        # User profile context
        "user_profile": profile_data.get("metadata", {}),  # Use backend metadata
        "profile_assessment": profile_assessment,
        "body_analysis": body_analysis,
        "health_limitations": [],  # Will be populated by the analysis agent
        
        # Reference data
        "previous_complete_response": profile_data.get("response", "") if profile_data else "",
        "previous_sections": profile_sections
    }
    
    print("\nInitial state created with profile data:")
    print(f"- Profile assessment length: {len(state['profile_assessment'])}")
    print(f"- Body analysis length: {len(state['body_analysis'])}")
    print(f"- User profile keys: {list(state['user_profile'].keys())}")
    
    # Handle original_workout for variations
    if workflow_type == "variation" and original_workout:
        from graph.workout_state import Workout
        try:
            print("\nValidating original workout...")
            # Convert the dict to a Workout object
            workout_obj = Workout.model_validate(original_workout)
            state["original_workout"] = workout_obj
            print(f"Successfully validated original workout: {workout_obj.name}")
        except Exception as e:
            print(f"Error converting original_workout: {str(e)}")
            # If conversion fails, set workflow_type to create
            state["workflow_type"] = "create"
            print("Falling back to create workflow")
    
    print("\n---STATE INITIALIZATION COMPLETE---")
    return state

def parse_markdown_sections(markdown_text: str) -> dict:
    """
    Parse markdown text and extract sections between h2 headers.
    Returns a dictionary with section names as keys and content as values.
    
    Args:
        markdown_text: The markdown text to parse
        
    Returns:
        dict: Dictionary with section names mapped to their content
    """
    print("\nParsing markdown sections...")
    
    # Initialize sections dict
    sections = {}
    
    # Split text by h2 headers (##)
    parts = markdown_text.split("\n## ")
    
    # Process each section
    for part in parts[1:]:  # Skip first part (before first h2)
        # Split into title and content
        lines = part.split("\n")
        title = lines[0].strip().lower().replace(" ", "_")  # Normalize title
        content = "\n".join(lines[1:]).strip()
        
        print(f"Found section: {title}")
        sections[title] = content
    
    return sections 