import uuid
from datetime import datetime
from typing import Dict, Any, AsyncGenerator, AsyncIterable
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import json

from graph.workout_state import WorkoutState, AgentState, QueryType
from graph.nodes.fitness_coach import ProfileAgent, DietaryAgent, FitnessAgent, QueryAgent
from graph.nodes.query_analyzer import QueryAnalyzer

def create_fitness_coach_workflow():
    """Create the fitness coach workflow with memory and thread management"""
    workflow = StateGraph(WorkoutState)
    
    # Add nodes for each agent
    workflow.add_node("analyzer", QueryAnalyzer())
    workflow.add_node("profile", ProfileAgent())
    workflow.add_node("dietary", DietaryAgent())
    workflow.add_node("fitness", FitnessAgent())
    workflow.add_node("query", QueryAgent())
    
    # Define conditional edges based on analysis
    def route_by_analysis(state: WorkoutState) -> str | list[str]:
        if not state["user_profile"]:
            return "profile"
        
        analysis = state.get("query_analysis", {})
        
        # Determine which agents to run based on analysis
        agents_to_run = []
        if analysis.get("should_run_dietary"):
            agents_to_run.append("dietary")
        if analysis.get("should_run_fitness"):
            agents_to_run.append("fitness")
        if analysis.get("should_run_general"):
            agents_to_run.append("query")
            
        if not agents_to_run:
            return "query"  # Default to general query if no specific agents
            
        # If only one agent, return string, otherwise return list
        return agents_to_run[0] if len(agents_to_run) == 1 else agents_to_run
    
    # Add edges for initial profile setup
    workflow.add_edge("profile", "dietary")
    workflow.add_edge("dietary", "fitness")
    workflow.add_edge("fitness", END)
    
    # Add edges for query handling
    workflow.add_conditional_edges(
        "analyzer",
        route_by_analysis,
        {
            "dietary": END,
            "fitness": END,
            "query": END,
            "profile": "profile"
        }
    )
    
    # Set entry point
    workflow.set_entry_point("analyzer")
    
    # Create the compiler with memory management
    return workflow.compile(checkpointer=MemorySaver())

# Create the app instance
app = create_fitness_coach_workflow()

def get_initial_state(user_profile: dict = None) -> WorkoutState:
    """Create initial state with a new thread ID"""
    return WorkoutState(
        thread_id=str(uuid.uuid4()),
        user_profile=user_profile or {},
        dietary_state=AgentState(
            last_update=datetime.now().isoformat(),
            content="",
            is_streaming=False
        ),
        fitness_state=AgentState(
            last_update=datetime.now().isoformat(),
            content="",
            is_streaming=False
        ),
        current_query="",
        query_analysis={},
        conversation_history=[],
        original_workout=None,
        variations=[],
        analysis={},
        generation=None
    )

async def stream_response(state: WorkoutState, query: str = None) -> AsyncIterable[str]:
    """Stream responses from the appropriate agents based on query analysis"""
    if query:
        # Query mode - analyze and respond
        analyzer = QueryAnalyzer()
        query_agent = QueryAgent()
        
        analysis = analyzer.analyze(query, state["user_profile"])
        state["query_analysis"] = analysis.dict()
        
        # Stream response through query agent
        async for chunk in query_agent.stream(state):
            if chunk:
                yield chunk
    else:
        # Profile creation mode - use simple markdown streaming
        dietary_agent = DietaryAgent()
        fitness_agent = FitnessAgent()
        
        # Initial header
        yield "# Creating Your Personalized Fitness Profile\n\n"
        
        # Generate dietary recommendations
        yield "## Dietary Plan\n\n"
        async for chunk in dietary_agent.stream(state):
            if chunk:
                yield chunk
        
        # Generate fitness recommendations
        yield "\n## Fitness Plan\n\n"
        async for chunk in fitness_agent.stream(state):
            if chunk:
                yield chunk
        
        # Final message
        yield "\n---\n\n"
        yield "Your personalized fitness and dietary plans have been created. You can now ask specific questions about your plans."

def process_query(state: WorkoutState, query: str, config: dict = None) -> WorkoutState:
    """Process a user query using the existing thread"""
    # Create a new state with the existing data plus the query
    query_state = WorkoutState(
        thread_id=state["thread_id"],
        user_profile=state["user_profile"],
        dietary_state=state["dietary_state"],
        fitness_state=state["fitness_state"],
        current_query=query,
        query_analysis={},  # Will be set by analyzer
        conversation_history=state["conversation_history"],
        original_workout=state.get("original_workout"),
        variations=state.get("variations", []),
        analysis=state.get("analysis", {}),
        generation=state.get("generation")
    )
    
    if config is None:
        config = {
            "configurable": {
                "thread_id": state["thread_id"],
                "checkpoint_ns": "fitness_coach",
                "checkpoint_id": f"session_{state['thread_id']}"
            }
        }
    
    # Process the query through the workflow
    result = app.invoke(query_state, config=config)
    
    # Update the session state with the results
    state.update(result)
    return state 