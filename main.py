from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import json
from typing import Optional, List, Dict, Any, AsyncIterable
import uuid
import logging
from datetime import datetime
import traceback

load_dotenv()

from graph.graph import app as qa_app
from graph.workout_graph import app as workout_app
from graph.workout_state import Workout, UserProfile, WorkoutState, AgentState, QueryType
from graph.fitness_coach_graph import get_initial_state, process_query, stream_response, app 
from fastapi.middleware.cors import CORSMiddleware
from graph.chains.workout_variation import analyze_workout

# Add RAG imports
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
# Import get_vectorstore from ingestion
from ingestion import get_vectorstore

# Create FastAPI app
api = FastAPI(title="Fitness Coach API")

# Add CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Accept"],
)

# Initialize RAG system
def init_rag_system():
    try:
        # Load the vector database using the get_vectorstore function
        vectorstore = get_vectorstore()
        
        if vectorstore is None:
            logging.warning("Vector database not found. Run the data collection scripts first.")
            return None
        
        # Create a retriever
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={"k": 5, "fetch_k": 10}  # Retrieve top 5 documents from top 10 candidates
        )
        
        # Create the QA chain
        llm = ChatOpenAI(temperature=0, model="gpt-4")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # Stuff documents into a single prompt
            retriever=retriever,
            return_source_documents=True,
        )
        
        logging.info("RAG system initialized successfully")
        return qa_chain
    except Exception as e:
        logging.error(f"Error initializing RAG system: {str(e)}")
        return None

# Initialize RAG system on startup
rag_system = init_rag_system()

# Create request models
class Question(BaseModel):
    question: str

class FitnessProfileRequest(BaseModel):
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    age: int
    gender: str
    height: str  # Changed from float to str to accept "199cm" format
    weight: str  # Changed from float to str to accept "97kg" format
    activity_level: str
    fitness_goals: list[str]
    dietary_preferences: list[str]
    health_restrictions: list[str]
    body_type: Optional[str] = None  # Analysis result of body type
    imagePaths: Optional[Dict[str, List[str]]] = None  # Structured image paths with front, side, back views

class BodyCompositionRequest(BaseModel):
    userId: str
    profileId: str
    profileData: Dict[str, Any]
    imagePaths: Dict[str, List[str]]
    thread_id: Optional[str] = None

class FitnessQueryRequest(BaseModel):
    user_id: str
    thread_id: str
    query: str

class FitnessRAGQueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    thread_id: Optional[str] = None

# Store active sessions
active_sessions: Dict[str, Any] = {}

@api.post("/ask")
async def ask_question(question: Question):
    result = qa_app.invoke(input={"question": question.question})
    return {"response": result}

@api.post("/workout/variation")
async def generate_workout_variation(request: Request):
    # Get raw request data
    raw_data = await request.json()
    print("\n=== Raw Request Data ===")
    print(json.dumps(raw_data, indent=2))
    
    # Extract the workout data from requestData
    workout_data = raw_data.get("requestData")
    if not workout_data:
        return {"error": "No requestData found in request"}
    
    # Parse into Workout model
    try:
        workout = Workout.model_validate(workout_data)
        print("\n=== Parsed Workout ===")
        print(f"Name: {workout.name}")
        print(f"Description: {workout.description}")
        print(f"Number of exercises: {len(workout.exercises)}")
        
        # Generate variations
        result = workout_app.invoke({
            "original_workout": workout,
            "variations": [],
            "analysis": {},
            "generation": None
        })
        
        print("\n=== Generated Variations ===")
        print(f"Number of variations: {len(result.get('variations', []))}")
        
        return {"variations": result.get("variations", [])}
        
    except Exception as e:
        print(f"\n=== Error ===\n{str(e)}")
        return {"error": str(e)}

async def generate_response_stream(state: dict, query: str = None) -> AsyncIterable[str]:
    """Generate streaming response in SSE format"""
    try:
        async for chunk in stream_response(state, query):
            # Ensure the chunk is properly formatted for SSE
            if chunk:
                yield f"data: {json.dumps({'content': chunk})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        yield "data: [DONE]\n\n"

@api.post("/fitness/profile")
async def create_fitness_profile(profile: FitnessProfileRequest):
    """Create a new fitness coaching session with user profile"""
    try:
        # Convert the request to a dict, keeping thread_id and user_id
        user_profile = profile.model_dump()
        
        # Use user_id from the profile or generate a new one if not provided
        user_id = profile.user_id or str(uuid.uuid4())
        user_profile["user_id"] = user_id
        
        # Use thread_id from the profile or generate a new one if not provided
        thread_id = profile.thread_id or str(uuid.uuid4())
        
        # Create initial state
        initial_state = get_initial_state(user_profile=user_profile, thread_id=thread_id, user_id=user_id)
        
        # Store the original user_profile dictionary
        original_user_profile = user_profile.copy()
        initial_state["original_user_profile"] = original_user_profile
        
        # Check if we have body photos to analyze
        has_body_photos = False
        if isinstance(user_profile, dict):
            has_body_photos = "imagePaths" in user_profile and any(user_profile.get("imagePaths", {}).values())
        
        # Process body analysis before running the graph
        body_analysis = None
        if has_body_photos:
            # Use the ProfileAgent for body analysis
            from graph.nodes.fitness_coach import ProfileAgent
            profile_agent = ProfileAgent()
            
            try:
                # Perform body analysis
                body_analysis = await profile_agent._analyze_body_composition(initial_state["user_profile"])
                
                # Store body analysis in state if it contains actual analysis
                if body_analysis and len(body_analysis) > 50:
                    initial_state["body_analysis"] = body_analysis
                    
                    # Store image timestamps if available
                    if "image_timestamps" in initial_state["user_profile"]:
                        initial_state["image_timestamps"] = initial_state["user_profile"]["image_timestamps"]
            except Exception as e:
                logging.error(f"Error during body analysis: {str(e)}")
                logging.error(f"Traceback: {traceback.format_exc()}")
        
        # Set config for this run
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": f"session_{thread_id}"
            }
        }
        
        # Previous data already loaded in get_initial_state, so no need to load it here again
        # The previous_sections field should be populated with structured sections
        
        # Process through the workflow directly and generate complete content
        logging.info(f"Generating fitness profile for user {user_id} using stream_response")
        
        # Use the stream_response function to ensure storage happens
        complete_content = ""
        async for chunk in stream_response(initial_state):
            complete_content += chunk
        
        logging.info(f"Generated complete content, length: {len(complete_content)}")
        
        # Get the final state with complete_response from active_sessions if available
        # Or use the HeadCoachAgent to format it directly
        if thread_id in active_sessions:
            formatted_state = active_sessions[thread_id]
            logging.info("Retrieved formatted state from active_sessions")
        else:
            # If not in active_sessions (shouldn't happen but just in case)
            logging.info("State not found in active_sessions, processing directly")
            result_state = app.invoke(initial_state, config=config)
            from graph.nodes.fitness_coach import HeadCoachAgent
            head_coach = HeadCoachAgent()
            formatted_state = head_coach(result_state)
            active_sessions[thread_id] = formatted_state
        
        # Get image timestamps from user_profile_data
        image_timestamps = {}
        if "user_profile_data" in formatted_state and isinstance(formatted_state["user_profile_data"], dict):
            image_timestamps = formatted_state["user_profile_data"].get("image_timestamps", {})
        
        # Return the response with indicator for previous data
        has_previous_data = formatted_state.get("previous_sections") is not None or formatted_state.get("previous_complete_response") is not None
        
        return {
            "user_id": user_id,
            "thread_id": thread_id,
            "body_analysis": formatted_state.get("body_analysis"),
            "image_timestamps": image_timestamps,
            "content": formatted_state.get("complete_response", ""),
            "has_previous_data": has_previous_data
        }
    
    except Exception as e:
        logging.error(f"Error creating fitness profile: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@api.post("/fitness/query")
async def process_fitness_query(query: FitnessQueryRequest):
    """Process a query in the fitness coaching session with streaming response"""
    try:
        thread_id = query.thread_id
        user_id = query.user_id
        
        if thread_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get current session state
        current_state = active_sessions[thread_id]
        
        # Ensure user_id matches
        if current_state.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="User ID does not match session")
        
        # Enhance with RAG if available
        if rag_system and query.query:
            try:
                # Get relevant information from RAG system
                rag_result = rag_system({"query": query.query})
                
                # Add RAG results to state for context
                if "source_documents" in rag_result:
                    current_state["rag_context"] = [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "timestamp": datetime.now().isoformat()
                        }
                        for doc in rag_result["source_documents"]
                    ]
                    
                # Log retrieved contexts
                logging.info(f"RAG retrieved {len(current_state.get('rag_context', []))} documents for query: {query.query}")
            except Exception as e:
                logging.error(f"Error using RAG system: {str(e)}")
        
        # Process the query using our simplified query handler
        updated_state = process_query(current_state, query.query)
        
        # Update the active session
        active_sessions[thread_id] = updated_state
        
        # Get the response directly from the conversation history
        if updated_state["conversation_history"]:
            response = updated_state["conversation_history"][-1]["content"]
            return {"content": response, "user_id": user_id, "thread_id": thread_id}
        
        # If no history updated, return streaming response as fallback
        return StreamingResponse(
            generate_response_stream(current_state, query.query),
            media_type="text/event-stream"
        )
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error processing query for thread {thread_id}: {str(e)}")
        logging.error(f"Query: {query.query}")
        raise HTTPException(status_code=500, detail=str(e))

@api.post("/fitness/rag-query")
async def query_fitness_knowledge(query: FitnessRAGQueryRequest):
    """Directly query the fitness knowledge base using RAG"""
    try:
        if not rag_system:
            raise HTTPException(status_code=503, detail="RAG system not available")
        
        # Get result from RAG system
        result = rag_system({"query": query.query})
        
        # Extract source information
        sources = []
        if "source_documents" in result:
            sources = [
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "author": doc.metadata.get("author", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                for doc in result["source_documents"]
            ]
        
        return {
            "answer": result.get("result", "No answer found"),
            "sources": sources,
            "query": query.query,
            "user_id": query.user_id,
            "thread_id": query.thread_id
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error querying RAG system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/fitness/session/{thread_id}")
async def get_session_state(thread_id: str):
    """Get the current state of a fitness coaching session"""
    try:
        if thread_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        state = active_sessions[thread_id]
        
        # Get the image timestamps from structured profile if available
        image_timestamps = {}
        if "structured_user_profile" in state and isinstance(state["structured_user_profile"], dict):
            image_timestamps = state["structured_user_profile"].get("image_timestamps", {})
        
        return {
            "user_id": state.get("user_id"),
            "thread_id": thread_id,
            "user_profile": state["user_profile"],
            "dietary_state": {
                "content": state["dietary_state"].content,
                "last_update": state["dietary_state"].last_update,
                "is_streaming": state["dietary_state"].is_streaming
            },
            "fitness_state": {
                "content": state["fitness_state"].content,
                "last_update": state["fitness_state"].last_update,
                "is_streaming": state["fitness_state"].is_streaming
            },
            "conversation_history": state["conversation_history"][-10:],  # Last 10 messages
            "query_analysis": state.get("query_analysis", {}),  # Include the query analysis
            "rag_context_count": len(state.get("rag_context", [])) if "rag_context" in state else 0,
            "body_analysis": state.get("body_analysis", None),  # Include body analysis if available
            "image_timestamps": image_timestamps,
            "progress_comparison": state.get("progress_comparison", None),  # Include progress comparison if available
            "has_previous_data": "previous_complete_response" in state  # Indicate if there's previous data
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error retrieving session state for thread {thread_id}: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/fitness/rag-status")
async def get_rag_status():
    """Check the status of the RAG system"""
    try:
        if not rag_system:
            return {"status": "unavailable", "message": "RAG system not initialized"}
        
        # Try to access the vectorstore to confirm it's working
        vectorstore = get_vectorstore()
        if vectorstore is None:
            return {"status": "unavailable", "message": "Vector database not found. Run the data collection scripts first."}
            
        count = vectorstore._collection.count()
        
        return {
            "status": "available", 
            "document_count": count,
            "message": f"RAG system initialized with {count} document chunks"
        }
    except Exception as e:
        logging.error(f"Error checking RAG status: {str(e)}")
        return {"status": "error", "message": str(e)}

@api.get("/fitness/test-db")
async def test_db_connection():
    """Test the database connection and table setup"""
    try:
        from graph.memory_store import test_supabase_connection_and_table
        
        # Run the comprehensive test
        test_result = test_supabase_connection_and_table()
        
        if test_result:
            return {
                "status": "success",
                "message": "Database connection and table setup validated successfully"
            }
        else:
            return {
                "status": "error",
                "message": "Database connection or table setup has issues - check logs"
            }
    except Exception as e:
        logging.error(f"Error testing database connection: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to test database connection"
        }

@api.get("/fitness/profiles/{user_id}")
async def get_user_profiles(user_id: str):
    """Get all stored profiles for a specific user"""
    try:
        from graph.memory_store import get_previous_profile_overviews
        
        # Get all previous overviews for this user
        profiles = get_previous_profile_overviews(user_id, limit=10)
        
        # Format the response
        formatted_profiles = []
        for profile in profiles:
            # Limit the response text size for the overview
            response_preview = profile.get("response", "")
            if len(response_preview) > 500:
                response_preview = response_preview[:497] + "..."
                
            formatted_profiles.append({
                "id": profile.get("id"),
                "thread_id": profile.get("thread_id"),
                "timestamp": profile.get("timestamp"),
                "metadata": profile.get("metadata"),
                "response_preview": response_preview,
                "response_length": len(profile.get("response", ""))
            })
        
        return {
            "status": "success",
            "user_id": user_id,
            "profile_count": len(formatted_profiles),
            "profiles": formatted_profiles
        }
    except Exception as e:
        logging.error(f"Error retrieving profiles for user {user_id}: {str(e)}")
        return {
            "status": "error",
            "user_id": user_id,
            "error": str(e),
            "message": "Failed to retrieve profiles"
        }

@api.post("/fitness/test-storage")
async def test_profile_storage():
    """Test directly storing a profile in Supabase"""
    try:
        from graph.memory_store import store_profile_overview
        
        # Generate test data
        test_user_id = f"test_user_{uuid.uuid4()}"
        test_thread_id = f"test_thread_{uuid.uuid4()}"
        test_overview = f"This is a test overview from the direct storage test endpoint l. Generated at {datetime.now().isoformat()}"
        test_metadata = {
            "test": True,
            "age": 30,
            "timestamp": datetime.now().isoformat()
        }
        
        logging.info(f"Testing direct storage with user_id: {test_user_id}")
        
        # Try to store the test data
        result = store_profile_overview(
            test_user_id,
            test_thread_id,
            test_overview,
            test_metadata
        )
        
        # Then try to retrieve it
        from graph.memory_store import get_previous_profile_overviews
        retrieved = get_previous_profile_overviews(test_user_id, limit=1)
        
        return {
            "status": "success" if result and retrieved else "error",
            "storage_result": result,
            "retrieval_result": retrieved,
            "message": "Test storage and retrieval completed",
            "test_user_id": test_user_id,
            "test_thread_id": test_thread_id
        }
    except Exception as e:
        logging.error(f"Error testing profile storage: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to test profile storage"
        }

if __name__ == "__main__":
    import uvicorn
    import signal
    import sys
    
    # Define shutdown handler
    def handle_shutdown(signum, frame):
        """Handle graceful shutdown by closing connections"""
        print("\nShutting down server...")
        
        # Close any open connections or resources
        try:
            # Close PostgreSQL connection pool if it exists
            from graph.memory_store import _connection_pool
            if _connection_pool:
                print("Closing PostgreSQL connection pool...")
                _connection_pool.close()
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")
        
        print("Shutdown complete.")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)  # Ctrl+C
    signal.signal(signal.SIGTERM, handle_shutdown)  # kill command
    
    # Run the server
    uvicorn.run(api, host="0.0.0.0", port=8000)
