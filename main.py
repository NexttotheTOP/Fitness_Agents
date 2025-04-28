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
from graph.fitness_coach_graph import get_initial_state, process_query, stream_response
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
    thread_id: str
    query: str

class FitnessRAGQueryRequest(BaseModel):
    query: str
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

async def generate_complete_response(state: dict) -> str:
    """Generate complete response without streaming"""
    try:
        from graph.nodes.fitness_coach import DietaryAgent, FitnessAgent, ProfileAgent
        
        # Create direct instances of the agents
        profile_agent = ProfileAgent()
        dietary_agent = DietaryAgent()
        fitness_agent = FitnessAgent()
        
        # Generate profile assessment directly (non-streaming)
        profile_response = profile_agent(state)
        profile_content = profile_response["user_profile"]
        
        # Generate dietary content directly (non-streaming)
        dietary_response = dietary_agent(state)
        dietary_content = dietary_response["dietary_state"].content
        
        # Generate fitness content directly (non-streaming)
        fitness_response = fitness_agent(state)
        fitness_content = fitness_response["fitness_state"].content
        
        # Update state with the generated content
        state["user_profile"] = profile_content
        state["dietary_state"].content = dietary_content
        state["fitness_state"].content = fitness_content
        
        # Combine with markdown formatting
        complete_response = "## Profile Assessment\n\n"
        complete_response += profile_content if isinstance(profile_content, str) else json.dumps(profile_content, indent=2)
        complete_response += "\n\n## Dietary Plan\n\n"
        complete_response += dietary_content
        complete_response += "\n\n## Fitness Plan\n\n"
        complete_response += fitness_content
        complete_response += "\n\n---\n\n"
        complete_response += "Your personalized fitness and dietary plans have been created. You can now ask specific questions about your plans."
        
        return complete_response
    except Exception as e:
        logging.error(f"Error generating complete response: {str(e)}")
        return f"Error: {str(e)}"

@api.post("/fitness/analyze-body-composition")
async def analyze_body_composition(request: BodyCompositionRequest):
    """Analyze body composition using vision model"""
    try:
        # Generate thread_id if not provided
        thread_id = request.thread_id or str(uuid.uuid4())
        
        # Create initial state with user profile and image paths
        user_profile = request.profileData
        user_profile["imagePaths"] = request.imagePaths
        user_profile["userId"] = request.userId
        user_profile["profileId"] = request.profileId
        
        initial_state = get_initial_state(user_profile=user_profile)
        initial_state["thread_id"] = thread_id
        
        # Store session state
        active_sessions[thread_id] = initial_state
        
        # Instead of using ProfileAgent directly, create a custom async flow
        from graph.nodes.fitness_coach import ProfileAgent, DietaryAgent, FitnessAgent
        profile_agent = ProfileAgent()
        
        # Check if Supabase credentials are available
        if not profile_agent.supabase_service_key:
            logging.warning("Supabase service key not found, but body images were provided. Image analysis may fail.")
        
        # Save a copy of the original user profile data before it gets converted to a string
        original_user_profile = user_profile.copy()
        
        # First, get a basic profile assessment (this is synchronous)
        basic_state = profile_agent(initial_state)
        
        # IMPORTANT: Restore the original user profile dictionary for the vision analysis
        # but keep the profile text analysis for later use
        profile_text_analysis = basic_state["user_profile"]
        basic_state["user_profile"] = original_user_profile
        basic_state["profile_text_analysis"] = profile_text_analysis
        
        # Now, perform the async body analysis with the original dict
        try:
            body_analysis = await profile_agent._analyze_body_composition(basic_state["user_profile"])
            
            # If image analysis succeeded, store it
            if body_analysis and len(body_analysis) > 50:  # If we got a real analysis
                # Store the body analysis in the state
                basic_state["body_analysis"] = body_analysis
                
                # Process the profile with the body analysis included
                profile_prompt = f"""
                Analyze the following user profile information and body analysis to create 
                a comprehensive fitness profile:
                
                User Profile Information: {json.dumps(basic_state["user_profile"])}
                
                Initial Profile Assessment: {basic_state["profile_text_analysis"]}
                
                Body Analysis: {body_analysis}
                
                Provide a comprehensive profile assessment that combines this information into a 
                cohesive assessment for the user. Focus on how their body composition, structure,
                and goals align to create actionable fitness recommendations.
                """
            else:
                # If body analysis failed but we have image paths, log diagnostic info
                logging.warning(f"Body analysis failed or returned insufficient content despite having image paths. Using fallback approach.")
                
                # Log image access errors if present for better diagnostics
                if "image_access_errors" in basic_state["user_profile"]:
                    logging.error(f"Image access errors: {json.dumps(basic_state['user_profile']['image_access_errors'])}")
                
                # Check if there are actual URLs in the imagePaths that we tried to access
                image_paths_flat = []
                for view_type in ['front', 'side', 'back']:
                    if view_type in basic_state["user_profile"].get("imagePaths", {}):
                        image_paths_flat.extend(basic_state["user_profile"].get("imagePaths", {}).get(view_type, []))
                
                profile_prompt = f"""
                Analyze the following user profile information to create 
                a comprehensive fitness profile:
                
                User Profile Information: {json.dumps(basic_state["user_profile"])}
                
                Initial Profile Assessment: {basic_state["profile_text_analysis"]}
                
                Note: Body images were provided but could not be analyzed successfully.
                This is likely due to Supabase authentication or incorrect image paths.
                Please focus on the available information and provide general recommendations
                based on the user's stated goals and metrics.
                
                Provide a comprehensive profile assessment based on the available information.
                """
        except Exception as e:
            logging.error(f"Error during body analysis: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            body_analysis = None
            
            # If body analysis failed with an exception, proceed without it
            profile_prompt = f"""
            Analyze the following user profile information to create 
            a comprehensive fitness profile:
            
            User Profile Information: {json.dumps(basic_state["user_profile"])}
            
            Initial Profile Assessment: {basic_state["profile_text_analysis"]}
            
            Provide a comprehensive profile assessment based on the available information.
            Note that body composition analysis was not possible due to technical issues, 
            so focus on the user's stated goals, preferences, and other available information.
            """
        
        # Get profile assessment
        profile_analysis = await profile_agent.llm.ainvoke(profile_prompt)
        
        # Store both the structured data and the text analysis
        basic_state["structured_user_profile"] = basic_state["user_profile"]
        basic_state["user_profile"] = profile_analysis.content
        
        # Store updated state
        active_sessions[thread_id] = basic_state
        
        # Now generate dietary and fitness plans
        dietary_agent = DietaryAgent()
        fitness_agent = FitnessAgent()
        
        # Generate dietary content
        dietary_state = dietary_agent(basic_state)
        
        # Generate fitness content
        fitness_state = fitness_agent(dietary_state)
        
        # Update final state
        active_sessions[thread_id] = fitness_state
        
        # Format the complete response
        complete_response = "## Profile Assessment\n\n"
        complete_response += fitness_state["user_profile"] if isinstance(fitness_state["user_profile"], str) else json.dumps(fitness_state["user_profile"], indent=2)
        
        if body_analysis and len(body_analysis) > 50:  # Only include if we got a real analysis
            complete_response += "\n\n## Body Composition Analysis\n\n"
            complete_response += body_analysis
        
        complete_response += "\n\n## Dietary Plan\n\n"
        complete_response += fitness_state["dietary_state"].content
        complete_response += "\n\n## Fitness Plan\n\n"
        complete_response += fitness_state["fitness_state"].content
        complete_response += "\n\n---\n\n"
        complete_response += "Your personalized fitness and dietary plans have been created. You can now ask specific questions about your plans."
        
        return {
            "thread_id": thread_id, 
            "body_analysis": body_analysis,
            "image_timestamps": basic_state["structured_user_profile"].get("image_timestamps", {}),
            "content": complete_response
        }
    
    except Exception as e:
        logging.error(f"Error analyzing body composition: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@api.post("/fitness/profile")
async def create_fitness_profile(profile: FitnessProfileRequest):
    """Create a new fitness coaching session with user profile"""
    try:
        # Generate thread_id if not provided
        thread_id = profile.thread_id or str(uuid.uuid4())
        
        # Create initial state
        user_profile = profile.model_dump(exclude={"thread_id"})  # Use model_dump instead of dict
        initial_state = get_initial_state(user_profile=user_profile)
        initial_state["thread_id"] = thread_id
        
        # Store session state
        active_sessions[thread_id] = initial_state
        
        # Check if we have body photos to analyze
        has_body_photos = False
        if isinstance(user_profile, dict):
            has_body_photos = "imagePaths" in user_profile and any(user_profile.get("imagePaths", {}).values())
        
        body_analysis = None
        
        if has_body_photos:
            # Use the same approach as the analyze-body-composition endpoint
            from graph.nodes.fitness_coach import ProfileAgent, DietaryAgent, FitnessAgent
            profile_agent = ProfileAgent()
            
            # Check if Supabase credentials are available
            if not profile_agent.supabase_service_key and has_body_photos:
                logging.warning("Supabase service key not found, but body images were provided. Image analysis may fail.")
            
            # Save a copy of the original user profile data before it gets converted to a string
            original_user_profile = user_profile.copy()
            
            # First, get a basic profile assessment (this is synchronous)
            basic_state = profile_agent(initial_state)
            
            # IMPORTANT: Restore the original user profile dictionary for the vision analysis
            # but keep the profile text analysis for later use
            profile_text_analysis = basic_state["user_profile"]
            basic_state["user_profile"] = original_user_profile
            basic_state["profile_text_analysis"] = profile_text_analysis
            
            try:
                # Now, perform the async body analysis with the original dict
                # Call the async method directly
                body_analysis = await profile_agent._analyze_body_composition(basic_state["user_profile"])
                
                if body_analysis and len(body_analysis) > 50:  # If we got a real analysis
                    # Store the body analysis in the state
                    basic_state["body_analysis"] = body_analysis
                    
                    # Process the profile with the body analysis included
                    profile_prompt = f"""
                    Analyze the following user profile information and body analysis to create 
                    a comprehensive fitness profile:
                    
                    User Profile Information: {json.dumps(basic_state["user_profile"])}
                    
                    Initial Profile Assessment: {basic_state["profile_text_analysis"]}
                    
                    Body Analysis: {body_analysis}
                    
                    Provide a comprehensive profile assessment that combines this information into a 
                    cohesive assessment for the user. Focus on how their body composition, structure,
                    and goals align to create actionable fitness recommendations.
                    """
                else:
                    # If body analysis failed but we have image paths, there might be an issue with image access
                    logging.warning(f"Body analysis failed or returned insufficient content despite having image paths. Using fallback approach.")
                    
                    # Log image access errors if present for better diagnostics
                    if "image_access_errors" in basic_state["user_profile"]:
                        logging.error(f"Image access errors: {json.dumps(basic_state['user_profile']['image_access_errors'])}")
                    
                    # Check if there are actual URLs in the imagePaths that we tried to access
                    image_paths_flat = []
                    for view_type in ['front', 'side', 'back']:
                        if view_type in basic_state["user_profile"].get("imagePaths", {}):
                            image_paths_flat.extend(basic_state["user_profile"].get("imagePaths", {}).get(view_type, []))
                    
                    if image_paths_flat:
                        # There are paths but we couldn't access them - include this info in the prompt
                        profile_prompt = f"""
                        Analyze the following user profile information to create 
                        a comprehensive fitness profile:
                        
                        User Profile Information: {json.dumps(basic_state["user_profile"])}
                        
                        Initial Profile Assessment: {basic_state["profile_text_analysis"]}
                        
                        Note: Body images were provided but could not be analyzed due to access issues.
                        This is likely due to Supabase authentication or incorrect image paths.
                        Please focus on the available information and provide general recommendations
                        based on the user's stated goals and metrics.
                        
                        Provide a comprehensive profile assessment based on the available information.
                        """
                    else:
                        # If body analysis failed, proceed without it
                        profile_prompt = f"""
                        Analyze the following user profile information to create 
                        a comprehensive fitness profile:
                        
                        User Profile Information: {json.dumps(basic_state["user_profile"])}
                        
                        Initial Profile Assessment: {basic_state["profile_text_analysis"]}
                        
                        Provide a comprehensive profile assessment based on the available information.
                        Note that body composition analysis was not possible, so focus on the user's
                        stated goals, preferences, and other available information.
                        """
            except Exception as e:
                logging.error(f"Error during body analysis: {str(e)}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                # If body analysis failed, proceed without it
                profile_prompt = f"""
                Analyze the following user profile information to create 
                a comprehensive fitness profile:
                
                User Profile Information: {json.dumps(basic_state["user_profile"])}
                
                Initial Profile Assessment: {basic_state["profile_text_analysis"]}
                
                Provide a comprehensive profile assessment based on the available information.
                Note that body composition analysis was not possible due to an error, so focus on the user's
                stated goals, preferences, and other available information.
                """
            
            # Get profile assessment
            profile_analysis = await profile_agent.llm.ainvoke(profile_prompt)
            
            # Store both the structured data and the text analysis
            basic_state["structured_user_profile"] = basic_state["user_profile"]
            basic_state["user_profile"] = profile_analysis.content
            
            # Store updated state
            active_sessions[thread_id] = basic_state
            
            # Now generate dietary and fitness plans
            dietary_agent = DietaryAgent()
            fitness_agent = FitnessAgent()
            
            # Generate dietary content
            dietary_state = dietary_agent(basic_state)
            
            # Generate fitness content
            fitness_state = fitness_agent(dietary_state)
            
            # Update final state
            active_sessions[thread_id] = fitness_state
            
            # Format the complete response
            complete_response = "## Profile Assessment\n\n"
            complete_response += fitness_state["user_profile"] if isinstance(fitness_state["user_profile"], str) else json.dumps(fitness_state["user_profile"], indent=2)
            
            if body_analysis and len(body_analysis) > 50:  # Only include if we got a real analysis
                complete_response += "\n\n## Body Composition Analysis\n\n"
                complete_response += body_analysis
            
            complete_response += "\n\n## Dietary Plan\n\n"
            complete_response += fitness_state["dietary_state"].content
            complete_response += "\n\n## Fitness Plan\n\n"
            complete_response += fitness_state["fitness_state"].content
            complete_response += "\n\n---\n\n"
            complete_response += "Your personalized fitness and dietary plans have been created. You can now ask specific questions about your plans."
            
            return {
                "thread_id": thread_id, 
                "body_analysis": body_analysis,
                "image_timestamps": basic_state["structured_user_profile"].get("image_timestamps", {}),
                "content": complete_response
            }
        else:
            # No body photos, use the standard approach
            # Generate complete response instead of streaming
            complete_response = await generate_complete_response(initial_state)
            
            return {"thread_id": thread_id, "content": complete_response}
    
    except Exception as e:
        logging.error(f"Error creating fitness profile: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@api.post("/fitness/query")
async def process_fitness_query(query: FitnessQueryRequest):
    """Process a query in the fitness coaching session with streaming response"""
    try:
        thread_id = query.thread_id
        if thread_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get current session state
        current_state = active_sessions[thread_id]
        
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
            "query": query.query
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
            "image_timestamps": image_timestamps
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)
