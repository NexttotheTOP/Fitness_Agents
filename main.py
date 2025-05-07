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
import asyncio
import re

load_dotenv()

from graph.graph import app as qa_app
from graph.workout_graph import app as workout_app, initialize_workout_state
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
    allow_origins=["*"],  # Add your Netlify domain and localhost for testing
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
    user_id: Optional[str] = None
    thread_id: Optional[str] = None

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

class WorkoutNLQRequest(BaseModel):
    user_id: str
    prompt: str
    thread_id: Optional[str] = None

class WorkoutVariationRequest(BaseModel):
    user_id: str
    original_workout: Dict[str, Any]
    thread_id: Optional[str] = None

# Store active sessions
active_sessions: Dict[str, Any] = {}

@api.post("/ask")
async def ask_question(question: Question):
    """
    Process a question and stream the response along with processing steps and sources.
    
    The streaming response follows this format:
    
    1. type: 'step' - Shows processing steps and routing decisions
       example: {"type": "step", "content": "Retrieving documents from vector database..."}
       
    2. type: 'source' - Information about a source document used for the answer
       example: {"type": "source", "content": {"content": "Document snippet...", "metadata": {...}}}
       
    3. type: 'answer' - The actual answer content, streamed in chunks
       example: {"type": "answer", "content": "Part of the answer text..."}
       
    4. type: 'sources_summary' - Summary of all sources at the end
       example: {"type": "sources_summary", "content": [{"content": "...", "metadata": {...}}, ...]}
       
    5. type: 'error' - Any error information
       example: {"type": "error", "content": "Error message"}
       
    6. type: 'metadata' - Metadata about the conversation
       example: {"type": "metadata", "content": {"thread_id": "abc123", "user_id": "user456"}}
       
    This endpoint returns text/event-stream content which can be consumed
    by the EventSource API in browsers or any SSE (Server-Sent Events) client.
    """
    try:
        # Generate user_id if not provided
        user_id = question.user_id or str(uuid.uuid4())
        
        # Generate thread_id if not provided (new conversation)
        thread_id = question.thread_id or str(uuid.uuid4())
        
        # Import conversation memory functions
        from graph.conversation_memory import get_conversation_history, store_conversation
        
        # Retrieve existing conversation history
        conversation_history = []
        if thread_id:
            try:
                conversation_history = get_conversation_history(thread_id) or []
                logging.info(f"Retrieved conversation history for thread {thread_id}, {len(conversation_history)} messages")
            except Exception as e:
                logging.error(f"Error retrieving conversation history: {e}")
        
        # Add current question to history
        conversation_history.append({
            "role": "user", 
            "content": question.question, 
            "timestamp": datetime.now().isoformat()
        })
        
        # Direct streaming from LangGraph
        async def stream_langgraph_response():
            try:
                # First get the routing decision and initial processing
                initial_state = {
                    "question": question.question, 
                    "documents": [], 
                    "generation": "", 
                    "web_search": False,
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "conversation_history": conversation_history
                }
                
                # Import the necessary components
                from graph.graph import route_question, workflow
                from graph.nodes.generate import generate_streaming
                from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
                from graph.nodes import retrieve, grade_documents, web_search
                
                # Collection for answer chunks
                answer_content = ""
                
                # Stream the step information
                yield f"data: {json.dumps({'type': 'step', 'content': 'Starting query processing...'})}\n\n"
                
                # Get routing decision
                route = route_question(initial_state)
                yield f"data: {json.dumps({'type': 'step', 'content': f'Routing decision: {route}'})}\n\n"
                
                # Initialize state based on routing
                state = initial_state.copy()
                
                # Process based on routing decision
                sources = []
                if route == RETRIEVE:
                    # Retrieve documents
                    yield f"data: {json.dumps({'type': 'step', 'content': 'Retrieving documents from vector database...'})}\n\n"
                    state = retrieve(state)
                    
                    # Debug logging for documents
                    doc_count = len(state.get("documents", []))
                    logging.info(f"Retrieved {doc_count} documents from vector database")
                    
                    # Grade documents
                    yield f"data: {json.dumps({'type': 'step', 'content': 'Grading retrieved documents for relevance...'})}\n\n"
                    pre_grade_docs = len(state.get("documents", []))
                    state = grade_documents(state)
                    post_grade_docs = len(state.get("documents", []))
                    logging.info(f"Document grading: {pre_grade_docs} before, {post_grade_docs} after")
                    
                    # Process source information
                    if state["documents"]:
                        logging.info(f"Processing {len(state['documents'])} graded documents for source information")
                        for i, doc in enumerate(state["documents"]):
                            try:
                                # Extract source metadata
                                metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
                                
                                # Format source
                                source_info = {
                                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                                    "title": metadata.get("title", "Unknown Title"),
                                    "author": metadata.get("author", "Unknown Author"),
                                    "source_type": metadata.get("video_id") and "YouTube Video" or "Fitness Knowledge Base",
                                    "url": metadata.get("source", "")
                                }
                                
                                # Add to sources collection for summary
                                sources.append({
                                    "content": source_info["content"],
                                    "metadata": {
                                        "title": source_info["title"],
                                        "author": source_info["author"],
                                        "source": metadata.get("source", ""),
                                        "source_type": source_info["source_type"]
                                    }
                                })
                                
                                # Stream each source
                                yield f"data: {json.dumps({'type': 'source', 'content': source_info})}\n\n"
                            except Exception as e:
                                logging.error(f"Error processing source document {i}: {e}")
                    
                    # Check if we need web search
                    if state["web_search"]:
                        yield f"data: {json.dumps({'type': 'step', 'content': 'Some documents not relevant. Adding web search results...'})}\n\n"
                        state = web_search(state)
                        
                        # Add web search sources
                        if len(state["documents"]) > len(sources):
                            for i in range(len(sources), len(state["documents"])):
                                web_source = {
                                    "content": state["documents"][i].page_content[:200] + "..." if len(state["documents"][i].page_content) > 200 else state["documents"][i].page_content,
                                    "metadata": {"source": "web_search"}
                                }
                                sources.append(web_source)
                                yield f"data: {json.dumps({'type': 'source', 'content': web_source})}\n\n"
                
                elif route == WEBSEARCH:
                    # Use web search directly
                    yield f"data: {json.dumps({'type': 'step', 'content': 'Using web search to find information...'})}\n\n"
                    state = web_search(state)
                    
                    # Add web search sources
                    if state["documents"]:
                        for i, doc in enumerate(state["documents"]):
                            try:
                                web_source = {
                                    "content": doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content,
                                    "title": "Web Search Result",
                                    "author": "Web",
                                    "source_type": "Web Search",
                                    "url": ""
                                }
                                sources.append({"content": web_source["content"], "metadata": {"source": "web_search"}})
                                yield f"data: {json.dumps({'type': 'source', 'content': web_source})}\n\n"
                            except Exception as e:
                                logging.error(f"Error processing web search result: {e}")
                
                # Stream that we're generating the answer
                yield f"data: {json.dumps({'type': 'step', 'content': 'Generating response based on gathered information...'})}\n\n"
                
                # Log state before generation
                print(f"\n==== STATE BEFORE GENERATION ====")
                print(f"State: {state}")
                
                
                # Stream the generation with answer type
                async for chunk in generate_streaming(state):
                    if chunk:
                        answer_content += chunk
                        yield f"data: {json.dumps({'type': 'answer', 'content': chunk})}\n\n"
                
                # Store assistant response in conversation history
                if answer_content:
                    conversation_history.append({
                        "role": "assistant", 
                        "content": answer_content,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Persist to database
                    try:
                        store_result = store_conversation(user_id, thread_id, conversation_history)
                        logging.info(f"Stored conversation for thread {thread_id}: {store_result}")
                    except Exception as e:
                        logging.error(f"Error storing conversation: {e}")
                        logging.error(traceback.format_exc())
                
                # Complete source summary at end
                if sources:
                    yield f"data: {json.dumps({'type': 'sources_summary', 'content': sources})}\n\n"
                
                # Add thread_id and user_id metadata to response
                yield f"data: {json.dumps({'type': 'metadata', 'content': {'thread_id': thread_id, 'user_id': user_id}})}\n\n"
                
            except Exception as e:
                logging.error(f"Error in ask streaming: {str(e)}")
                logging.error(traceback.format_exc())
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            finally:
                yield "data: [DONE]\n\n"
        
        # Return as streaming response
        print(f"\n==== ASK QUESTION ====")
        print(f"User ID: {user_id}")
        print(f"Thread ID: {thread_id}")
        print(f"Question: {question.question}")
        return StreamingResponse(
            stream_langgraph_response(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logging.error(f"Error in ask endpoint: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@api.post("/workout/create")
async def create_workout(request: WorkoutNLQRequest):
    """Create a new workout based on natural language query"""
    try:
        # Initialize state with user profile
        state = initialize_workout_state(
            user_id=request.user_id,
            workout_prompt=request.prompt,
            thread_id=request.thread_id
        )
        
        # Run the graph
        result = workout_app.invoke(state)
        print("API returning result:", result)
        
        # Ensure reasoning is always present
        if "reasoning" not in result:
            result["reasoning"] = ""
        
        # Return created workouts
        return result
    except Exception as e:
        logging.error(f"Error creating workout: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
        
@api.post("/workout/variations")
async def generate_workout_variations(request: WorkoutVariationRequest):
    """Generate variations of an existing workout"""
    try:
        # Initialize state
        state = initialize_workout_state(
            user_id=request.user_id,
            workout_prompt="Generate variations of this workout",  # Default prompt
            workflow_type="variation",
            original_workout=request.original_workout,
            thread_id=request.thread_id
        )
        
        # Process through graph
        result = workout_app.invoke(state)
        
        # Return variations
        return {
            "status": "success",
            "variations": [w.model_dump() for w in result.get("variations", [])],
            "user_id": request.user_id,
            "thread_id": result.get("thread_id")
        }
    except Exception as e:
        logging.error(f"Error generating workout variations: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

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



if __name__ == "__main__":
    import uvicorn
    import signal
    import sys
    
    # Define shutdown handler
    def handle_shutdown(signum, frame):
        """Handle graceful shutdown"""
        print("\nShutting down server...")
        
        # No need to close PostgreSQL connections anymore
        # All connections are managed by Supabase client
        
        print("Shutdown complete.")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)  # Ctrl+C
    signal.signal(signal.SIGTERM, handle_shutdown)  # kill command
    
    # Get port from environment variable or use 8080 as default (Railway's expected port)
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting server on port {port}")
    
    # Run the server
    uvicorn.run(api, host="0.0.0.0", port=port)
