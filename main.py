from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
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
import socketio

load_dotenv()

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log') if os.environ.get("ENVIRONMENT") != "production" else logging.StreamHandler()
    ]
)

# Set specific logger levels
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("fastapi").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

from graph.graph import app as qa_app
from graph.workout_graph import app as workout_app, initialize_workout_state
from graph.workout_state import Workout, UserProfile, WorkoutState, AgentState, QueryType
from graph.fitness_coach_graph import get_initial_state, process_query, stream_response, app as fitness_coach_app
from fastapi.middleware.cors import CORSMiddleware
from graph.chains.workout_variation import analyze_workout
# Import shared state module
from graph.shared_state import thread_to_sid, register_sid, get_sid, register_socketio, emit_event

# Add RAG imports
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

# Add 3D model API imports
from graph.model_graph import model_graph
from graph.memory_store import get_previous_profile_overviews
from graph.nodes.workout_variation import generate_workout_variation
from langgraph.types import Command, Interrupt

# Create FastAPI app
api = FastAPI(title="Fitness Coach API")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fitness-friend.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Accept"],
)

# Error handling middleware
@api.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}", exc_info=True)
        return HTTPException(
            status_code=500,
            detail="Internal server error. Please try again later."
        )

# Request logging middleware
@api.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = datetime.now()
    logger.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"Response: {response.status_code} - {duration:.2f}s")
    
    return response

# --- Socket.IO Integration ---
# Create Socket.IO server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
# Wrap FastAPI app with Socket.IO ASGI app
app = socketio.ASGIApp(sio, api)

# Register the Socket.IO instance with shared_state
register_socketio(sio)
print("Registered Socket.IO instance with shared_state")

memory_saver = MemorySaver()

paused_workout_states = {}

@api.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Fitness Coach API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health commented out",
            "docs": "/docs",
            "fitness_profile": "/fitness/profile/stream",
            "fitness_query": "/fitness/query",
            "ask": "/ask",
            "workout_create": "/workout/create",
            "model_chat": "/model/chat/stream"
        }
    }

# Note: We're no longer defining thread_to_sid here - it's now in shared_state.py

# Helper function to extract final state from LangGraph results
def _extract_final_state(state_result, thread_id, user_id, messages):
    """Extract the final state from LangGraph state result, with fallback handling."""
    # Extract state from tuple if needed
    final_state = None
    if isinstance(state_result, tuple):
        for item in state_result:
            if isinstance(item, dict) and "thread_id" in item:
                final_state = item
                break
    else:
        final_state = state_result
    
    # Handle case where state extraction fails
    if not final_state or not isinstance(final_state, dict) or "thread_id" not in final_state:
        print(f"WARNING: Invalid state result type: {type(state_result)}")
        # Use initial state as fallback
        final_state = model_graph.get_or_create_state(thread_id, user_id)
        #final_state["messages"] = messages
    
    return final_state

# --- Socket.IO Event Handlers ---
@sio.event
def connect(sid, environ):
    logging.info(f"Socket.IO client connected: {sid}")

@sio.event
def disconnect(sid):
    logging.info(f"Socket.IO client disconnected: {sid}")

@sio.on('model_message')
async def handle_model_message(sid, data):
    """
    Handle incoming model messages from the frontend via Socket.IO.
    Streams assistant response token by token in real time.
    """
    try:
        message = data.get("message")
        thread_id = data.get("thread_id")
        user_id = data.get("user_id")
        
        print(f"\n==================== SOCKET.IO MODEL MESSAGE ====================")
        print(f"SID: {sid}")
        print(f"Message: {message}")
        print(f"Thread ID: {thread_id}")
        print(f"User ID: {user_id}")
        print(f"================================================================\n")
        
        if not message:
            await sio.emit('model_response', {"error": "Missing 'message' in data."}, to=sid)
            return

        # Get or create state
        state = model_graph.get_or_create_state(thread_id, user_id)
        
        # Update thread_id and user_id from state for consistency
        thread_id = state["thread_id"]
        user_id = state["user_id"]
        
        # Store the Socket.IO session ID with the thread for future use
        # Use shared state function instead of direct dictionary access
        register_sid(thread_id, sid)
        
        # Copy message history and add new message
        messages = state.get("messages", []).copy()
        messages.append({
            "role": "user", 
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Create input state with the new message
        input_state = state.copy()
        input_state["messages"] = messages
        
        # Configure checkpoint
        config = {
            "configurable": {
                "thread_id": state["thread_id"]
            }
        }
        
        # Define a writer function for streaming
        async def socket_io_writer(chunk):
            if isinstance(chunk, dict) and "type" in chunk and "content" in chunk:
                chunk_type = chunk["type"]
                
                if chunk_type == "response":
                    content = chunk["content"]
                    if content:  # Only emit non-empty tokens
                        await sio.emit('model_response', {
                            "response": content,
                            "thread_id": thread_id,
                            "user_id": user_id,
                            "stream": True
                        }, to=sid)
                        print(f"EMITTED RESPONSE TOKEN: '{content}'")
                
                elif chunk_type == "thinking":
                    await sio.emit('model_thinking', {"content": chunk["content"]}, to=sid)
                    print(f"EMITTED THINKING: {chunk['content']}")
                
                elif chunk_type == "event":
                    # Emit events immediately and add logging
                    print(f"[socket_io_writer] Emitting event: {chunk['content']['type']}")
                    await sio.emit('model_event', chunk["content"], to=sid)
                    print(f"[socket_io_writer] Successfully emitted event: {chunk['content']['type']}")
        
        # Register the writer with the global registry
        from graph.nodes.model_graph_nodes import register_writer, clear_writer
        # IMPORTANT: Clear any existing writer first to avoid stale references
        clear_writer(thread_id)
        register_writer(thread_id, socket_io_writer)
        print(f"Registered writer for thread {thread_id}")
        
        # Send an initial thinking message
        await sio.emit('model_thinking', {"content": "Analyzing your question..."}, to=sid)
        
        # Track events to send at the end
        pending_events = []
        
        # Process using LangGraph's native streaming
        print(f"[Socket.IO] Starting LangGraph stream")
        token_buffer = ""  # Buffer to collect tokens
        
        try:
            async for chunk_type, chunk in model_graph.graph.astream(
                input_state,
                stream_mode=["custom", "updates", "messages"],
                config=config
            ):
                if chunk_type == "custom" and isinstance(chunk, dict):
                    if chunk.get("type") == "response":
                        token = chunk.get("content", "")
                        if token:
                            token_buffer += token
                            await sio.emit('model_response', {
                                "response": token,
                                "thread_id": thread_id,
                                "user_id": user_id,
                                "stream": True
                            }, to=sid)
                            print(f"EMITTED RESPONSE TOKEN: '{token}'")
                    
                    elif chunk.get("type") == "thinking":
                        await sio.emit('model_thinking', {"content": chunk["content"]}, to=sid)
                        print(f"EMITTED THINKING: {chunk['content']}")
                    
                    elif chunk.get("type") == "event":
                        # Emit events immediately as well as collect them
                        await sio.emit('model_event', chunk["content"], to=sid)
                        print(f"EMITTED EVENT: {chunk['content']['type']}")
                        pending_events.append(chunk["content"])
                
                elif chunk_type == "updates" and isinstance(chunk, dict):
                    # Check for events field in updates
                    if "events" in chunk and isinstance(chunk["events"], list):
                        for event in chunk["events"]:
                            if isinstance(event, dict) and "type" in event:
                                await sio.emit('model_event', event, to=sid)
                                print(f"EMITTED EVENT FROM UPDATES: {event['type']}")
                                pending_events.append(event)
                elif chunk_type == "messages":
                    # chunk is a tuple (message_chunk, metadata)
                    message_chunk, _meta = chunk
                    token = None
                    if hasattr(message_chunk, "content"):
                        token = message_chunk.content
                    elif isinstance(message_chunk, str):
                        token = message_chunk
                    if token:
                        token_buffer += token
                        await sio.emit('model_response', {
                            "response": token,
                            "thread_id": thread_id,
                            "user_id": user_id,
                            "stream": True
                        }, to=sid)
                        print(f"EMITTED RESPONSE TOKEN: '{token}'")
        except Exception as e:
            print(f"Error during LangGraph streaming: {e}")
            import traceback
            traceback.print_exc()
            # Send error notification
            await sio.emit('model_error', {"error": str(e)}, to=sid)
        finally:
            # Always clean up the writer
            clear_writer(thread_id)
            print(f"Cleaned writer for thread {thread_id}")
        
        # Get final state
        state_result = await model_graph.graph.aget_state(config=config)
        
        # Extract state from tuple if needed
        final_state = None
        if isinstance(state_result, tuple):
            for item in state_result:
                if isinstance(item, dict) and "thread_id" in item:
                    final_state = item
                    break
        else:
            final_state = state_result
        
        # Handle case where state extraction fails
        if not final_state or not isinstance(final_state, dict) or "thread_id" not in final_state:
            print(f"WARNING: Invalid state result type: {type(state_result)}")
            # Use our initial state as fallback
            final_state = model_graph.get_or_create_state(thread_id, user_id)
            final_state["messages"] = messages
        
        # Update the model's active sessions with the final state
        model_graph.active_sessions[final_state["thread_id"]] = final_state
        
        # Save to memory saver
        try:
            import time
            memory_saver.put(
                final_state["thread_id"],
                {"state": final_state},
                {},
                [f"{final_state['thread_id']}_v{int(time.time())}"]
            )
            print(f"Persisted state to memory_saver")
        except Exception as e:
            print(f"Error persisting to memory_saver: {e}")
        
        # Extract the complete assistant response if available
        latest_messages = final_state.get("messages", [])
        final_content = ""
        for msg in reversed(latest_messages):
            if msg.get("role") == "assistant":
                final_content = msg.get("content", "")
                break
                
        # If we didn't get a final response from the messages but have a token buffer,
        # use that instead
        if not final_content and token_buffer:
            final_content = token_buffer
            print(f"Using token buffer for final content. Length: {len(token_buffer)}")
            
        # Send completion signal with final content
        await sio.emit('model_response', {
            "response": final_content,
            "thread_id": final_state["thread_id"],
            "user_id": final_state["user_id"],
            "done": True
        }, to=sid)
        
        print(f"Sent completion with content length: {len(final_content)}")
        
        # Send events summary if collected any
        if pending_events:
            await sio.emit('model_events_summary', {"count": len(pending_events)}, to=sid)
            print(f"EMITTED EVENTS SUMMARY: {len(pending_events)} events")
        
    except Exception as e:
        logging.error(f"Socket.IO model_message error: {e}")
        traceback.print_exc()
        await sio.emit('model_response', {"error": str(e)}, to=sid)

@sio.on('model_start')
async def handle_model_start(sid, data):
    """
    Notify the server that a message is coming, but response will be streamed via HTTP.
    This is used ONLY for bi-directional events related to the model, not for token streaming.
    """
    message = data.get("message")
    thread_id = data.get("thread_id")
    user_id = data.get("user_id")
    
    # Always ensure the thread_id is valid
    if not thread_id:
        thread_id = str(uuid.uuid4())
        print(f"Generated new thread_id: {thread_id}")
    
    # Store the Socket.IO session ID with the thread 
    register_sid(thread_id, sid)
    print(f"Registered Socket.IO session {sid} for thread {thread_id}")
    print(f"Active thread_to_sid mappings: {list(thread_to_sid.keys())}")
    
    print(f"Model start notification received via Socket.IO, response will stream via HTTP")
    print(f"SID: {sid}, Thread: {thread_id}, User: {user_id}")
    
    # No processing happens here - the HTTP stream endpoint will handle that
    await sio.emit('model_started', {
        "thread_id": thread_id,
        "status": "streaming"
    }, to=sid)
    
    # Also send a confirmation that we're ready to receive events
    await sio.emit('model_ready_for_events', {
        "thread_id": thread_id,
    }, to=sid)


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
    body_issues: str
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
    context: Optional[Dict[str, List[Dict[str, Any]]]] = None  # Add context field
    has_gym_access: Optional[bool] = False

class WorkoutVariationRequest(BaseModel):
    user_id: str
    original_workout: Dict[str, Any]
    thread_id: Optional[str] = None

# Store active sessions
active_sessions: Dict[str, Any] = {}

# 3D Model Chat Request/Response Models
class ModelChatRequest(BaseModel):
    """Request model for 3D model chat interactions."""
    message: str
    thread_id: Optional[str] = None
    user_id: Optional[str] = None

class ModelChatResponse(BaseModel):
    """Response model for 3D model chat interactions."""
    response: str
    events: List[Dict[str, Any]]
    thread_id: str
    user_id: str

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
                    state = retrieve(state)
                    
                    # Debug logging for documents
                    doc_count = len(state.get("documents", []))
                    logging.info(f"Retrieved {doc_count} documents from vector database")
                    
                    yield f"data: {json.dumps({'type': 'step', 'content': 'Retrieving documents from vector database...'})}\n\n"

                    subqueries = state.get("subqueries", [])
                    if subqueries:
                        if len(subqueries) == 1:
                            subq_msg = f"Used this query to search our knowledge base: \"{subqueries[0]}\""
                        else:
                            subq_msg = f"Used these {len(subqueries)} queries to search our knowledge base: " + ", ".join(f"\"{q}\"" for q in subqueries)
                        yield f"data: {json.dumps({'type': 'step', 'content': subq_msg})}\n\n"

                    pre_grade_docs = len(state.get("documents", []))
                    state = grade_documents(state)
                    yield f"data: {json.dumps({'type': 'step', 'content': 'Grading retrieved documents for relevance...'})}\n\n"
                    post_grade_docs = len(state.get("documents", []))
                    logging.info(f"Document grading: {pre_grade_docs} before, {post_grade_docs} after")
                    
                    # Log the full state after grading for debugging
                    logging.info(f"Full state after grading: {json.dumps(state, default=str)}")
                    
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
                                # Check if the document is from web search
                                if state["documents"][i].metadata.get("source_type") == "web_search":
                                    web_source = {
                                        "content": state["documents"][i].page_content[:250] + "..." if len(state["documents"][i].page_content) > 250 else state["documents"][i].page_content,
                                        "title": state["documents"][i].metadata.get("title", "Web Search Result"),
                                        "author": "Web",
                                        "source_type": "Web Search",
                                        "url": state["documents"][i].metadata.get("source", ""),
                                        "domain": state["documents"][i].metadata.get("domain", ""),
                                        "search_rank": i+1,
                                        "result_score": state["documents"][i].metadata.get("result_score", 1.0)
                                    }
                                    sources.append({
                                        "content": web_source["content"], 
                                        "metadata": {
                                            "title": web_source["title"],
                                            "author": web_source["author"],
                                            "source": web_source["url"],
                                            "source_type": web_source["source_type"],
                                            "domain": web_source["domain"],
                                            "search_rank": web_source["search_rank"],
                                            "result_score": web_source["result_score"]
                                        }
                                    })
                                    yield f"data: {json.dumps({'type': 'source', 'content': web_source})}\n\n"
                
                elif route == WEBSEARCH:
                    # Use web search directly
                    yield f"data: {json.dumps({'type': 'step', 'content': 'Using web search to find information...'})}\n\n"
                    state = web_search(state)
                    
                    # Add web search sources
                    if state["documents"]:
                        for i, doc in enumerate(state["documents"]):
                            try:
                                # Check if the document is from web search
                                if doc.metadata.get("source_type") == "web_search":
                                    web_source = {
                                        "content": doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content,
                                        "title": doc.metadata.get("title", "Web Search Result"),
                                        "author": "Web",
                                        "source_type": "Web Search",
                                        "url": doc.metadata.get("source", ""),
                                        "domain": doc.metadata.get("domain", ""),
                                        "search_rank": i+1,
                                        "result_score": doc.metadata.get("result_score", 1.0)
                                    }
                                    sources.append({
                                        "content": web_source["content"], 
                                        "metadata": {
                                            "title": web_source["title"],
                                            "author": web_source["author"],
                                            "source": web_source["url"],
                                            "source_type": web_source["source_type"],
                                            "domain": web_source["domain"],
                                            "search_rank": web_source["search_rank"],
                                            "result_score": web_source["result_score"]
                                        }
                                    })
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
                    sources_with_metadata = []
                    for source in sources:
                        if isinstance(source, dict) and "metadata" in source:
                            metadata = source["metadata"]
                            source_entry = {
                                "content": source["content"],
                                "title": metadata.get("title", "Unknown"),
                                "url": metadata.get("source", ""),
                                "source_type": metadata.get("source_type", "Unknown"),
                                "domain": metadata.get("domain", ""),
                                "rank": metadata.get("search_rank", 0) if metadata.get("source_type") == "web_search" else None
                            }
                            sources_with_metadata.append(source_entry)
                    
                    yield f"data: {json.dumps({'type': 'sources_summary', 'content': sources_with_metadata})}\n\n"
                
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
    

@api.post("/workout/feedback")
async def workout_feedback(request: Request):
    """
    Receives user feedback and resumes the workout graph.
    Expects JSON: { "thread_id": ..., "feedback": ... }
    """
    data = await request.json()
    thread_id = data.get("thread_id")
    feedback = data.get("feedback")
    print(f"THREAD ID feedback endpoint: {thread_id}")

    config = {
        "configurable": {
            "thread_id": thread_id,
            # "checkpoint_id": f"session_{thread_id}"
        }
    }
    print(f"FEEDBACK ===============================================:\n\n {feedback}")

    async def stream_feedback():
        try:
            async for chunk_type, chunk in workout_app.astream(
                Command(resume=feedback),
                stream_mode=["updates", "custom"],
                config=config
            ):
                # Handle Interrupt objects (pause for user feedback)
                if chunk_type == "custom":
                    if isinstance(chunk, dict) and "__interrupt__" in chunk:
                        interrupt_obj = chunk["__interrupt__"]
                        yield f"data: {json.dumps({'type': 'await_user_feedback', 'content': getattr(interrupt_obj, 'value', None)})}\n\n"
                        continue
                    if isinstance(chunk, dict):
                        if chunk.get("type") == "await_feedback":
                            paused_workout_states[thread_id] = chunk.copy()
                        yield f"data: {json.dumps(chunk)}\n\n:\n\n"
                        if chunk.get("type") == "progress":
                            print("➡️  emitting progress:", chunk["content"])
                    else:
                        try:
                            value, meta = chunk if isinstance(chunk, tuple) and len(chunk) == 2 else (chunk, {})
                            if isinstance(value, dict):
                                yield f"data: {json.dumps(value)}\n\n:\n\n"
                                if value.get("type") == "progress":
                                    print("➡️  emitting progress (node_stream):", value.get("content"))
                            else:
                                token = str(value)
                                if token:
                                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                        except Exception as _err:
                            print("Error handling node_stream chunk:", _err)
                elif chunk_type == "updates" and isinstance(chunk, dict):
                    if "__interrupt__" in chunk:
                        interrupt_obj = chunk["__interrupt__"]
                        yield f"data: {json.dumps({'type': 'await_user_feedback', 'content': getattr(interrupt_obj, 'value', None)})}\n\n"
                        continue
                    yield f"data: {json.dumps({'type': 'update', 'content': chunk})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logging.error(f"Error in feedback streaming: {str(e)}")
            logging.error(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        stream_feedback(),
        media_type="text/event-stream"
    )

@api.post("/workout/create")
async def create_workout(request: WorkoutNLQRequest):
    """Create a new workout based on natural language query"""
    try:
        # Initialize state with user profile
        state = initialize_workout_state(
            user_id=request.user_id,
            workout_prompt=request.prompt,
            thread_id=request.thread_id,
            context=request.context,  # Pass context to initialization
            has_gym_access=request.has_gym_access
        )
        print(f"thread ID create workout endpoint from state: {state['thread_id']}")
        print(f"thread ID create workout endpoint from request: {request.thread_id}")

        config = {
            "configurable": {
                "thread_id": state["thread_id"],
                #"checkpoint_id": f"session_{state['thread_id']}"
            }
        }
        
        async def stream_workout():
            try:
                async for chunk_type, chunk in workout_app.astream(
                    state,
                    # We no longer want to forward raw LLM tokens – omit "messages" stream mode
                    stream_mode=["updates", "custom"],   
                    config=config
                ):
                    # Handle Interrupt objects (pause for user feedback)
                    if chunk_type == "custom":
                        # Interrupts are yielded as dicts with '__interrupt__' key
                        if isinstance(chunk, dict) and "__interrupt__" in chunk:
                            interrupt_obj = chunk["__interrupt__"]
                            yield f"data: {json.dumps({'type': 'await_user_feedback', 'content': getattr(interrupt_obj, 'value', None)})}\n\n"
                            continue
                        # Forward any custom or node_stream chunks (dicts) directly to the client
                        if isinstance(chunk, dict):
                            # # Store state if awaiting feedback
                            if chunk.get("type") == "await_feedback":
                                paused_workout_states[state["thread_id"]] = state.copy()
                            yield f"data: {json.dumps(chunk)}\n\n:\n\n"  # colon line forces flush
                            if chunk.get("type") == "progress":
                                print("➡️  emitting progress:", chunk["content"])
                        else:
                            # node_stream yields are tuples: (value, metadata)
                            try:
                                value, meta = chunk if isinstance(chunk, tuple) and len(chunk) == 2 else (chunk, {})

                                # If the yielded value is a dict with a `type`, forward as-is (e.g., progress)
                                if isinstance(value, dict):
                                    yield f"data: {json.dumps(value)}\n\n:\n\n"
                                    if value.get("type") == "progress":
                                        print("➡️  emitting progress (node_stream):", value.get("content"))
                                else:
                                    # Otherwise treat as plain token
                                    token = str(value)
                                    if token:
                                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                            except Exception as _err:
                                print("Error handling node_stream chunk:", _err)
                    elif chunk_type == "updates" and isinstance(chunk, dict):
                        # Check for Interrupt in updates as well
                        if "__interrupt__" in chunk:
                            interrupt_obj = chunk["__interrupt__"]
                            yield f"data: {json.dumps({'type': 'await_user_feedback', 'content': getattr(interrupt_obj, 'value', None)})}\n\n"
                            continue
                        # Safe to serialize
                        yield f"data: {json.dumps({'type': 'update', 'content': chunk})}\n\n"
                # Send completion signal
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            except Exception as e:
                logging.error(f"Error in workout streaming: {str(e)}")
                logging.error(traceback.format_exc())
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        return StreamingResponse(
            stream_workout(),
            media_type="text/event-stream"
        )
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
            workout_prompt="Based on my profile, generate 3-5 variations of this workout that are tailored to my profile.", 
            workflow_type="variation",
            original_workout=request.original_workout,
            thread_id=request.thread_id
        )
        
        # Process through graph
        result = generate_workout_variation(state)
        
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
    



@api.post("/fitness/profile/stream")
async def stream_fitness_profile(profile: FitnessProfileRequest):
    """Create a new fitness coaching session with user profile with real-time streaming"""
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
        
        # Define the streaming response function
        async def generate_sse_stream():
            # Process body analysis and stream it if photos are available
            if has_body_photos:
                # Use the ProfileAgent for body analysis
                from graph.nodes.fitness_coach import ProfileAgent
                profile_agent = ProfileAgent()
                
                try:
                    # Stream body analysis tokens directly as they're generated
                    body_analysis = ""
                    token_count = 0
                    
                    # Log that we're starting the body analysis
                    logging.info(f"Starting body analysis streaming for user {user_id}")
                    
                    # Stream directly from the body analysis generator
                    async for token in profile_agent._analyze_body_composition(initial_state["user_profile"]):
                        if token:
                            body_analysis += token
                            token_count += 1
                            
                            # Stream each token immediately to the client
                            yield f"data: {json.dumps({'type': 'content', 'content': token})}\n\n"
                            
                            # Force flush for better real-time streaming
                            if token_count % 5 == 0:  # Every 5 tokens
                                yield f":\n\n"  # Empty comment forces flush
                    
                    logging.info(f"Body analysis streaming complete: {token_count} tokens, {len(body_analysis)} chars")
                    
                    # Store the complete analysis in state for later use
                    if body_analysis and len(body_analysis) > 50:
                        initial_state["body_analysis"] = body_analysis
                        
                        # Store image timestamps if available
                        if "image_timestamps" in initial_state["user_profile"]:
                            initial_state["image_timestamps"] = initial_state["user_profile"]["image_timestamps"]
                except Exception as e:
                    logging.error(f"Error during body analysis: {str(e)}")
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    yield f"data: {json.dumps({'type': 'error', 'content': f'Error analyzing body photos: {str(e)}'})}\n\n"
            
            # Stream the rest of the response
            try:
                # Use LangGraph's native streaming capabilities through our stream_response function
                logging.info(f"Starting LangGraph stream_response for user {user_id}")
                
                # --- SIMPLIFIED: Just yield from stream_response, which now handles progress ---
                async for chunk in stream_response(initial_state):
                    if chunk:
                        yield chunk
                # Send completion signal
                yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            except Exception as e:
                logging.error(f"Error streaming profile: {str(e)}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                # Send error message
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            finally:
                # Signal end of stream
                yield "data: [DONE]\n\n"
                
                # Store the completed session in active_sessions
                try:
                    # Get final state
                    config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_id": f"session_{thread_id}"
                        }
                    }
                    final_state = await fitness_coach_app.aget_state(config=config)
                    
                    if isinstance(final_state, dict) and "user_id" in final_state:
                        # Store in active sessions for future queries
                        active_sessions[thread_id] = final_state
                        logging.info(f"Stored completed session in active_sessions for thread {thread_id}")
                except Exception as e:
                    logging.error(f"Error retrieving final state: {str(e)}")
        
        # Return streaming response
        logging.info(f"Starting streaming response for fitness profile: user_id={user_id}, thread_id={thread_id}")
        return StreamingResponse(
            generate_sse_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Transfer-Encoding": "chunked"
            }
        )
    
    except Exception as e:
        logging.error(f"Error in stream_fitness_profile: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@api.post("/fitness/query")
async def process_fitness_query(query: FitnessQueryRequest):
    """Process a query in the fitness coaching session with streaming response"""
    try:
        thread_id = query.thread_id
        user_id = query.user_id

        # Helper to load session from persistent storage
        def load_session_from_db(thread_id, user_id):
            # Try to get the most recent overview for this user and thread
            overviews = get_previous_profile_overviews(user_id, limit=10)
            for overview in overviews:
                print(f"Overview: {overview}")
                if overview.get("id") == thread_id:
                    # Reconstruct a minimal session state from the overview
                    # You may want to expand this to include more fields as needed
                    state = {
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "user_profile": {},
                        "dietary_state": {"content": "", "last_update": "", "is_streaming": False},
                        "fitness_state": {"content": "", "last_update": "", "is_streaming": False},
                        "current_query": query.query,
                        "query_type": "GENERAL",
                        "conversation_history": [],
                        "original_workout": None,
                        "variations": [],
                        "analysis": {},
                        "generation": None,
                        "body_analysis": None,
                        "complete_response": overview.get("content", ""),
                        "previous_complete_response": None,
                        "previous_sections": None
                    }
                    print(f"Loaded session from DB: {state}")
                    return state
                    
            return None

        if thread_id not in active_sessions:
            # Try to load from persistent storage
            session_state = load_session_from_db(thread_id, user_id)
            if session_state is None:
                raise HTTPException(status_code=404, detail="Session not found")
            active_sessions[thread_id] = session_state

        # Get current session state
        current_state = active_sessions[thread_id]

        # Ensure user_id matches
        if current_state.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="User ID does not match session")

        # If no history updated, return streaming response as fallback
        logging.info(f"Thread {thread_id} - User {user_id} - complete_response length: {len(current_state.get('complete_response', '') or '')}")
        logging.info(f"Thread {thread_id} - User {user_id} - complete_response preview: {repr((current_state.get('complete_response', '') or '')[:500])}")
        return StreamingResponse(
            process_query(current_state, query.query),
            media_type="text/event-stream"
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error processing query for thread {thread_id}: {str(e)}")
        logging.error(f"Query: {query.query}")
        raise HTTPException(status_code=500, detail=str(e))




@api.get("/model/token-stream")
async def model_token_stream(
    message: str,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    sid: Optional[str] = None  # Allow passing Socket.IO SID directly
):
    """Stream ONLY the LLM tokens to the client using HTTP streaming."""
    
    async def stream_tokens():
        # Get or create state - Don't use nonlocal as these are function parameters
        state = model_graph.get_or_create_state(thread_id, user_id)
        thread_id_to_use = state["thread_id"]
        user_id_to_use = state["user_id"]
        
        # CRITICAL: Ensure this thread ID is registered in thread_to_sid
        # If we don't have a Socket.IO session yet, create a mapping using thread_id as SID
        # This ensures tool_executor_node can always find a way to emit events
        if not get_sid(thread_id_to_use):
            # If sid was provided explicitly, use it
            if sid:
                register_sid(thread_id_to_use, sid)
                print(f"[HTTP] Registered provided Socket.IO SID {sid} for thread {thread_id_to_use}")
            else:
                # Otherwise, use thread_id as a fallback SID for broadcasting
                register_sid(thread_id_to_use, thread_id_to_use)
                print(f"[HTTP] No Socket.IO SID available, using thread_id as SID: {thread_id_to_use}")
                
        print(f"[HTTP] Active thread_to_sid mappings: {list(thread_to_sid.keys())}")
        
        # Send metadata
        yield f"event: metadata\ndata: {json.dumps({'thread_id': thread_id_to_use, 'user_id': user_id_to_use})}\n\n"
        yield f"event: thinking\ndata: {json.dumps({'content': 'Analyzing your question...'})}\n\n"
        
        # Check if we have a Socket.IO session for this thread
        sid = get_sid(thread_id_to_use)
        if sid:
            print(f"[HTTP] Found Socket.IO session {sid} for thread {thread_id_to_use}")
        
        # Add the new message to state
        messages = state.get("messages", []).copy()
        messages.append({
            "role": "user", 
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Create input state with the new message
        input_state = state.copy()
        input_state["messages"] = messages
        
        # CRITICAL: Set flag to inform tool_executor_node to use Socket.IO for events
        # This enables parallel channel event sending while keeping SSE for tokens
        input_state["_token_only_stream"] = True
        input_state["_sse_sid"] = sid  # Pass the Socket.IO session ID if available
        
        # Configure checkpoint
        config = {
            "configurable": {
                "thread_id": thread_id_to_use
            }
        }
        
        # Tracking
        token_count = 0
        token_content = ""
        
        # Stream directly from LangGraph
        print(f"[HTTP] Starting direct token streaming from LangGraph")
        try:
            # Include "node_stream" so we also get direct generator yields from nodes like responder_node
            async for chunk_type, chunk in model_graph.graph.astream(
                input_state,
                stream_mode=["custom", "updates", "messages"],
                config=config
            ):
                
                # Custom yields (e.g. events coming via StreamWriter)
                if chunk_type == "custom" and isinstance(chunk, dict):
                    if chunk.get("type") == "response" and "content" in chunk:
                        token = chunk["content"]
                        token_count += 1
                        token_content += token
                        print(f"[HTTP] Custom stream token #{token_count}: '{token}'")
                        yield f"event: token\ndata: {json.dumps({'content': token})}\n\n"
                        yield ":\n\n"  # Force flush
                    
                    elif chunk.get("type") == "thinking":
                        yield f"event: thinking\ndata: {json.dumps({'content': chunk.get('content', '')})}\n\n"
                    
                    elif chunk.get("type") == "event":
                        # Directly stream the event - will also be sent via Socket.IO in tool_executor_node
                        event_content = chunk.get("content", {})
                        yield f"event: event\ndata: {json.dumps({'content': event_content})}\n\n"
                        
                        # Also emit via Socket.IO if we have a session ID
                        if sid:
                            try:
                                await sio.emit('model_event', event_content, to=sid)
                                print(f"[HTTP] Also emitted event via Socket.IO: {event_content.get('type', 'unknown')}")
                            except Exception as e:
                                print(f"[HTTP] Error sending event via Socket.IO: {e}")
                
                # Direct node stream yields (e.g. responder_node token generator)
                elif chunk_type == "messages":
                    # chunk is a tuple (message_chunk, metadata)
                    message_chunk, metadata = chunk
                    token = None
                    if hasattr(message_chunk, "content"):
                        token = message_chunk.content
                    elif isinstance(message_chunk, str):
                        token = message_chunk
                    # FILTER: Only yield if from responder node
                    if metadata:
                        #print(f"[HTTP] Message metadata: {json.dumps(metadata)}")
                        if metadata.get("langgraph_node") != "responder":
                            continue
                    if token:
                        token_count += 1
                        token_content += token
                        yield f"event: token\ndata: {json.dumps({'content': token})}\n\n"
                        yield ":\n\n"
                        await asyncio.sleep(0)
                
                # Updates/model events batched in state
                elif chunk_type == "updates" and isinstance(chunk, dict) and "events" in chunk:
                    events = chunk.get("events", [])
                    if events:
                        # Emit each event individually
                        for event in events:
                            if isinstance(event, dict) and "type" in event:
                                yield f"event: event\ndata: {json.dumps({'content': event})}\n\n"
                                # Also emit via Socket.IO if we have a session ID
                                if sid:
                                    try:
                                        await sio.emit('model_event', event, to=sid)
                                        print(f"[HTTP] Also emitted event via Socket.IO from updates: {event.get('type', 'unknown')}")
                                    except Exception as e:
                                        print(f"[HTTP] Error sending event via Socket.IO: {e}")
                        
                        # Also emit summary
                        yield f"event: events_batch\ndata: {json.dumps({'count': len(events)})}\n\n"
        
            # Get final content
            state_result = await model_graph.graph.aget_state(config=config)
            final_state = _extract_final_state(state_result, thread_id_to_use, user_id_to_use, messages)
            model_graph.active_sessions[final_state["thread_id"]] = final_state
            
            # Get final message
            latest_messages = final_state.get("messages", [])
            final_response = ""
            if latest_messages and latest_messages[-1]["role"] == "assistant":
                final_response = latest_messages[-1].get("content", "")
            
            print(f"[HTTP] Streaming complete! Sent {token_count} tokens")
            
            # Completion events
            yield f"event: complete\ndata: {json.dumps({'content': final_response})}\n\n"
            yield "event: done\ndata: {}\n\n"
            
        except Exception as e:
            print(f"Error during streaming: {e}")
            traceback.print_exc()
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        stream_tokens(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked"
        }
    )

@api.get("/model/test-token-stream")
async def test_token_stream():
    async def stream_test():
        for i in range(30):
            yield f"event: token\ndata: {json.dumps({'content': f'Test token {i}'})}\n\n"
            yield ":\n\n"  # Force flush
            await asyncio.sleep(0.5)  # Simulate realistic token timing
            
    return StreamingResponse(
        stream_test(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@api.websocket("/model/ws")
async def model_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time 3D model chat interactions."""
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if "message" not in message_data:
                await websocket.send_json({"error": "Message field is required"})
                continue
            
            # Get or create state for this session
            state = model_graph.get_or_create_state(
                message_data.get("thread_id"), 
                message_data.get("user_id")
            )
            thread_id = state["thread_id"]
            
            # Register websocket SID (using websocket.client.host as SID)
            client_id = str(id(websocket))
            register_sid(thread_id, client_id)
            print(f"[WebSocket] Registered client {client_id} for thread {thread_id}")
            
            # Define a writer function for the WebSocket
            async def websocket_writer(chunk):
                if isinstance(chunk, dict) and "type" in chunk and "content" in chunk:
                    chunk_type = chunk["type"]
                    
                    if chunk_type == "response":
                        await websocket.send_json({
                            "type": "response",
                            "content": chunk["content"]
                        })
                        print(f"WS: EMITTED RESPONSE TOKEN")
                    
                    elif chunk_type == "thinking":
                        await websocket.send_json({
                            "type": "thinking",
                            "content": chunk["content"]
                        })
                        print(f"WS: EMITTED THINKING")
                    
                    elif chunk_type == "event":
                        await websocket.send_json({
                            "type": "event",
                            "content": chunk["content"]
                        })
                        print(f"WS: EMITTED EVENT: {chunk['content']['type']}")
            
            # Register the writer with the global registry
            from graph.nodes.model_graph_nodes import register_writer, clear_writer
            register_writer(thread_id, websocket_writer)
            
            # Add the new message to the messages history
            messages = state.get("messages", []).copy()
            messages.append({
                "role": "user", 
                "content": message_data["message"],
                "timestamp": datetime.now().isoformat()
            })
            
            # Create input state with the new message
            input_state = state.copy()
            input_state["messages"] = messages
            
            # Configure checkpoint
            config = {
                "configurable": {
                    "thread_id": state["thread_id"]
                }
            }
            
            # Track events to send at the end
            pending_events = []
            
            # Stream directly from LangGraph
            print(f"[WebSocket] Starting LangGraph stream")
            async for chunk_type, chunk in model_graph.graph.astream(
                input_state,
                stream_mode=["custom", "updates"],
                config=config
            ):
                if chunk_type == "custom" and isinstance(chunk, dict):
                    if chunk.get("type") == "response":
                        await websocket.send_json({
                            "type": "response",
                            "content": chunk["content"]
                        })
                        print(f"WS: EMITTED RESPONSE TOKEN")
                    
                    elif chunk.get("type") == "thinking":
                        await websocket.send_json({
                            "type": "thinking",
                            "content": chunk["content"]
                        })
                        print(f"WS: EMITTED THINKING")
                    
                    elif chunk.get("type") == "event":
                        # Collect events to send individually
                        await websocket.send_json({
                            "type": "event",
                            "content": chunk["content"]
                        })
                        print(f"WS: EMITTED EVENT: {chunk['content']['type']}")
                        # Also collect for summary
                        pending_events.append(chunk["content"])
                
                elif chunk_type == "updates" and isinstance(chunk, dict) and "events" in chunk:
                    # If we received events in updates, process them
                    for event in chunk.get("events", []):
                        await websocket.send_json({
                            "type": "event",
                            "content": event
                        })
                        print(f"WS: EMITTED EVENT FROM UPDATES: {event['type']}")
                        pending_events.append(event)
            
            # Clean up the writer from the registry
            clear_writer(thread_id)
            
            # Get final state
            state_result = await model_graph.graph.aget_state(config=config)
            
            # Extract state from tuple if needed
            final_state = None
            if isinstance(state_result, tuple):
                for item in state_result:
                    if isinstance(item, dict) and "thread_id" in item:
                        final_state = item
                        break
            else:
                final_state = state_result
            
            # Handle case where state extraction fails
            if not final_state or not isinstance(final_state, dict) or "thread_id" not in final_state:
                print(f"WARNING: Invalid state result type: {type(state_result)}")
                # Use our initial state as fallback
                final_state = model_graph.get_or_create_state(
                    message_data.get("thread_id", state["thread_id"]),
                    message_data.get("user_id", state["user_id"])
                )
                final_state["messages"] = messages
            
            # Update the model's active sessions with the final state
            model_graph.active_sessions[final_state["thread_id"]] = final_state
            
            # Save to memory saver
            try:
                import time
                memory_saver.put(
                    final_state["thread_id"],
                    {"state": final_state},
                    {},
                    [f"{final_state['thread_id']}_v{int(time.time())}"]
                )
                print(f"WS: Persisted state to memory_saver with {len(final_state.get('messages', []))} messages")
            except Exception as e:
                print(f"WS: Error persisting to memory_saver: {e}")
            
            # Send events summary if needed
            if pending_events:
                await websocket.send_json({
                    "type": "events_summary",
                    "content": {"count": len(pending_events)}
                })
            
            # Send metadata and completion message
            await websocket.send_json({
                "type": "metadata",
                "content": {
                    "thread_id": final_state["thread_id"],
                    "user_id": final_state["user_id"]
                }
            })
            
            # Send completion message
            await websocket.send_json({"type": "done"})
    
    except WebSocketDisconnect:
        # Client disconnected
        pass
    except Exception as e:
        # Handle other errors
        logging.error(f"Error in model websocket: {str(e)}")
        traceback.print_exc()
        if websocket.client_state.CONNECTED:
            await websocket.send_json({"error": str(e)})

@api.get("/model/reset")
async def model_reset_state(thread_id: Optional[str] = None, user_id: Optional[str] = None):
    """Reset the 3D model state to default values."""
    new_state = model_graph.reset(thread_id, user_id)
    return {
        "status": "3D model state reset successfully", 
        "thread_id": new_state["thread_id"],
        "user_id": new_state["user_id"]
    }

@api.get("/model/state/{thread_id}")
async def model_get_state(thread_id: str):
    """Get the current state of the 3D model for a specific thread."""
    state = model_graph.get_state(thread_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"State not found for thread ID: {thread_id}")
    return state

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
    uvicorn.run(app, host="0.0.0.0", port=port)
