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
        from graph.nodes.fitness_coach import DietaryAgent, FitnessAgent
        
        # Create direct instances of the agents
        dietary_agent = DietaryAgent()
        fitness_agent = FitnessAgent()
        
        # Generate dietary content directly (non-streaming)
        dietary_response = dietary_agent(state)
        dietary_content = dietary_response["dietary_state"].content
        
        # Generate fitness content directly (non-streaming)
        fitness_response = fitness_agent(state)
        fitness_content = fitness_response["fitness_state"].content
        
        # Update state with the generated content
        state["dietary_state"].content = dietary_content
        state["fitness_state"].content = fitness_content
        
        # Combine with markdown formatting

        complete_response = "## Dietary Plan\n\n"
        complete_response += dietary_content
        complete_response += "\n\n## Fitness Plan\n\n"
        complete_response += fitness_content
        complete_response += "\n\n---\n\n"
        complete_response += "Your personalized fitness and dietary plans have been created. You can now ask specific questions about your plans."
        
        return complete_response
    except Exception as e:
        logging.error(f"Error generating complete response: {str(e)}")
        return f"Error: {str(e)}"

@api.post("/fitness/profile")
async def create_fitness_profile(profile: FitnessProfileRequest):
    """Create a new fitness coaching session with user profile"""
    try:
        # Generate thread_id if not provided
        thread_id = profile.thread_id or str(uuid.uuid4())
        
        # Create initial state
        initial_state = get_initial_state(user_profile=profile.dict(exclude={"thread_id"}))
        initial_state["thread_id"] = thread_id
        
        # Store session state
        active_sessions[thread_id] = initial_state
        
        # Generate complete response instead of streaming
        complete_response = await generate_complete_response(initial_state)
        
        return {"thread_id": thread_id, "content": complete_response}
    
    except Exception as e:
        logging.error(f"Error creating fitness profile: {str(e)}")
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
            "rag_context_count": len(state.get("rag_context", [])) if "rag_context" in state else 0
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error retrieving session state for thread {thread_id}: {str(e)}")
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
