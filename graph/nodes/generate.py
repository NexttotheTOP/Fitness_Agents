from typing import Any, Dict, AsyncGenerator, AsyncIterable
from langchain_core.messages import HumanMessage, AIMessage
from graph.memory_store import get_most_recent_profile_overview
from graph.chains.generation import generation_chain, streaming_generation_chain, streaming_conversation_chain, conversation_chain
from graph.state import GraphState
import asyncio


def generate(state: GraphState) -> Dict[str, Any]:
    """Standard generate function for non-streaming operations"""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    user_id = state["user_id"]
    conversation_history = state.get("conversation_history", [])
    profile_overview_data = get_most_recent_profile_overview(user_id)
    profile_overview = profile_overview_data["content"] if profile_overview_data and "content" in profile_overview_data else ""
    print(f"============================================================================    Profile overview: {profile_overview}")
    # Convert conversation history to LangChain message format
    chat_history = []
    if conversation_history:
        # Include ALL messages including the current question
        for msg in conversation_history:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))

    generation = conversation_chain.invoke({
        "context": format_docs(documents), 
        "question": question,
        "chat_history": chat_history,
        "user_profile": profile_overview
    })
    return {"documents": documents, "question": question, "generation": generation}


def format_docs(docs):
    """Format documents into a single string with source info."""
    formatted_docs = []
    for i, doc in enumerate(docs):
        source_info = ""
        if hasattr(doc, "metadata"):
            if doc.metadata.get("source_type") == "web_search":
                source_info = f" [Source: {doc.metadata.get('title')} ({doc.metadata.get('domain')})]"
            else:
                source_info = f" [Source: {doc.metadata.get('title', 'Unknown')}]"
        
        formatted_docs.append(f"Document {i+1}{source_info}:\n{doc.page_content}")
    
    return "\n\n".join(formatted_docs)


async def generate_streaming(state: GraphState) -> AsyncIterable[str]:
    """Generate a response based on documents, streaming tokens as they are generated."""
    print("---GENERATE STREAMING WITH CONVERSATION HISTORY---")
    question = state["question"]
    documents = state["documents"]
    conversation_history = state.get("conversation_history", [])
    user_id = state["user_id"]
    profile_overview_data = get_most_recent_profile_overview(user_id)
    profile_overview = profile_overview_data["content"] if profile_overview_data and "content" in profile_overview_data else ""
    
    # Better logging for debugging
    print(f"Conversation history has {len(conversation_history)} messages")
    if conversation_history:
        print(f"First msg: {conversation_history[0]['role']} - {conversation_history[0]['content'][:30]}...")
    
    # Convert conversation history to LangChain message format
    chat_history = []
    if conversation_history:  # Use any history we have - include all messages
        # Include ALL previous messages INCLUDING the current question
        for i, msg in enumerate(conversation_history):
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
                print(f"Added user message {i}: {msg['content'][:30]}...")
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))
                print(f"Added assistant message {i}: {msg['content'][:30]}...")
        
        if chat_history:  # Only if we actually added messages
            print(f"Using conversation history with {len(chat_history)} messages")
            
            # Use the conversation chain with history
            async for chunk in streaming_conversation_chain.astream({
                "context": format_docs(documents), 
                "question": question,
                "chat_history": chat_history,
                "user_profile": profile_overview
            }):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                elif isinstance(chunk, str):
                    yield chunk
            return  # Important: return after using the conversation chain
    
    # If we get here, we're using the standard chain without history
    print("No usable conversation history, using standard chain")
    async for chunk in streaming_conversation_chain.astream({
        "context": format_docs(documents), 
        "question": question,
        "chat_history": [],
        "user_profile": profile_overview
    }):
        if hasattr(chunk, 'content') and chunk.content:
            yield chunk.content
        elif isinstance(chunk, str):
            yield chunk