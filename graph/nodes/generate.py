from typing import Any, Dict, AsyncGenerator, AsyncIterable
from langchain_core.messages import HumanMessage, AIMessage

from graph.chains.generation import generation_chain, streaming_generation_chain, streaming_conversation_chain
from graph.state import GraphState
import asyncio


def generate(state: GraphState) -> Dict[str, Any]:
    """Standard generate function for non-streaming operations"""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def format_docs(docs):
    """Format documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


async def generate_streaming(state: GraphState) -> AsyncIterable[str]:
    """Generate a response based on documents, streaming tokens as they are generated."""
    print("---GENERATE STREAMING WITH CONVERSATION HISTORY---")
    question = state["question"]
    documents = state["documents"]
    conversation_history = state.get("conversation_history", [])
    
    # Better logging for debugging
    print(f"Conversation history has {len(conversation_history)} messages")
    if conversation_history:
        print(f"First msg: {conversation_history[0]['role']} - {conversation_history[0]['content'][:30]}...")
    
    # Convert conversation history to LangChain message format
    chat_history = []
    if conversation_history:  # Remove the length check - we want to use any history we have
        # Include previous messages but skip the current question
        for i, msg in enumerate(conversation_history[:-1]):
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
                "chat_history": chat_history
            }):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                elif isinstance(chunk, str):
                    yield chunk
            return  # Important: return after using the conversation chain
    
    # If we get here, we're using the standard chain without history
    print("No usable conversation history, using standard chain")
    async for chunk in streaming_generation_chain.astream({
        "context": format_docs(documents), 
        "question": question
    }):
        if hasattr(chunk, 'content') and chunk.content:
            yield chunk.content
        elif isinstance(chunk, str):
            yield chunk