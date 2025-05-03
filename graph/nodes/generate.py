from typing import Any, Dict, AsyncGenerator

from graph.chains.generation import generation_chain, streaming_generation_chain
from graph.state import GraphState
import asyncio


def generate(state: GraphState) -> Dict[str, Any]:
    """Standard generate function for non-streaming operations"""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


async def generate_streaming(state: GraphState) -> AsyncGenerator[str, None]:
    """Streaming version of generate that yields chunks"""
    print("---GENERATE STREAMING---")
    question = state["question"]
    documents = state["documents"]
    
    # Stream the generation
    async for chunk in streaming_generation_chain.astream({"context": documents, "question": question}):
        if hasattr(chunk, 'content') and chunk.content:
            yield chunk.content
        elif isinstance(chunk, str):
            yield chunk