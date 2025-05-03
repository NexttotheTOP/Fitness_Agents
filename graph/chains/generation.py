from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os

# Standard LLM for non-streaming operations
llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Streaming LLM for streaming operations
streaming_llm = ChatOpenAI(
    temperature=0, 
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True
)

# Get the RAG prompt
prompt = hub.pull("rlm/rag-prompt")

# Regular chain for non-streaming operations
generation_chain = prompt | llm | StrOutputParser()

# Streaming chain - without output parser for streaming tokens
streaming_generation_chain = prompt | streaming_llm