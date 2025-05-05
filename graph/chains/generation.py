from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

# Get the base RAG prompt (we'll customize it)
base_prompt = hub.pull("rlm/rag-prompt")

# Create a custom prompt template that includes conversation history
conversation_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """
                You are a high-skilled, experienced and helpful fitness assistant working with your client (which is your friend) and that answers his questions.
                Your responses should be based on both the provided context and the conversation history.
                
                Guidelines:
                1. Always check the conversation history first to understand the context and previous interactions
                2. Use the provided context to supplement and validate your knowledge
                3. If you don't have enough information in either the context or conversation history, say "I don't know" or "I need more information to answer that"
                4. Be consistent with previous advice given in the conversation
                5. If new information contradicts previous advice, acknowledge this and explain the discrepancy
     
                So you are talking to your friend and you are helping him with his fitness questions, but make sure to always stay brutally honest 
                and do not assume just anything. If something is unclear, ask the user to clarify. 
                
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """Context:
{context}

Question: {question}

Answer:""")
])

# Regular chain for non-streaming operations
generation_chain = base_prompt | llm | StrOutputParser()

# Modified chain for when conversation history is available
conversation_chain = conversation_rag_prompt | llm | StrOutputParser()

# Streaming chains
streaming_generation_chain = base_prompt | streaming_llm
streaming_conversation_chain = conversation_rag_prompt | streaming_llm