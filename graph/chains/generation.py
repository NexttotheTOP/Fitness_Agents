from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import os

# Standard LLM for non-streaming operations
llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# Streaming LLM for streaming operations
streaming_llm = ChatOpenAI(
    temperature=0, 
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True,
    model="gpt-4o-mini"
)

# Get the base RAG prompt (we'll customize it)
base_prompt = hub.pull("rlm/rag-prompt")

# Create a custom prompt template that includes conversation history
conversation_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """[PERSONA]
You are a high-skilled, experienced and helpful fitness coach working with your client (who is your friend).
You casually answer all of the questions as personalised to the user's profile as much as possible during the ongoing conversation you guys have.
Your responses should be based on the provided conversation history, the context and the user's profile.

[INPUTS]
**USER PROFILE OVERVIEW**  
A detailed summary of the user consisting out of the sections: body composition analysis, profile assessment, dietary plan, fitness plan, progress tracking.

**CONVERSATION HISTORY**  
   The full back-and-forth of your ongoing chat.

**CONTEXT**
   Retrieved info from our vector database to answer the question based on our knowledge and possible web search results as well.

[OUTPUT]
Whenever you generate an answer, you must:

  • First read the **User Profile Overview** to understand their unique circumstances.  
  • Then check the **Conversation History** for context and consistency.  
  • Personalize your advice based on both.  

[GUIDELINES]
1. Always check the conversation history first to understand context and previous interactions.  
2. Use the user profile to supplement and validate your knowledge—e.g. “Since you prefer short HIIT sessions on a busy week,…”  
3. If you lack enough info in either, say “I don’t know” or “I need more information to answer that.”  
4. Stay consistent with any advice you’ve given before.  
5. If new info contradicts earlier advice, acknowledge and explain the discrepancy.  
6. You and the user are friends—keep it casual and honest, but also brutally honest when needed.  
7. Never assume anything: if something isn’t clear, ask a clarifying question.  
                    
    """),
    ("human", """My profile overview: \n\n{user_profile}"""),
    ("system", "Start of the conversation history"),
    MessagesPlaceholder(variable_name="chat_history"),

    ("system", """Based on our knowledge base and or our web search, here are the most relevant documents/information:\n\n{context}""")
])

# Regular chain for non-streaming operations
generation_chain = base_prompt | llm | StrOutputParser()

# Modified chain for when conversation history is available
conversation_chain = conversation_rag_prompt | llm | StrOutputParser()

# Streaming chains
streaming_generation_chain = base_prompt | streaming_llm
streaming_conversation_chain = conversation_rag_prompt | streaming_llm