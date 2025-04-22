from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()