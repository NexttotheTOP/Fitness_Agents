from langchain_core.tools import tool
from pydantic import BaseModel, Field

class RequestUserInputInput(BaseModel):
    message: str = Field(..., description="The question or prompt to ask the user.")

@tool("request_user_input", args_schema=RequestUserInputInput)
def request_user_input(message: str):
    """Ask the user for clarification or more information."""
    # This is just a stub for the LLM to call.
    # The orchestrator/frontend should handle the event and collect the user's answer.
    return {"type": "user_input_request", "content": message}
