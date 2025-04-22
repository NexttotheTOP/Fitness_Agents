from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

from graph.graph import app
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
api = FastAPI(title="Langgraph API")

# Add CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production to your specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a request model
class Question(BaseModel):
    question: str

@api.post("/ask")
async def ask_question(question: Question):
    result = app.invoke(input={"question": question.question})
    return {"response": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)
