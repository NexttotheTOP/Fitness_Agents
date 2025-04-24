from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from graph.workout_state import WorkoutState

class QueryAnalysis(BaseModel):
    should_run_dietary: bool = False
    should_run_fitness: bool = False
    should_run_general: bool = False
    confidence_scores: Dict[str, float] = {}
    reasoning: str = ""

class QueryAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4-1106-preview",
            temperature=0.1,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    def __call__(self, state: WorkoutState) -> WorkoutState:
        """Process the state and determine which agents should respond"""
        if not state["current_query"]:
            return state
            
        analysis = self.analyze(state["current_query"], state["user_profile"])
        state["query_analysis"] = analysis.dict()
        return state

    def analyze(self, query: str, user_profile: Dict[str, Any]) -> QueryAnalysis:
        """Analyze which agents should respond to the query"""
        
        prompt = f"""
        Analyze the following user query and determine which agents should respond.
        Consider the user's profile and the specific nature of the query.

        User Profile:
        {user_profile}

        Query: {query}

        Determine which agents should respond based on these criteria:
        1. Dietary Agent: For nutrition, meal planning, diet-related queries
        2. Fitness Agent: For exercise, workout, physical activity queries
        3. General Agent: For overall health, lifestyle, or combined queries

        Return a JSON object with:
        - should_run_dietary (boolean): If dietary agent should respond
        - should_run_fitness (boolean): If fitness agent should respond
        - should_run_general (boolean): If general agent should respond
        - confidence_scores (object): Confidence score (0-1) for each agent
        - reasoning (string): Brief explanation of the decision

        Example response:
        {
            "should_run_dietary": true,
            "should_run_fitness": false,
            "should_run_general": false,
            "confidence_scores": {
                "dietary": 0.9,
                "fitness": 0.2,
                "general": 0.3
            },
            "reasoning": "Query specifically asks about meal planning with no fitness components"
        }
        """

        response = self.llm.invoke(prompt)
        return QueryAnalysis.model_validate(response.content) 