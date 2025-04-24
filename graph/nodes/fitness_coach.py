from typing import Dict, Any, AsyncGenerator
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from graph.workout_state import WorkoutState, UserProfile, AgentState, QueryType
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class ProfileAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4-1106-preview",
            temperature=0.3,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    
    async def stream(self, state: WorkoutState) -> AsyncGenerator[str, None]:
        """Stream the profile analysis"""
        if not state["user_profile"]:
            state["user_profile"] = UserProfile().dict()
        
        profile_prompt = f"""
        Analyze the user profile information and ensure it's complete for creating 
        personalized health and fitness plans. Current profile: {state["user_profile"]}
        
        Validate or request additional information such as:
        1. Age
        2. Gender
        3. Height
        4. Weight
        5. Activity Level
        6. Fitness Goals
        7. Dietary Preferences
        8. Any health restrictions or medical conditions
        
        Provide a comprehensive profile assessment.
        """
        
        async for chunk in self.llm.astream(profile_prompt):
            yield chunk.content

    def __call__(self, state: WorkoutState) -> WorkoutState:
        """Update or validate the user profile"""
        if not state["user_profile"]:
            state["user_profile"] = UserProfile().dict()
        
        profile_prompt = f"""
        Analyze the user profile information and ensure it's complete for creating 
        personalized health and fitness plans. Current profile: {state["user_profile"]}
        
        Validate or request additional information such as:
        1. Age
        2. Gender
        3. Height
        4. Weight
        5. Activity Level
        6. Fitness Goals
        7. Dietary Preferences
        8. Any health restrictions or medical conditions
        
        Provide a comprehensive profile assessment.
        """
        
        response = self.llm.invoke(profile_prompt)
        state["user_profile"] = response.content
        return state

class DietaryAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4-1106-preview",
            temperature=0.4,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    
    async def stream(self, state: WorkoutState) -> AsyncGenerator[str, None]:
        """Stream dietary recommendations"""
        diet_prompt = f"""
        Create a comprehensive dietary plan based on the following user profile:
        {state["user_profile"]}
        
        Generate a detailed meal plan including:
        - Breakfast options
        - Lunch options
        - Dinner options
        - Healthy snacks
        
        Key Considerations:
        - Nutritional balance
        - Calorie intake
        - Hydration recommendations
        - Electrolyte balance
        - Fiber intake
        - Specific dietary preferences
        
        Provide portion sizes, macro breakdown, and nutritional insights.
        """
        
        state["dietary_state"].is_streaming = True
        state["dietary_state"].last_update = datetime.now().isoformat()
        
        async for chunk in self.llm.astream(diet_prompt):
            if chunk.content:
                yield chunk.content
        
        state["dietary_state"].is_streaming = False

    def __call__(self, state: WorkoutState) -> WorkoutState:
        """Generate a dietary plan based on user profile"""
        diet_prompt = f"""
        Create a comprehensive dietary plan based on the following user profile:
        {state["user_profile"]}
        
        Generate a detailed meal plan including:
        - Breakfast options
        - Lunch options
        - Dinner options
        - Healthy snacks
        
        Key Considerations:
        - Nutritional balance
        - Calorie intake
        - Hydration recommendations
        - Electrolyte balance
        - Fiber intake
        - Specific dietary preferences
        
        Provide portion sizes, macro breakdown, and nutritional insights.
        """
        
        response = self.llm.invoke(diet_prompt)
        state["dietary_state"].content = response.content
        state["dietary_state"].last_update = datetime.now().isoformat()
        return state

class FitnessAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4-1106-preview",
            temperature=0.4,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    
    async def stream(self, state: WorkoutState) -> AsyncGenerator[str, None]:
        """Stream fitness recommendations"""
        fitness_prompt = f"""
        Design a comprehensive fitness plan based on the user profile:
        {state["user_profile"]}
        
        Plan should include:
        - Warm-up routines
        - Main workout plan
        - Cool-down exercises
        - Progress tracking recommendations
        
        Considerations:
        - Fitness goals
        - Current fitness level
        - Equipment availability
        - Time constraints
        - Injury prevention
        
        Provide detailed exercise descriptions, sets, reps, and intensity levels.
        """
        
        state["fitness_state"].is_streaming = True
        state["fitness_state"].last_update = datetime.now().isoformat()
        
        async for chunk in self.llm.astream(fitness_prompt):
            if chunk.content:
                yield chunk.content
        
        state["fitness_state"].is_streaming = False

    def __call__(self, state: WorkoutState) -> WorkoutState:
        """Generate a fitness plan based on user profile"""
        fitness_prompt = f"""
        Design a comprehensive fitness plan based on the user profile:
        {state["user_profile"]}
        
        Plan should include:
        - Warm-up routines
        - Main workout plan
        - Cool-down exercises
        - Progress tracking recommendations
        
        Considerations:
        - Fitness goals
        - Current fitness level
        - Equipment availability
        - Time constraints
        - Injury prevention
        
        Provide detailed exercise descriptions, sets, reps, and intensity levels.
        """
        
        response = self.llm.invoke(fitness_prompt)
        state["fitness_state"].content = response.content
        state["fitness_state"].last_update = datetime.now().isoformat()
        return state

class QueryRouter:
    """Routes queries to appropriate agents based on content"""
    
    def __call__(self, state: WorkoutState) -> str:
        query = state["current_query"].lower()
        
        # Keywords for classification
        dietary_keywords = ["meal", "diet", "food", "nutrition", "eat", "calories", "macro"]
        fitness_keywords = ["workout", "exercise", "training", "routine", "sets", "reps"]
        
        if any(word in query for word in dietary_keywords):
            return QueryType.DIETARY
        elif any(word in query for word in fitness_keywords):
            return QueryType.FITNESS
        else:
            return QueryType.GENERAL

class QueryAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4-1106-preview",
            temperature=0.5,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        self.router = QueryRouter()
    
    async def stream(self, state: WorkoutState) -> AsyncGenerator[str, None]:
        """Stream response to query"""
        if not state["current_query"]:
            return
        
        # Determine query type
        query_type = self.router(state)
        state["query_type"] = query_type
        
        query_prompt = f"""
        User Query: {state["current_query"]}
        Query Type: {query_type}
        
        Context:
        Dietary Plan: {state["dietary_state"].content}
        Fitness Plan: {state["fitness_state"].content}
        User Profile: {state["user_profile"]}
        
        Provide a detailed, personalized response to the user's question.
        Ensure the answer is:
        - Specific to their health and fitness plan
        - Actionable
        - Based on the generated plans
        - Focused on {query_type} aspects if the query is specific
        """
        
        async for chunk in self.llm.astream(query_prompt):
            if chunk.content:
                yield chunk.content

    def __call__(self, state: WorkoutState) -> WorkoutState:
        """Handle user queries about their fitness and dietary plans"""
        if not state["current_query"]:
            return state
        
        # Determine query type
        query_type = self.router(state)
        state["query_type"] = query_type
        
        query_prompt = f"""
        User Query: {state["current_query"]}
        Query Type: {query_type}
        
        Context:
        Dietary Plan: {state["dietary_state"].content}
        Fitness Plan: {state["fitness_state"].content}
        User Profile: {state["user_profile"]}
        
        Provide a detailed, personalized response to the user's question.
        Ensure the answer is:
        - Specific to their health and fitness plan
        - Actionable
        - Based on the generated plans
        - Focused on {query_type} aspects if the query is specific
        """
        
        response = self.llm.invoke(query_prompt)
        state["conversation_history"].append(
            {"role": "user", "content": state["current_query"]},
        )
        state["conversation_history"].append(
            {"role": "assistant", "content": response.content}
        )
        return state 