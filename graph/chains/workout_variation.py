from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import os
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

class Exercise(BaseModel):
    name: str
    sets: int
    reps: int
    notes: str
    details: dict

class WorkoutVariation(BaseModel):
    name: str
    description: str
    exercises: List[Exercise]

class WorkoutVariations(BaseModel):
    """Container for multiple workout variations."""
    variations: List[WorkoutVariation] = Field(
        description="List of three workout variations"
    )

SYSTEM_PROMPT = """You are a professional fitness trainer who creates workout variations.
Your task is to create THREE different variations of the given workout while maintaining effectiveness."""

HUMAN_PROMPT = """Create 3 variations of this workout:

{input_workout}

Guidelines:
- Create exactly 3 different variations
- Modify exercises while keeping a good workout flow
- Adjust sets, reps, and equipment as needed
- Include clear form notes

Return your response as a JSON object with this structure:
{{
  "variations": [
    {{
      "name": "Name of Variation 1",
      "description": "Description of the variation",
      "exercises": [
        {{
          "name": "Exercise Name",
          "sets": 3,
          "reps": 12,
          "notes": "Form notes here",
          "details": {{
            "description": "Exercise description",
            "category": "strength",
            "muscle_groups": ["Target Muscle"],
            "difficulty": "intermediate",
            "equipment_needed": "Equipment Name"
          }}
        }}
      ]
    }}
  ]
}}"""

# Create the chain
workout_variation_chain = (
    ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])
    | ChatOpenAI(
        temperature=0.7,
        model="gpt-4-1106-preview",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        response_format={"type": "json_object"}
    )
)

def analyze_workout(workout: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = workout_variation_chain.invoke({"workout": workout})
        
        # Validate that we received a proper JSON response
        if not isinstance(response, dict):
            raise ValueError("Expected JSON object response, got: " + str(type(response)))
            
        return response
    except Exception as e:
        logger.error(f"Error analyzing workout: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze workout: {str(e)}"
        )

def validate_variation(original: Dict[str, Any], variation: Dict[str, Any]) -> bool:
    """Basic validation of the variation."""
    try:
        # Ensure we have at least 2 exercises
        if len(variation["exercises"]) < 2:
            return False
            
        # Ensure we have some overlapping muscle groups
        original_analysis = analyze_workout(original)
        variation_analysis = analyze_workout(variation)
        
        original_muscles = set(original_analysis["target_muscle_groups"])
        variation_muscles = set(variation_analysis["target_muscle_groups"])
        
        # At least 30% of original muscle groups should be present
        overlap = len(original_muscles.intersection(variation_muscles))
        min_required = max(1, len(original_muscles) * 0.3)
        
        return overlap >= min_required
    except Exception:
        return False 