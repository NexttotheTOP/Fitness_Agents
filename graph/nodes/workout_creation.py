from typing import Dict, Any, List
import json
from pydantic import ValidationError
from graph.workout_state import StateForWorkoutApp, Workout, Exercise, ExerciseDetails, WorkoutsResponse, ExercisesResponse
from graph.nodes.workout_analysis import build_context_info
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import StreamWriter
from langgraph.config import get_stream_writer

async def create_workout_from_nlq(state: StateForWorkoutApp):
    """Generate workouts based on natural language query and user profile"""
    
    writer = get_stream_writer()
    
    writer({"type": "progress", "content": "Creating your workouts now..."})
    
    workout_prompt = state.get("workout_prompt", "")
    if not workout_prompt:
        writer({"type": "error", "content": "No workout prompt provided"})
        return
    
    try:
        # Get user profile context
        user_profile = state.get("user_profile", {})
        profile_assessment = state.get("profile_assessment", "")
        body_analysis = state.get("body_analysis", "")
        plan_proposal_markdown = state.get("plan_proposal_markdown", "")
        context = state.get("context", {})
        context_exercises = context.get("exercises", [])
        context_workouts = context.get("workouts", [])
        
        # Build formatted context information
        context_info = build_context_info(context, workout_prompt)
        overview = state.get("previous_complete_response", "")
        
        # Build messages
        messages = []
        messages.append(SystemMessage(content="""You are a world-class fitness coach and workout generator. 
Your job is to take the provided structured workout plan, or single workouts and or exercises from the proposal and turn it into a detailed, high-quality, ready-to-use workout JSON object.

Instructions:
- Use the plan proposal below as your main source of truth for creating the detailed workouts and or exercises.
- Strictly follow the required JSON schema and include all required fields (name, description, exercises, sets, reps, rest, etc.).
- If the proposal is missing details (e.g., rest times, tempo, cues), fill them in with best practices for the user's goals and level.
- Make sure to include all the mentioned workouts and or exercises in the proposal.

Focus on creating highly detailed and informative content:
- Write comprehensive workout descriptions that explain the purpose, benefits, and special considerations
- Provide detailed form guidance and technique notes for each exercise
- Include specific cues and tips for proper execution
- Add relevant safety considerations and modifications
- Explain the reasoning behind exercise selection and progression
- Specify exact sets and either reps OR duration for each exercise (use reps for strength exercises, duration for cardio/endurance)

Output format:
- Always return a single JSON object with both `workouts` and `exercises` keys
- The `workouts` array should contain complete workout definitions (core sessions only).
- The `exercises` array should contain any standalone exercises—including all warm-up, cooldown, mobility, and stretching items.  
- Do not use ranges for the number of sets or reps. Use specific numbers.
"""))
        
        example_json = '''
        {
          "workouts": [
            {
              "name": "Workout Name",
              "description": "Detailed description...",
              "difficulty_level": "beginner/intermediate/advanced",
              "estimated_duration": "45 minutes",
              "target_muscle_groups": ["Primary Muscle Groups"],
              "equipment_required": ["Required Equipment"],
              "exercises": [
                {
                  "name": "Exercise Name",
                  "sets": 3,
                  "reps": 12,
                  "duration": "30 seconds",
                  "notes": "Comprehensive form notes...",
                  "details": {
                    "description": "Exercise description",
                    "category": "strength/cardio/flexibility/etc",
                    "muscle_groups": ["Primary Muscle", "Secondary Muscle"],
                    "difficulty": "beginner/intermediate/advanced",
                    "equipment_needed": "Equipment required"
                  }
                }
              ]
            }
          ]
        }
        
        {
          "exercises": [
            {
              "name": "Push-Up",
              "sets": 4,
              "reps": 10,
              "notes": "Keep your core tight and back flat.",
              "details": {
                "description": "A body-weight exercise that targets the chest, shoulders, and triceps.",
                "category": "strength",
                "muscle_groups": ["Chest", "Triceps", "Shoulders"],
                "difficulty": "intermediate",
                "equipment_needed": "None"
              }
            }
          ]
        }
        '''
        messages.append(SystemMessage(content=f"AGREED PROPOSED PLAN:\n{plan_proposal_markdown}"))
        messages.append(SystemMessage(content=f"Simple Example Output Format:\n{example_json}"))
        
        # Call the LLM
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5,  # Lower temperature for more consistent output
            #response_format={"type": "json_object"}
        )
        
        print("\nSending prompt to LLM...")
        # Use non-streaming invocation so we receive the full answer at once
        response = await llm.ainvoke(messages)
        # ChatOpenAI returns a ChatMessage object – extract its text content
        response_text = getattr(response, "content", str(response))

        # Clean markdown if needed
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        print("\nParsing LLM response...")
        print("\nRaw LLM response:")
        print(response_text[:500])  # Log first 500 chars for debug
        writer({"type": "result", "content": response_text})
        
        # --------------------------------------------------------------
        # Validate the JSON response with our Pydantic models
        # --------------------------------------------------------------

        # try:
        #     # Attempt to validate as a list of full workouts first
        #     workouts_response = WorkoutsResponse.model_validate_json(response_text)
        #     print("Successfully validated response as WorkoutsResponse. Total workouts:", len(workouts_response.workouts))

        #     # Ensure key exists
        #     if "created_workouts" not in state:
        #         state["created_workouts"] = []

        #     # Extend state list with validated Workout objects
        #     state["created_workouts"].extend(workouts_response.workouts)

        # except ValidationError as w_err:
        #     print("Response is not a valid WorkoutsResponse. Trying ExercisesResponse...")
        #     try:
        #         exercises_response = ExercisesResponse.model_validate_json(response_text)
        #         print("Successfully validated response as ExercisesResponse. Total exercises:", len(exercises_response.exercises))

        #         # Ensure key exists
        #         if "created_exercises" not in state:
        #             state["created_exercises"] = []

        #         state["created_exercises"].extend(exercises_response.exercises)

        #     except ValidationError as e_err:
        #         # If both validations fail, propagate the error
        #         print("Failed to validate response as either WorkoutsResponse or ExercisesResponse")
        #         print("Workout validation error:", w_err)
        #         print("Exercise validation error:", e_err)
        #         raise e_err  # Let outer except handle

        # Store reasoning at the top level if previously captured
        
        # Simple completion message
        writer({"type": "progress", "content": "Workout creation completed"})
        
        # Yield the result so it is captured in the graph state
        yield {
            "type": "result",
            "content": response_text
        }
        
    except Exception as e:
        print("\n---ERROR GENERATING WORKOUTS---")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        if isinstance(e, ValidationError):
            print("\nValidation errors:")
            for error in e.errors():
                print(f"- {error['loc']}: {error['msg']}")
        writer({"type": "error", "content": str(e)})
        yield {"type": "error", "content": str(e)}

