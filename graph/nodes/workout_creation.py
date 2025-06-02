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
        messages.append(SystemMessage(content=f"PLAN PROPOSAL:\n{plan_proposal_markdown}"))
        messages.append(SystemMessage(content="""You are an expert fitness coach and personal trainer with extensive experience creating personalized workout plans. You excel at designing safe, effective, and engaging workouts that align with your clients' goals while considering their unique circumstances and limitations.

Instructions:
- Use the plan proposal below as your primary blueprint for creating the detailed workouts.
- If the user has referenced specific exercises or workouts in the context below, incorporate them appropriately into your workout designs, depending on the user query.
- Follow the structure, rationale, and focus areas outlined in the proposal.
- Create at least 4-6 comprehensive, detailed workout plans unless the proposal or user request specifies otherwise.
- Vary the number of exercises per workout (not always 4), and vary the estimated duration (e.g., some 30, 45, 60, or 90 minutes).
- Each workout should:
  1. Address the specific request in the user prompt and plan proposal
  2. Consider the client's fitness level, goals, and all analysis results
  3. Account for ALL health limitations and exercise restrictions identified by our Analysis Agent
  4. Incorporate the recommended exercise modifications when relevant
  5. Prioritize exercises for the identified focus areas while maintaining overall balance
  6. Use available/preferred equipment when possible
  7. Include appropriate exercises with proper comprehensive form guidance
  8. Include a detailed, user-facing workout description that explains the purpose, benefits, and any special considerations for the client.
  9. Specify sets and either reps OR duration for each exercise (use reps for strength exercises, duration for cardio/endurance)

Output format:
- If you are returning **full workouts**, wrap them in a JSON object with a single key `workouts`.
- If you are returning **stand-alone exercises only** (no grouping into workouts), wrap them in a JSON object with a single key `exercises`.

Do **NOT** include any extra keys or free-text outside the JSON object."""))
        
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
        if overview:
            messages.append(SystemMessage(content=f"User Profile Overview:\n{overview}"))
        if context_info:
            messages.append(SystemMessage(content=f"Referenced Context:\n{context_info}"))
        messages.append(HumanMessage(content=f"User Request:\n{workout_prompt}"))
        messages.append(SystemMessage(content=f"Simple Example Output Format:\n{example_json}"))
        
        # Call the LLM
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o",
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

