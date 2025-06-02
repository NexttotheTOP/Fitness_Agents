from typing import Dict, Any, List
import json
from pydantic import ValidationError
from graph.workout_state import StateForWorkoutApp, Workout, Exercise, ExerciseDetails
from graph.nodes.workout_analysis import build_context_info
from langchain_core.messages import SystemMessage, HumanMessage

async def create_workout_from_nlq(state: StateForWorkoutApp):
    """Generate workouts based on natural language query and user profile"""
    
    # Simple status message
    yield {"type": "progress", "content": "Starting workout creation..."}
    
    workout_prompt = state.get("workout_prompt", "")
    if not workout_prompt:
        yield {"type": "error", "content": "No workout prompt provided"}
        return
    
    try:
        # Get user profile context
        user_profile = state.get("user_profile", {})
        profile_assessment = state.get("profile_assessment", "")
        body_analysis = state.get("body_analysis", "")
        health_limitations = state.get("health_limitations", [])
        exercise_restrictions = state.get("exercise_restrictions", [])
        recommended_modifications = state.get("recommended_modifications", [])
        equipment_preferences = state.get("equipment_preferences", [])
        focus_areas = state.get("focus_areas", [])
        plan_proposal_markdown = state.get("plan_proposal_markdown", "")
        context = state.get("context", {})
        context_exercises = context.get("exercises", [])
        context_workouts = context.get("workouts", [])
        
        # Build formatted context information
        context_info = build_context_info(context, workout_prompt)
        overview = state.get("previous_complete_response", "")
        
        # Build messages
        messages = []
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
- Start your response with a section titled 'reasoning' (as a string) where you explain your thought process and design choices directly to the client (the user), using clear, simple, and honest language as if you are explaining to a 15-year-old.
- Then, output the workouts as a JSON array under the key 'workouts' as before.
- Return a single JSON object with two keys: 'reasoning' and 'workouts'.
- Do not include any text or explanation outside the JSON object."""))
        
        example_json = '''
        {
          "reasoning": "Your explanation here...",
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
        '''
        if plan_proposal_markdown:
            messages.append(SystemMessage(content=f"Plan Proposal:\n{plan_proposal_markdown}"))
        if overview:
            messages.append(SystemMessage(content=f"User Profile Overview:\n{overview}"))
        if context_info:
            messages.append(SystemMessage(content=f"Referenced Context:\n{context_info}"))
        messages.append(HumanMessage(content=f"User Request:\n{workout_prompt}"))
        messages.append(SystemMessage(content=f"Example Output Format:\n{example_json}"))
        
        # Call the LLM
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.5,  # Lower temperature for more consistent output
            response_format={"type": "json_object"}
        )
        
        print("\nSending prompt to LLM...")
        response_text = ""
        async for chunk in llm.astream(messages):
            token = getattr(chunk, "content", None) or str(chunk)
            response_text += token
            yield {"type": "token", "content": token}

        # Clean markdown if needed
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        print("\nParsing LLM response...")
        #result = json.loads(response_text)
        print("\nRaw LLM response:")
        print(json.dumps(result, indent=2))
        yield {"type": "result", "content": result}
        
        # Extract reasoning and workouts
        #reasoning = result.get("reasoning", "")
        #workouts_data = result.get("workouts", [])
        
        # Store reasoning at the top level
        #state["reasoning"] = reasoning
        print("Returning state from workout creation node, reasoning:", state.get("reasoning"))
        
        # if not workouts_data:
        #     print("No workouts found in response")
        #     yield {"type": "error", "content": "No workouts found in response"}
        #     return
        
        # Initialize created_workouts list if it doesn't exist
        if "created_workouts" not in state:
            state["created_workouts"] = []
        
        # Validate and store each workout
        print("\nValidating workout structures...")
        
        # After workout creation logic
        
        # Simple completion message
        yield {"type": "progress", "content": "Workout creation completed"}
        
        yield {"type": "result", "content": response_text}
        
    except Exception as e:
        print("\n---ERROR GENERATING WORKOUTS---")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        if isinstance(e, ValidationError):
            print("\nValidation errors:")
            for error in e.errors():
                print(f"- {error['loc']}: {error['msg']}")
        yield {"type": "error", "content": str(e)}

