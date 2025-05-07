from typing import Dict, Any, List
import json
from pydantic import ValidationError
from graph.workout_state import StateForWorkoutApp, Workout, Exercise, ExerciseDetails

def create_workout_from_nlq(state: StateForWorkoutApp) -> Dict[str, Any]:
    """Generate workouts based on natural language query and user profile"""
    print("\n---GENERATING WORKOUTS FROM NLQ---")
    
    workout_prompt = state.get("workout_prompt", "")
    if not workout_prompt:
        print("No workout prompt provided in state")
        return state
    
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
        
        # Create the prompt
        context_prompt = f"""
        [Persona]
        You are an expert fitness coach and personal trainer with extensive experience creating personalized workout plans.
        You excel at designing safe, effective, and engaging workouts that align with your clients' goals while considering their unique circumstances and limitations.

        [Task]
        First, explain your reasoning for designing the workout(s) for this client, considering their profile, body analysis,goals, and limitations. Share your thought process, why you chose certain exercises, structure, and approach, and how you accounted for any health or equipment restrictions.
        Then, generate 2-3 detailed, structured workout plans that satisfy the client's requirements and its profile.
        
        [Client Request]
        {workout_prompt}
        
        [Client Profile]
        {json.dumps(user_profile, indent=2)}
        
        [Client Profile Assessment]
        {profile_assessment}
        
        [Client Body Analysis]
        {body_analysis}
        
        [AI Profile Analysis]
        Our dedicated AI Profile Analysis Agent has thoroughly analyzed the client's profile, body composition, and health data.
        Based on this comprehensive analysis, here are the key factors to consider in workout design:

        Health Limitations: {json.dumps(health_limitations, indent=2)}
        Exercise Restrictions: {json.dumps(exercise_restrictions, indent=2)}
        Recommended Exercise Modifications: {json.dumps(recommended_modifications, indent=2)}
        Equipment Preferences/Available: {json.dumps(equipment_preferences, indent=2)}
        Focus Areas for Improvement: {json.dumps(focus_areas, indent=2)}
        
        [Instructions]
        Create workout plans tailored to this specific Client's needs, goals, and limitations.
        Each workout should:
        1. Address the specific request in the user prompt
        2. Consider their fitness level and goals
        3. Account for ALL health limitations and exercise restrictions identified by our Analysis Agent
        4. Incorporate the recommended exercise modifications when relevant
        5. Prioritize exercises for the identified focus areas while maintaining overall balance
        6. Use available/preferred equipment when possible
        7. Include appropriate exercises with proper form guidance
        8. Specify sets and either reps OR duration for each exercise (use reps for strength exercises, duration for cardio/endurance)
        
        Start your response with a section titled "reasoning" (as a string) where you explain your thought process and design choices for this client.
        Then, output the workouts as a JSON array under the key "workouts" as before.
        Return a single JSON object with two keys: "reasoning" and "workouts".
        Do not include any text or explanation outside the JSON object.
        
        The expected format is:
        {{
          "reasoning": "Your explanation here...",
          "workouts": [
            {{
              "name": "Workout Name",
              "description": "Detailed description of the workout's purpose and benefits",
              "difficulty_level": "beginner/intermediate/advanced",
              "estimated_duration": "45 minutes",
              "target_muscle_groups": ["Primary Muscle Groups"],
              "equipment_required": ["Required Equipment"],
              "exercises": [
                {{
                  "name": "Exercise Name",
                  "sets": 3,
                  "reps": 12,  # Use for strength/resistance exercises
                  "duration": "30 seconds",  # Use for cardio/endurance exercises
                  "notes": "Form notes and guidance",
                  "details": {{
                    "description": "Exercise description",
                    "category": "strength/cardio/flexibility/etc",
                    "muscle_groups": ["Primary Muscle", "Secondary Muscle"],
                    "difficulty": "beginner/intermediate/advanced",
                    "equipment_needed": "Equipment required"
                  }}
                }}
              ]
            }}
          ]
        }}
        """
        
        # Call the LLM
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,  # Lower temperature for more consistent output
            response_format={"type": "json_object"}
        )
        
        print("\nSending prompt to LLM...")
        response = llm.invoke(context_prompt)
        
        # Parse the response
        print("\nParsing LLM response...")
        result = json.loads(response.content)
        print("\nRaw LLM response:")
        print(json.dumps(result, indent=2))
        
        # Extract reasoning and workouts
        reasoning = result.get("reasoning", "")
        workouts_data = result.get("workouts", [])
        
        # Store reasoning at the top level
        state["reasoning"] = reasoning
        print("Returning state from workout creation node, reasoning:", state.get("reasoning"))
        
        if not workouts_data:
            print("No workouts found in response")
            return state
        
        # Initialize created_workouts list if it doesn't exist
        if "created_workouts" not in state:
            state["created_workouts"] = []
        
        # Validate and store each workout
        print("\nValidating workout structures...")
        valid_workouts = []
        for workout_data in workouts_data:
            try:
                workout = Workout.model_validate(workout_data)
                valid_workouts.append(workout)
                print(f"Successfully validated workout: {workout.name}")
            except ValidationError as e:
                print(f"Validation error in workout:")
                for error in e.errors():
                    print(f"- {error['loc']}: {error['msg']}")
                continue
        
        if valid_workouts:
            # Add the new workouts to the list
            state["created_workouts"].extend(valid_workouts)
            
            # Set the first workout as original_workout for variation compatibility
            state["original_workout"] = valid_workouts[0]
            
            print(f"\nSuccessfully created {len(valid_workouts)} workouts")
        else:
            print("\nNo valid workouts were created")
        
        return state
        
    except Exception as e:
        print("\n---ERROR GENERATING WORKOUTS---")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        if isinstance(e, ValidationError):
            print("\nValidation errors:")
            for error in e.errors():
                print(f"- {error['loc']}: {error['msg']}")
        return state
