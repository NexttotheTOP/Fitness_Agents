from typing import Dict, Any
import json

from graph.chains.workout_variation import workout_variation_chain
from graph.workout_state import StateForWorkoutApp, Workout

def generate_workout_variation(state: StateForWorkoutApp) -> Dict[str, Any]:
    """Generate three variations of the input workout."""
    print("\n---GENERATING WORKOUT VARIATIONS---")
    
    original_workout = state["original_workout"]
    
    try:
        # Generate variations
        input_json = json.dumps(original_workout.model_dump(), indent=2)
        print("\nInput workout JSON:")
        print(input_json)
        
        # Call the chain
        print("\nCalling workout variation chain...")
        response = workout_variation_chain.invoke({
            "input_workout": input_json,
            "profile_overview": state["previous_complete_response"]
        })
        print(f"=========================================previous_complete_response: {state['previous_complete_response']}")
        
        # Parse the response
        print("\nRaw response from LLM:")
        print(response.content)
        
        result = json.loads(response.content)
        print("\nParsed JSON result:")
        print(json.dumps(result, indent=2))
        
        if "variations" not in result:
            print("\nERROR: No variations key in response")
            return {
                "original_workout": original_workout,
                "variations": [],
                "generation": "error"
            }
            
        # Convert to Workout objects
        variations = []
        for i, var in enumerate(result["variations"]):
            try:
                print(f"\nValidating variation {i+1}...")
                workout = Workout.model_validate(var)
                variations.append(workout)
                print(f"Variation {i+1} valid")
            except Exception as e:
                print(f"Error validating variation {i+1}: {str(e)}")
                continue
        
        print(f"\nSuccessfully generated {len(variations)} variations")
        return {
            "original_workout": original_workout,
            "variations": variations,
            "generation": "success"
        }
        
    except Exception as e:
        print("\n---ERROR GENERATING VARIATIONS---")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        print("Full error details:", e)
        return {
            "original_workout": original_workout,
            "variations": [],
            "generation": "error"
        } 