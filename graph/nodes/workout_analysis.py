from langchain_openai import ChatOpenAI
from graph.workout_state import StateForWorkoutApp
import json
import logging

class WorkoutAnalysisAgent:
    """Agent that analyzes user profile for workout-relevant information"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
        )
    
    def analyze_user_profile(self, state: StateForWorkoutApp) -> StateForWorkoutApp:
        """
        Analyze the user profile and extract health limitations, injuries,
        and other relevant information for workout planning.
        """
        print("\n---ANALYZING USER PROFILE---")
        
        # Get profile information from state
        profile_assessment = state.get("profile_assessment", "")
        body_analysis = state.get("body_analysis", "")
        user_profile = state.get("user_profile", {})
        
        print(f"\nAnalyzing backend profile data:")
        print(f"- Profile assessment available: {'Yes' if profile_assessment else 'No'}")
        print(f"- Body analysis available: {'Yes' if body_analysis else 'No'}")
        print(f"- User profile metadata keys: {list(user_profile.keys())}")
        print(f"- User profile: {user_profile}")
        
        # Extract relevant parts from previous sections if available
        previous_sections = state.get("previous_sections", {})
        previous_profile = previous_sections.get("profile_assessment", "")
        
        print(f"\nPrevious data found:")
        print(f"- Previous sections keys: {list(previous_sections.keys())}")
        print(f"- Previous profile length: {len(previous_profile)}")
        
        # Build the prompt for analysis
        prompt = f"""
        [Persona]
        You are a highly skilled fitness assessment specialist who excels at analyzing client profiles, identifying potential health risks, and determining appropriate exercise modifications.
        Your expertise lies in thoroughly evaluating the client's profile data to ensure safe and effective workout planning.
        
        [Task]
        Analyze the user's fitness profile and identify any health limitations, injuries, or special 
        considerations that should be factored into workout planning.
        
        [User Profile Data]
        {json.dumps(user_profile, indent=2)}
        
        [Profile Assessment]
        {profile_assessment}
        
        [Body Analysis]
        {body_analysis}
        
        [Previous Profile Assessment]
        {previous_profile}
        
        [Instructions]
        1. Identify any explicit mentions of injuries, health conditions, or physical limitations
        2. Infer potential limitations based on body analysis (posture issues, imbalances)
        3. Note any equipment limitations or preferences mentioned
        4. Identify exercise types that should be avoided based on this analysis
        5. Suggest modifications or alternatives for common problematic exercises
        
        Format your response as a JSON object with these keys:
        - health_limitations: List of specific health issues or injuries
        - exercise_restrictions: List of exercises or movement types to avoid
        - recommended_modifications: List of specific exercise modifications
        - equipment_preferences: List of preferred or available equipment
        - focus_areas: Body areas needing special attention or strengthening
        
        IMPORTANT: Return ONLY the JSON object, no additional text or formatting.
        """
        
        print("\nCalling LLM for analysis...")
        
        try:
            # Call the LLM
            response = self.llm.invoke(prompt)
            print("\nReceived LLM response")
            
            # Clean the response - remove any markdown formatting and get just the JSON
            response_text = response.content
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse the response as JSON
            print("\nParsing LLM response...")
            analysis_result = json.loads(response_text)
            print(f"Successfully parsed analysis with keys: {list(analysis_result.keys())}")
            
            # Update the state with the analysis
            state["health_limitations"] = analysis_result.get("health_limitations", [])
            state["exercise_restrictions"] = analysis_result.get("exercise_restrictions", [])
            state["recommended_modifications"] = analysis_result.get("recommended_modifications", [])
            state["equipment_preferences"] = analysis_result.get("equipment_preferences", [])
            state["focus_areas"] = analysis_result.get("focus_areas", [])
            
            # Store the full analysis
            state["workout_profile_analysis"] = analysis_result
            
            print("\nAnalysis results:")
            print(f"- Health limitations: {len(state['health_limitations'])} found")
            print(f"- Exercise restrictions: {len(state['exercise_restrictions'])} found")
            print(f"- Recommended modifications: {len(state['recommended_modifications'])} found")
            
            return state
            
        except Exception as e:
            print(f"\n---ERROR IN PROFILE ANALYSIS---")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            print(f"Raw response content: {response.content if 'response' in locals() else 'No response received'}")
            
            # Set defaults if parsing fails
            state["health_limitations"] = []
            state["exercise_restrictions"] = []
            state["recommended_modifications"] = []
            state["equipment_preferences"] = []
            state["focus_areas"] = []
            state["workout_profile_analysis"] = {}
            
            return state
