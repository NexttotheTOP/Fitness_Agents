from typing import Dict, Any, AsyncGenerator
from datetime import datetime
import os
import re
import requests
import base64
from io import BytesIO
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from graph.workout_state import WorkoutState, UserProfile, AgentState, QueryType
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ProfileAgent:
    def __init__(self):
        # Standard text model for non-visual analysis
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Vision model for body composition analysis
        self.vision_model = ChatOpenAI(
            model="gpt-4o",  # Using gpt-4o instead of gpt-image-1
            temperature=0.1,
            max_tokens=4096,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Get the Supabase service key and URL from environment variables
        self.supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")
        self.supabase_url = os.getenv("SUPABASE_URL", "https://hjzszgaugorgqbbsuqgc.supabase.co")
        
        if not self.supabase_service_key:
            print("Warning: SUPABASE_SERVICE_KEY not found in environment variables")
    
    def _extract_timestamp(self, filename: str) -> str:
        """Extract timestamp from image filename"""
        # Example: front-1745841393048-494691126_700784015796392_635117349760825196_n.jpg
        timestamp_match = re.search(r'-([\d]+)-', filename)
        if timestamp_match:
            unix_timestamp = int(timestamp_match.group(1)) / 1000  # Convert to seconds
            try:
                # Convert Unix timestamp to readable date
                date_time = datetime.fromtimestamp(unix_timestamp)
                return date_time.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, OverflowError):
                # If conversion fails, return the raw timestamp
                return timestamp_match.group(1)
        return "Unknown"
    
    def _get_image_as_base64(self, image_url: str) -> str:
        """Download image and convert to base64"""
        try:
            print(f"Attempting to download image from: {image_url}")
            
            # Check if this is a Supabase storage URL
            is_supabase_url = "supabase" in image_url and "/body-images/" in image_url
            
            # Set up authentication headers
            headers = {}
            if is_supabase_url and self.supabase_service_key:
                # For private Supabase buckets, we need to use authentication
                headers = {
                    "Authorization": f"Bearer {self.supabase_service_key}"
                }
                print("Using authenticated Supabase access with Authorization header")
                
                # Convert from public to authenticated URL pattern if needed
                if "/object/public/" in image_url:
                    image_url = image_url.replace("/object/public/", "/object/authenticated/")
                    print(f"Converted to authenticated URL: {image_url}")
            
            # Make the request with proper headers
            try:
                response = requests.get(image_url, headers=headers, timeout=15)
                response.raise_for_status()
                print(f"Successfully accessed image: Status {response.status_code}")
            except Exception as e:
                print(f"Initial access failed: {str(e)}")
                
                # If first attempt failed and it's a Supabase URL, try alternative endpoint
                if is_supabase_url and self.supabase_service_key:
                    try:
                        # Try the direct object endpoint without 'public' or 'authenticated'
                        alt_url = image_url.replace("/object/public/", "/object/")
                        alt_url = alt_url.replace("/object/authenticated/", "/object/")
                        
                        print(f"Trying alternative URL: {alt_url}")
                        response = requests.get(alt_url, headers=headers, timeout=15)
                        response.raise_for_status()
                        print(f"Successfully accessed image with alternative URL: Status {response.status_code}")
                    except Exception as e:
                        print(f"Alternative URL access failed: {str(e)}")
                        return None
            
            # Convert image to base64
            image_data = base64.b64encode(response.content).decode('utf-8')
            
            # Determine MIME type
            mime_type = "image/jpeg"  # Default to JPEG
            if image_url.lower().endswith('.png'):
                mime_type = "image/png"
            elif image_url.lower().endswith('.gif'):
                mime_type = "image/gif"
            elif "png" in image_url.lower():
                mime_type = "image/png"
            
            # Return as data URL
            return f"data:{mime_type};base64,{image_data}"
        except Exception as e:
            print(f"Error in _get_image_as_base64 for {image_url}: {str(e)}")
            return None
    
    def _format_body_photos(self, user_profile: dict) -> list:
        """Format body photos for vision model input"""
        formatted_images = []
        image_timestamps = {}
        
        # Check if we have image paths in the right format
        image_paths = user_profile.get("imagePaths", {})
        
        # Set the base URLs for Supabase storage - try authenticated endpoints for private buckets
        storage_base_url = f"{self.supabase_url}/storage/v1/object/authenticated/body-images/"
        storage_alt_url = f"{self.supabase_url}/storage/v1/object/body-images/"
        
        # Track failure information for better error handling
        failed_urls = []
        
        # Helper function to process a single image
        def process_single_image(view_type, path):
            timestamp = self._extract_timestamp(os.path.basename(path))
            
            # Try authenticated URLs since we know the bucket is private
            urls_to_try = [
                # Primary authenticated endpoint
                f"{storage_base_url}{path}",
                # Alternative endpoint (just /object/ without authenticated)
                f"{storage_alt_url}{path}",
                # If the path seems to be a complete URL already
                path if path.startswith('http') else None
            ]
            
            # Filter out None values
            urls_to_try = [url for url in urls_to_try if url]
            
            # Try each URL format
            base64_image = None
            for url in urls_to_try:
                base64_image = self._get_image_as_base64(url)
                if base64_image:
                    print(f"Successfully retrieved image for {view_type} view")
                    break
            
            if base64_image:
                formatted_images.append({
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image,
                        "detail": "high"
                    }
                })
                
                # Store timestamp 
                if view_type not in image_timestamps:
                    image_timestamps[view_type] = []
                image_timestamps[view_type].append({"path": path, "timestamp": timestamp})
                return True
            else:
                failed_urls.append(f"{view_type}: {path}")
                return False
        
        # Process front view images
        for front_path in image_paths.get("front", []):
            process_single_image("front", front_path)
        
        # Process side view images
        for side_path in image_paths.get("side", []):
            process_single_image("side", side_path)
        
        # Process back view images
        for back_path in image_paths.get("back", []):
            process_single_image("back", back_path)
        
        # If no images were successfully processed, log detailed information
        if not formatted_images:
            print("No images were successfully processed. Skipping body analysis.")
            print(f"Failed URLs: {failed_urls}")
            
            # Add diagnostic info to user_profile for debugging
            user_profile["image_access_errors"] = {
                "failed_urls": failed_urls,
                "timestamp": datetime.now().isoformat()
            }
        
        # Store timestamps in user profile for later use
        user_profile["image_timestamps"] = image_timestamps
        
        return formatted_images
    
    async def _analyze_body_composition(self, user_profile: dict) -> str:
        """Analyze body composition using vision model"""
        formatted_images = self._format_body_photos(user_profile)
        
        if not formatted_images:
            return "No body photos available for analysis."
        
        # Track if this is a progress assessment
        is_progress_analysis = False
        for view_type in ["front", "back", "side"]:
            if user_profile.get("imagePaths", {}).get(view_type, []) and len(user_profile.get("imagePaths", {}).get(view_type, [])) > 1:
                is_progress_analysis = True
                break
        
        # Get timestamps for chronological ordering if it's a progress assessment
        timestamps_info = ""
        if is_progress_analysis and "image_timestamps" in user_profile:
            timestamps_info = "\nImage Timestamps:\n"
            for view_type, timestamps in user_profile["image_timestamps"].items():
                if len(timestamps) > 1:
                    timestamps_info += f"\n{view_type.capitalize()} View Images:\n"
                    for i, ts_data in enumerate(timestamps):
                        timestamps_info += f"  - Image {i+1}: {ts_data['timestamp']}\n"
        
        # Create the prompt
        prompt_content = [
            {
                "type": "text",
                "text": f"""As an expert fitness coach and body composition specialist, carefully analyze these body photos.

User Profile:
- Age: {user_profile.get('age', 'Unknown')}
- Gender: {user_profile.get('gender', 'Unknown')}
- Height: {user_profile.get('height', 'Unknown')}
- Weight: {user_profile.get('weight', 'Unknown')}
- Activity Level: {user_profile.get('activity_level', 'Unknown')}
- Fitness Goals: {', '.join(user_profile.get('fitness_goals', []))}
{timestamps_info}

Provide a comprehensive analysis including:
1. Current body composition (estimated body fat percentage, muscle distribution)
2. Posture assessment (anterior/posterior pelvic tilt, shoulder alignment, etc.)
3. Somatotype assessment (ectomorph, mesomorph, endomorph, or combination)
4. Muscle development areas to focus on
5. Specific recommendations based on visible body composition
6. Potential mobility/flexibility issues
7. Strengths to leverage in training
"""
            }
        ]
        
        # Add progress tracking instructions if multiple photos
        if is_progress_analysis:
            prompt_content[0]["text"] += """
8. Progress assessment comparing the multiple photos
9. Changes in muscle development, body composition, and posture
10. Recommendations based on visible progress
"""
        
        # Add the images to the prompt
        prompt_content.extend(formatted_images)
        
        # Add final instructions
        prompt_content.append({
            "type": "text",
            "text": """
Format your analysis in clear sections.
Be specific, detailed, and professional while maintaining a supportive coaching tone.
Include a 'body_type' field that clearly states the somatotype.
"""
        })
        
        # Create the vision message
        message = HumanMessage(content=prompt_content)
        
        # Get response from vision model
        response = await self.vision_model.ainvoke([message])
        return response.content
    
    async def stream(self, state: WorkoutState) -> AsyncGenerator[str, None]:
        """Stream the profile analysis"""
        if not state["user_profile"]:
            state["user_profile"] = UserProfile().dict()
        
        # Check if we have body photos for analysis
        has_body_photos = False
        if isinstance(state["user_profile"], dict):
            has_body_photos = "imagePaths" in state["user_profile"] and any(state["user_profile"].get("imagePaths", {}).values())
        
        if has_body_photos:
            # Stream the body composition analysis
            body_analysis = await self._analyze_body_composition(state["user_profile"])
            
            # Prepare prompts with body analysis
            profile_prompt = f"""
            Analyze the following user profile information and body analysis to create 
            a comprehensive fitness profile:
            
            User Profile: {json.dumps(state["user_profile"])}
            
            Body Analysis: {body_analysis}
            
            Provide a comprehensive profile assessment that combines this information into a 
            cohesive assessment for the user. Focus on how their body composition, structure,
            and goals align to create actionable fitness recommendations.
            """
            
            async for chunk in self.llm.astream(profile_prompt):
                yield chunk.content
                
            # Store the body analysis in the state
            state["body_analysis"] = body_analysis
            if "body_type" not in state["user_profile"] or not state["user_profile"]["body_type"]:
                # Try to extract body_type from the body analysis
                if "body type:" in body_analysis.lower():
                    body_type_part = body_analysis.lower().split("body type:")[1].strip()
                    body_type = body_type_part.split("\n")[0].strip()
                    state["user_profile"]["body_type"] = body_type
                elif "somatotype:" in body_analysis.lower():
                    body_type_part = body_analysis.lower().split("somatotype:")[1].strip()
                    body_type = body_type_part.split("\n")[0].strip()
                    state["user_profile"]["body_type"] = body_type
        else:
            # Standard profile analysis without body photos
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
            
            Provide a comprehensive profile assessment including recommendations based on the 
            provided information.
            """
            
            async for chunk in self.llm.astream(profile_prompt):
                yield chunk.content

    async def _call_with_body_analysis(self, state: WorkoutState) -> WorkoutState:
        """Process state with body analysis using vision model"""
        body_analysis = await self._analyze_body_composition(state["user_profile"])
        
        # Update the state with body analysis
        state["body_analysis"] = body_analysis
        
        # Extract body type if present
        if "body type:" in body_analysis.lower():
            body_type_part = body_analysis.lower().split("body type:")[1].strip()
            body_type = body_type_part.split("\n")[0].strip()
            state["user_profile"]["body_type"] = body_type
        elif "somatotype:" in body_analysis.lower():
            body_type_part = body_analysis.lower().split("somatotype:")[1].strip()
            body_type = body_type_part.split("\n")[0].strip()
            state["user_profile"]["body_type"] = body_type
        
        # Create full profile analysis
        profile_prompt = f"""
        Analyze the following user profile information and body analysis to create 
        a comprehensive fitness profile:
        
        User Profile: {json.dumps(state["user_profile"])}
        
        Body Analysis: {body_analysis}
        
        Provide a comprehensive profile assessment that combines this information into a 
        cohesive assessment for the user. Focus on how their body composition, structure,
        and goals align to create actionable fitness recommendations.
        """
        
        response = self.llm.invoke(profile_prompt)
        state["user_profile"]["body_analysis"] = body_analysis
        state["user_profile"] = response.content
        return state

    def __call__(self, state: WorkoutState) -> WorkoutState:
        """Update or validate the user profile"""
        if not state["user_profile"]:
            state["user_profile"] = UserProfile().dict()
        
        # Check if we have body photos for analysis
        has_body_photos = False
        if isinstance(state["user_profile"], dict):
            has_body_photos = "imagePaths" in state["user_profile"] and any(state["user_profile"].get("imagePaths", {}).values())
        
        if has_body_photos:
            # We're in a synchronous method but need to handle async code
            # Just do basic text analysis here instead of vision analysis
            # The actual vision analysis will happen in the async stream method
            profile_prompt = f"""
            Analyze the user profile information to create a preliminary assessment.
            The detailed body analysis will be performed separately.
            Current profile: {state["user_profile"]}
            
            Provide an initial profile assessment based on the provided information.
            """
            
            response = self.llm.invoke(profile_prompt)
            state["user_profile"] = response.content
            # Mark that this profile needs visual analysis
            state["needs_visual_analysis"] = True
            return state
        
        # Standard profile analysis without body photos
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
        
        Provide a comprehensive profile assessment including recommendations based on the 
        provided information.
        In your response, include a field called "body_type" that summarizes the user's body type if photos are provided.
        """
        
        response = self.llm.invoke(profile_prompt)
        
        # Update the user profile
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