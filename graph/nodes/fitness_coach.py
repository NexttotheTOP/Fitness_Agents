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
            model="gpt-4o",
            temperature=0.1,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Vision model for body composition analysis
        self.vision_model = ChatOpenAI(
            model="gpt-4o",  # GPT-4o supports vision out of the box
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
                print(f"Successfully accessed image: Status {response.status_code}, Content-Length: {len(response.content)} bytes")
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
                        print(f"Successfully accessed image with alternative URL: Status {response.status_code}, Content-Length: {len(response.content)} bytes")
                    except Exception as e:
                        print(f"Alternative URL access failed: {str(e)}")
                        return None
            
            # Check if we actually got image data
            if not response.content or len(response.content) < 100:
                print(f"Warning: Response may not contain valid image data (too small: {len(response.content)} bytes)")
                return None
                
            # Check if image size is too large for API (typically 20MB limit for OpenAI)
            content_size_mb = len(response.content) / (1024 * 1024)
            if content_size_mb > 19:  # 19MB to be safe
                print(f"Warning: Image is too large ({content_size_mb:.2f} MB). Resizing might be needed.")
                # We continue for now, but future enhancement could resize the image
            
            # Determine MIME type from content (more reliable)
            content_type = response.headers.get('Content-Type', '')
            
            if 'image/' in content_type:
                mime_type = content_type  # Use the server-provided MIME type
            else:
                # Try to detect MIME type from URL or fall back to defaults
                if image_url.lower().endswith('.png'):
                    mime_type = "image/png"
                elif image_url.lower().endswith('.gif'):
                    mime_type = "image/gif"
                elif image_url.lower().endswith('.jpeg') or image_url.lower().endswith('.jpg'):
                    mime_type = "image/jpeg"
                elif "png" in image_url.lower():
                    mime_type = "image/png"
                else:
                    mime_type = "image/jpeg"  # Default to JPEG
            
            # Convert image to base64
            image_data = base64.b64encode(response.content).decode('utf-8')
            
            # Make sure the data URL is properly formatted
            data_url = f"data:{mime_type};base64,{image_data}"
            
            # Print sample of base64 data for debugging (first 20 chars)
            data_sample = image_data[:20] + "..." if len(image_data) > 20 else image_data
            print(f"Successfully encoded image as base64 ({mime_type}, {len(image_data)} chars)")
            print(f"Data URL format: data:{mime_type};base64,{data_sample}...")
            
            # Verify the data URL format to ensure it's correct for the API
            if not data_url.startswith("data:image/"):
                print(f"Warning: Data URL format seems incorrect: {data_url[:30]}...")
                
            return data_url
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
                # Follow OpenAI's exact format for image_url objects
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
        else:
            # Print information about the successfully formatted images
            print(f"Successfully processed {len(formatted_images)} images:")
            for i, img in enumerate(formatted_images):
                # Check if the image URL is a base64 string
                is_base64 = isinstance(img['image_url']['url'], str) and img['image_url']['url'].startswith('data:')
                
                # Print summary info about each image
                url_preview = "base64 data" if is_base64 else img['image_url']['url']
                print(f"  Image {i+1}: type={img['type']}, format={url_preview[:30]}...")
        
        # Store timestamps in user profile for later use
        user_profile["image_timestamps"] = image_timestamps
        
        return formatted_images
    
    async def _analyze_body_composition(self, user_profile):
        """Analyze body composition using a vision model and user input"""
        print("Starting body composition analysis...")
        
        # Exit early if no imagePaths found
        image_paths = user_profile.get("imagePaths", {})
        if not image_paths or not any(image_paths.values()):
            print("No body photos found in profile, cannot analyze body composition")
            return None

        # Format the images for OpenAI API call
        formatted_images = self._format_body_photos(user_profile)
        
        if not formatted_images:
            print("No valid body photos could be processed, skipping analysis")
            return None
            
        print(f"Formatted {len(formatted_images)} images for analysis")
        
        # Create analysis prompt based on user information
        prompt_text = f"""
You are a professional fitness coach and nutritionist analyzing body composition photos.
Your task is to analyze these body photos, which show the person from different angles.

User details:
- Age: {user_profile.get('age', 'unknown')}
- Gender: {user_profile.get('gender', 'unknown')}
- Height: {user_profile.get('height', 'unknown')}
- Weight: {user_profile.get('weight', 'unknown')}
- Fitness goals: {user_profile.get('fitness_goals', 'unknown')}

Analyze each photo carefully. First describe what you see in a neutral, professional way.
Then estimate:
1. Approximate body fat percentage
2. Current muscle mass regions (areas with good development vs. needs improvement)
3. Posture observations 
4. Apparent imbalances

Finally, summarize your assessment and provide objective recommendations based on these visual observations combined with the user's stated goals.
"""
        
        # Create a content array for the message
        content = [
            {"type": "text", "text": prompt_text}
        ]
        
        # Add each formatted image to the content array
        for image in formatted_images:
            content.append(image)  # Each image is already in {"type": "image_url", "image_url": {"url": ...}} format
        
        # Add additional instructions at the end
        content.append({
            "type": "text", 
            "text": "Based on all images shown, provide a complete analysis with percentage estimates. Be professional and honest in your assessment, while remaining encouraging."
        })
        
        # Make sure the content format is correct for a multimodal message
        print(f"Content has {len(content)} items: {len([i for i in content if i['type'] == 'text'])} text and {len([i for i in content if i['type'] == 'image_url'])} images")
        
        # Print the structure of content for debugging (without the actual image data)
        debug_content = []
        for item in content:
            if item['type'] == 'text':
                debug_content.append({'type': 'text', 'text': item['text'][:50] + '...'})
            else:
                url_preview = "data:image..." if isinstance(item['image_url']['url'], str) and item['image_url']['url'].startswith('data:') else str(item['image_url']['url'])[:30] + '...'
                debug_content.append({'type': 'image_url', 'image_url': {'url': url_preview}})
        
        print(f"Message content structure: {json.dumps(debug_content, indent=2)}")
        
        try:
            # Use the vision model initialized in __init__
            messages = [HumanMessage(content=content)]
            print(f"Sending request to vision model with {len(messages)} messages")
            
            response = await self.vision_model.ainvoke(messages)
            
            # Extract the analysis from the response
            analysis = response.content
            
            # Print a sample of the response for debugging
            print(f"Analysis response received, {len(analysis)} chars")
            print(f"Sample of analysis: {analysis[:150]}...")
            
            return analysis
        except Exception as e:
            print(f"Error during body composition analysis: {str(e)}")
            # Print more detailed error information
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            return None
    
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
        if body_analysis:
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
        
        Body Analysis: {body_analysis if body_analysis else "No body analysis available"}
        
        Provide a comprehensive profile assessment that combines this information into a 
        cohesive assessment for the user. Focus on how their body composition, structure,
        and goals align to create actionable fitness recommendations.
        """
        
        response = await self.llm.ainvoke(profile_prompt)
        if body_analysis:
            state["user_profile"]["body_analysis"] = body_analysis
        state["user_profile"] = response.content
        return state

    def __call__(self, state: WorkoutState) -> WorkoutState:
        """Update or validate the user profile, preserving the original dictionary structure"""
        if not state["user_profile"]:
            state["user_profile"] = UserProfile().dict()
        
        # Check if we have body photos for analysis
        has_body_photos = False
        if isinstance(state["user_profile"], dict):
            has_body_photos = "imagePaths" in state["user_profile"] and any(state["user_profile"].get("imagePaths", {}).values())
        
        # No body analysis in the main flow to avoid await issues, only in stream method
        # Instead, use standard profile analysis
        profile_prompt = f"""
        Analyze the user profile information and ensure it's complete for creating 
        personalized health and fitness plans. Current profile: {state["user_profile"]}
        
        Based on the provided information, create a comprehensive profile assessment including:
        1. An evaluation of their current fitness level
        2. Analysis of their fitness goals and how achievable they are
        3. How their dietary preferences align with their goals
        4. Recommendations for their fitness journey
        
        Provide a professional and encouraging profile assessment based on this information.
        """
        
        # Use synchronous invoke for the __call__ method
        response = self.llm.invoke(profile_prompt)
        
        # Store the response in user_profile for downstream agents
        state["user_profile_data"] = state["user_profile"]
        state["structured_user_profile"] = state["user_profile"]  # Keep original data
        
        # Send structured response
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

class QueryAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.5,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    
    async def stream(self, state: WorkoutState) -> AsyncGenerator[str, None]:
        """Stream response to query"""
        if not state["current_query"]:
            return
        
        # Simple query handling without routing logic
        query_prompt = f"""
        User Query: {state["current_query"]}
        
        Context:
        Dietary Plan: {state["dietary_state"].content}
        Fitness Plan: {state["fitness_state"].content}
        User Profile: {state["user_profile"]}
        
        Provide a detailed, personalized response to the user's question.
        Ensure the answer is:
        - Specific to their health and fitness plan
        - Actionable
        - Based on the generated plans
        - Considers all relevant aspects of their fitness and dietary plans
        """
        
        async for chunk in self.llm.astream(query_prompt):
            if chunk.content:
                yield chunk.content

    def __call__(self, state: WorkoutState) -> WorkoutState:
        """Handle user queries about their fitness and dietary plans"""
        if not state["current_query"]:
            return state
        
        # Simple query handling without routing logic
        query_prompt = f"""
        User Query: {state["current_query"]}
        
        Context:
        Dietary Plan: {state["dietary_state"].content}
        Fitness Plan: {state["fitness_state"].content}
        User Profile: {state["user_profile"]}
        
        Provide a detailed, personalized response to the user's question.
        Ensure the answer is:
        - Specific to their health and fitness plan
        - Actionable
        - Based on the generated plans
        - Considers all relevant aspects of their fitness and dietary plans
        """
        
        response = self.llm.invoke(query_prompt)
        state["conversation_history"].append(
            {"role": "user", "content": state["current_query"]},
        )
        state["conversation_history"].append(
            {"role": "assistant", "content": response.content}
        )
        return state

class HeadCoachAgent:
    """Coordinates the fitness coaching process and produces the final structured output."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    
    async def stream(self, state: WorkoutState) -> AsyncGenerator[str, None]:
        """Stream comprehensive coaching plan"""
        # If complete_response is already generated, stream it directly
        if state.get("complete_response"):
            # Split by sections for more natural streaming
            sections = state["complete_response"].split("\n\n")
            for section in sections:
                yield section + "\n\n"
            return
        
        # If not already generated, create the structured output with all components
        yield "# Your Personalized Fitness Plan\n\n"
        
        # Add profile assessment
        yield "## Profile Assessment\n\n"
        yield state["user_profile"] if isinstance(state["user_profile"], str) else json.dumps(state["user_profile"], indent=2)
        yield "\n\n"
        
        # Include body analysis if available
        if state.get("body_analysis") and len(state.get("body_analysis", "")) > 50:
            yield "## Body Composition Analysis\n\n"
            yield state["body_analysis"]
            yield "\n\n"
        
        # Add dietary plan
        yield "## Dietary Plan\n\n"
        yield state["dietary_state"].content
        yield "\n\n"
        
        # Add fitness plan
        yield "## Fitness Plan\n\n"
        yield state["fitness_state"].content
        yield "\n\n"

        # Add progress comparison if available
        if state.get("progress_comparison"):
            yield "## Progress Comparison\n\n"
            yield state["progress_comparison"]
            yield "\n\n"
        
        # Final message
        yield "---\n\n"
        yield "Your personalized fitness and dietary plans have been created. You can now ask specific questions about your plans."
    
    def compare_responses(self, previous_overview, current_overview, user_profile):
        """Compare previous and current fitness overviews to identify progress"""
        if not previous_overview or not current_overview:
            return "This is your first fitness assessment. Future assessments will include progress tracking."
            
        comparison_prompt = f"""
        You are a fitness coach analyzing a client's progress over time.
        
        Compare the previous fitness assessment with the current one to identify changes and progress.
        Focus on significant changes in:
        1. Body statistics (weight, measurements, body fat percentage)
        2. Fitness level indicators
        3. Dietary improvements
        4. Workout performance
        
        Previous Assessment:
        {previous_overview}
        
        Current Assessment:
        {current_overview}
        
        Current user Profile used for the currect overview:
        {json.dumps(user_profile, indent=2)}
        
        Provide a concise, encouraging summary of their progress. Include:
        - Key improvements
        - Areas still needing focus
        - Specific metrics that have changed
        - Recommendations based on this progress
        
        Format the response in a motivating way that acknowledges achievements while encouraging continued effort.
        """
        
        # Use the LLM to generate the comparison
        response = self.llm.invoke(comparison_prompt)
        return response.content
    
    def __call__(self, state: WorkoutState) -> WorkoutState:
        """Produce the final structured output with all components"""
        # Ensure we preserve the original structured data
        if "original_user_profile" in state:
            user_profile_data = state["original_user_profile"]
        elif "user_profile_data" in state:
            user_profile_data = state["user_profile_data"]
        else:
            user_profile_data = {}
            
        if isinstance(user_profile_data, dict) and "image_timestamps" in user_profile_data:
            image_timestamps = user_profile_data["image_timestamps"]
        else:
            image_timestamps = {}
        
        # Add profile assessment
        complete_response = "## Profile Assessment\n\n"
        complete_response += state["user_profile"] if isinstance(state["user_profile"], str) else json.dumps(state["user_profile"], indent=2)
        
        # Include body analysis if available
        if state.get("body_analysis") and len(state.get("body_analysis", "")) > 50:
            complete_response += "\n\n## Body Composition Analysis\n\n"
            complete_response += state["body_analysis"]
        
        # Add dietary plan
        complete_response += "\n\n## Dietary Plan\n\n"
        complete_response += state["dietary_state"].content
        
        # Add fitness plan
        complete_response += "\n\n## Fitness Plan\n\n"
        complete_response += state["fitness_state"].content
        
        try:
            # Try to get previous complete_response from the same thread_id
            # This should be done by the calling code that has access to the app instance
            # and can use app.get_state_history()
            if "previous_complete_response" in state and state["previous_complete_response"]:
                previous_response = state["previous_complete_response"]
                
                # Generate progress comparison
                progress_comparison = self.compare_responses(
                    previous_response,
                    complete_response,
                    user_profile_data
                )
                
                # Add progress comparison to complete response
                if progress_comparison:
                    complete_response += "\n\n## Progress Comparison\n\n"
                    complete_response += progress_comparison
                    state["progress_comparison"] = progress_comparison
            else:
                # No previous response available
                state["progress_comparison"] = "This is your first fitness assessment. Future assessments will include progress tracking."
                complete_response += "\n\n## Progress Comparison\n\n"
                complete_response += state["progress_comparison"]
        except Exception as e:
            print(f"Error generating progress comparison: {str(e)}")
            # Continue without the comparison
        
        # Final message
        complete_response += "\n\n---\n\n"
        complete_response += "Your personalized fitness and dietary plans have been created. You can now ask specific questions about your plans."
        
        # Store the complete response and original user profile data
        state["complete_response"] = complete_response
        state["user_profile_data"] = user_profile_data
        
        return state 