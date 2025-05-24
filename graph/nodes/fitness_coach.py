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
from langchain_core.messages import HumanMessage, SystemMessage
import json
from dotenv import load_dotenv
# Add Anthropic imports
import anthropic
from anthropic import Anthropic
# Add stream writer for custom streaming
from langgraph.config import get_stream_writer

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
        
        # Initialize Anthropic client for vision analysis
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            print("Warning: ANTHROPIC_API_KEY not found in environment variables")
            # Fall back to OpenAI if no Anthropic key is available
            self.use_claude = False
            # Vision model for body composition analysis (fallback to OpenAI)
            self.vision_model = ChatOpenAI(
                model="gpt-4o",
                temperature=0.1,
                max_tokens=4096,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()]
            )
        else:
            self.use_claude = True
            self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
            self.claude_model = "claude-3-sonnet-20240229"
            print("Using Claude for vision analysis")
        
        # Get the Supabase service key and URL from environment variables
        self.supabase_service_key = os.environ.get("SUPABASE_SERVICE_KEY")
        self.supabase_url = os.environ.get("SUPABASE_URL", "https://hjzszgaugorgqbbsuqgc.supabase.co")
        
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
                elif image_url.lower().endswith('.webp'):
                    mime_type = "image/webp"
                elif "png" in image_url.lower():
                    mime_type = "image/png"
                else:
                    mime_type = "image/jpeg"  # Default to JPEG
            
            # Convert image to base64
            image_data = base64.b64encode(response.content).decode('utf-8')
            
            if self.use_claude:
                # For Claude, we need just the base64 string and media type, not a data URL
                print(f"Successfully encoded image as base64 ({mime_type}, {len(image_data)} chars)")
                return {"data": image_data, "media_type": mime_type}
            else:
                # For OpenAI, we need a data URL
                data_url = f"data:{mime_type};base64,{image_data}"
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
                if self.use_claude:
                    # Format for Claude
                    formatted_images.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": base64_image["media_type"],
                            "data": base64_image["data"]
                        }
                    })
                else:
                    # Format for OpenAI
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
                if self.use_claude:
                    # Print for Claude format
                    print(f"  Image {i+1}: type={img['type']}, media_type={img['source']['media_type']}")
                else:
                    # Print for OpenAI format
                    is_base64 = isinstance(img['image_url']['url'], str) and img['image_url']['url'].startswith('data:')
                    url_preview = "base64 data" if is_base64 else img['image_url']['url']
                    print(f"  Image {i+1}: type={img['type']}, format={url_preview[:30]}...")
        
        # Store timestamps in user profile for later use
        user_profile["image_timestamps"] = image_timestamps
        
        return formatted_images
    
    async def _analyze_body_composition(self, user_profile):
        """Analyze body composition using a vision model and user input - now streams tokens"""
        print("Starting body composition analysis...")
        
        # Exit early if no imagePaths found
        image_paths = user_profile.get("imagePaths", {})
        if not image_paths or not any(image_paths.values()):
            print("No body photos found in profile, cannot analyze body composition")
            #yield "No body photos found for analysis."
            return  # Just return without value
            
        # Format the images for API call
        formatted_images = self._format_body_photos(user_profile)
        
        if not formatted_images:
            print("No valid body photos could be processed, skipping analysis")
            #yield "Unable to process body photos for analysis."
            return  # Just return without value
            
        print(f"Formatted {len(formatted_images)} images for analysis")
        
        # Create analysis prompt based on user information
        prompt_text = f"""
            [Persona]
            You are a high-skilled professional fitness coach and nutritionist analyzing body composition photos of clients.
            
            [Task]
            Your task is to analyze the provided body photos of a client, which show the person from different angles.
            You are given a list of images of the client, and you need to analyze each image carefully and provide your findings.

User details:
- Age: {user_profile.get('age', 'unknown')}
- Gender: {user_profile.get('gender', 'unknown')}
- Height: {user_profile.get('height', 'unknown')}
- Weight: {user_profile.get('weight', 'unknown')}
- Fitness goals: {user_profile.get('fitness_goals', 'unknown')}

            [Instructions]
            1. Begin your analysis with the section header "## Body Composition Analysis"
            
            2. Analyze each photo carefully. First describe what you see in a neutral, professional way.
            
            3. Then estimate and present in bold:
               - **Approximate Body Fat: 15-20%** (use this format)
               - **BMI: 24.3** (if you can calculate it)
               - Any other relevant measurements

            4. Analyze and report on:
               - Current muscle mass regions (areas with good development vs. needs improvement)
               - Posture observations 
               - Apparent imbalances

            5. Summarize your assessment and provide objective recommendations to the client based on these visual observations combined with the user's stated goals, be brutally honest.
            
            6. Formatting guidelines:
               - Use "## Body Composition Analysis" as the ONLY level 2 (H2) heading in your response
               - Use level 3 headings (###) for all subsections such as "### Measurements" or "### Muscle Assessment"
               - NEVER use level 2 headings (##) for any subsection - only for the main section title
               - Format all measurements and calculations in bold (e.g., **Body Fat: 18%**)
               - Use standard markdown for lists and formatting
               - Separate major sections with a blank line
               - Keep your analysis concise and easy to read
               - Always end your response with "---" (three dashes) as a separator, followed by two blank lines before the next section title.
"""
        
        # Accumulated analysis text
        analysis = ""
        
        if self.use_claude:
            # For Claude API
            # Create content array for Claude
            content = [
                {"type": "text", "text": prompt_text}
            ]
            
            # Add each formatted image to the content array
            for image in formatted_images:
                content.append(image)
            
            # Add additional instructions at the end
            content.append({
                "type": "text", 
                "text": "Based on all images shown, provide a complete analysis with percentage estimates. Be professional and brutally honest in your assessment, while remaining encouraging."
            })
            
            # Print the structure of content for debugging
            debug_content = []
            for item in content:
                if item["type"] == "text":
                    debug_content.append({"type": "text", "text": item["text"][:50] + "..."})
                else:
                    debug_content.append({"type": "image", "source": {"type": "base64", "media_type": item["source"]["media_type"]}})
            
            print(f"Content has {len(content)} items: {len([i for i in content if i['type'] == 'text'])} text and {len([i for i in content if i['type'] == 'image'])} images")
            print(f"Message content structure: {json.dumps(debug_content, indent=2)}")
            
            try:
                # Format the message for Claude's Messages API
                messages = [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
                
                print(f"Sending request to Claude vision model with {len(messages)} messages")
                
                # Stream the response from Claude
                stream = self.anthropic_client.messages.create(
                    model=self.claude_model,
                    max_tokens=1024,
                    messages=messages,
                    stream=True
                )
                for chunk in stream:
                    print("CHUNK:", chunk)
                    # Skip tuples or non-content objects
                    if isinstance(chunk, tuple):
                        continue
                    if hasattr(chunk, "type") and chunk.type == "content_block_delta":
                        if hasattr(chunk, "delta") and hasattr(chunk.delta, "type") and chunk.delta.type == "text_delta":
                            token = chunk.delta.text
                            if token:
                                analysis += token
                                yield token
                
                # Print a sample of the response for debugging
                print(f"Analysis response received, {len(analysis)} chars")
                print(f"Sample of analysis: {analysis[:150]}...")
                
                # Don't return analysis - just end the generator
                return
            except Exception as e:
                print(f"Error during body composition analysis with Claude: {str(e)}")
                # Print more detailed error information
                import traceback
                print(f"Detailed error: {traceback.format_exc()}")
                #yield f"Error analyzing body photos: {str(e)}"
                return
        else:
            # For OpenAI API (existing code)
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
                "text": "Based on all images shown, provide a complete analysis with percentage estimates. Be professional and brutally honest in your assessment, while remaining encouraging."
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
                
                # Stream the response from OpenAI
                async for chunk in self.vision_model.astream(messages):
                    token = chunk.content
                    if token:
                        analysis += token
                        yield token
                
                # Print a sample of the response for debugging
                print(f"Analysis response received, {len(analysis)} chars")
                print(f"Sample of analysis: {analysis[:150]}...")
                
                # Don't return analysis - just end the generator
                return
            except Exception as e:
                print(f"Error during body analysis streaming: {str(e)}")
                # Print more detailed error information
                import traceback
                print(f"Detailed error: {traceback.format_exc()}")
                #yield f"Error analyzing body photos: {str(e)}"
                return

    async def stream(self, state: WorkoutState) -> AsyncGenerator[str, None]:
        """Stream the profile analysis"""
        # Get stream writer for emitting events
        writer = get_stream_writer()
        
        # Emit start step
        writer({"type": "step", "content": "Starting profile analysis..."})
        
        if not state["user_profile"]:
            state["user_profile"] = UserProfile().dict()
        
        # Check if we have body photos for analysis
        has_body_photos = False
        if isinstance(state["user_profile"], dict):
            has_body_photos = "imagePaths" in state["user_profile"] and any(state["user_profile"].get("imagePaths", {}).values())
        
        # Check if we have previous profile data
        previous_profile_assessment = None
        previous_body_analysis = None
        if state.get("previous_sections"):
            previous_profile_assessment = state["previous_sections"].get("profile_assessment", "")
            previous_body_analysis = state["previous_sections"].get("body_analysis", "")
        
        # Emit progress event
        writer({"type": "step", "content": "Analyzing user profile data..."})
        
        # Check if body analysis is already in the state (from streaming endpoint)
        body_analysis = state.get("body_analysis")
        
        if has_body_photos and not body_analysis:
            # If we have body photos but no analysis yet, perform and stream it
            # Emit body analysis step
            writer({"type": "step", "content": "Analyzing body composition from photos..."})
            
            # Stream the body composition analysis token by token
            body_analysis_content = ""
            async for token in self._analyze_body_composition(state["user_profile"]):
                if token:
                    body_analysis_content += token
                    # Stream each token
                    writer({"type": "profile", "content": token})
                    yield token
            
            # Store the complete body analysis in the state
            if body_analysis_content and len(body_analysis_content) > 50:
                state["body_analysis"] = body_analysis_content
                body_analysis = body_analysis_content
                
                # Signal completion
                writer({"type": "step", "content": "Body analysis complete. Generating profile assessment..."})
        elif body_analysis and has_body_photos:
            # Body analysis was already performed in the endpoint
            writer({"type": "step", "content": "Using previously generated body analysis for profile assessment..."})
            # We don't need to yield it again since it was already streamed from the endpoint
        
        # Prepare prompts with body analysis and previous data
        profile_prompt = f"""
            [Persona]
            You are an elite professional fitness coach and nutritionist with expertise in longitudinal fitness tracking. You specialize in creating personalized fitness assessments that track progress over time.

            [Task]
            Analyze the client's current profile data and body analysis, compare with any previous assessments, and create a comprehensive fitness profile that highlights changes, improvements, and areas needing attention.

            [Context]
            --- CURRENT CLIENT PROFILE ---
            {json.dumps(state["user_profile"])}

            --- CURRENT BODY ANALYSIS ---
            {body_analysis if body_analysis else "No body analysis available"}

            {f'''--- PREVIOUS ASSESSMENTS ---
            Previous Profile Assessment:
            {previous_profile_assessment}

            Previous Body Analysis:
            {previous_body_analysis}
            ''' if previous_profile_assessment else '--- NO PREVIOUS ASSESSMENTS AVAILABLE ---'}

            [Instructions]
            1. Create a clear, structured assessment with the following sections:
            - Current fitness status overview
            - Analysis of body composition and structure 
            - Alignment between client's goals and current physical state
            - Realistic timeframes for achieving stated goals
            - Key focus areas for improvement

            2. {f'''Progress Comparison Analysis:
            - Compare current measurements/stats with previous assessment
            - Identify specific improvements in body composition, posture, or muscle development
            - Quantify changes in weight, muscle mass, and body fat percentage when possible
            - Highlight areas showing good progress
            - Identify areas still requiring targeted work
            - Analyze if progress is aligned with previously stated goals
            ''' if previous_profile_assessment else 'Since this is the first assessment, establish clear baseline metrics for future comparison.'}

            3. Professional tone:
            - Be honest but encouraging
            - Use precise, measurable language
            - Avoid generic statements; be specific to this client
            - Balance constructive feedback with positive reinforcement

            4. Formatting guidelines:
            - Use "## Profile Assessment" as the main section header
            - Use only H3 ("###") headings for subsections 
            - NEVER use level 2 headings (##) for any subsection - only for the main section title
            - Format all measurements and calculations in bold (e.g., **Weight: 81kg**)
            - Use horizontal rules (---) to separate major sections
            - Use bullet points for lists of recommendations
            - Keep paragraphs concise and easy to read
            - Always end your response with "---" (three dashes) as a separator, followed by two blank lines before the next section title.

            Format your response as a cohesive professional assessment that the client can use as a roadmap for their fitness journey.
        """
        
        writer({"type": "step", "content": "Generating profile assessment..."})
        
        # Stream the profile assessment using LLM
        async for chunk in self.llm.astream(profile_prompt):
            if chunk.content:
                # Emit profile content for custom stream
                writer({"type": "profile", "content": chunk.content})
                yield chunk.content
                
        # If we have a body type from the analysis, store it in user_profile
        if body_analysis and "body_type" not in state["user_profile"] or not state["user_profile"]["body_type"]:
            # Try to extract body_type from the body analysis
            if "body type:" in body_analysis.lower():
                body_type_part = body_analysis.lower().split("body type:")[1].strip()
                body_type = body_type_part.split("\n")[0].strip()
                state["user_profile"]["body_type"] = body_type
            elif "somatotype:" in body_analysis.lower():
                body_type_part = body_analysis.lower().split("somatotype:")[1].strip()
                body_type = body_type_part.split("\n")[0].strip()
                state["user_profile"]["body_type"] = body_type

        # Signal completion
        writer({"type": "step", "content": "Profile assessment completed."})

    async def _call_with_body_analysis(self, state: WorkoutState) -> WorkoutState:
        """Process state with body analysis using vision model"""
        # Collect the body analysis tokens from the streaming generator
        full_body_analysis = ""
        async for token in self._analyze_body_composition(state["user_profile"]):
            if token:
                full_body_analysis += token
        
        # Update the state with body analysis
        if full_body_analysis and len(full_body_analysis) > 50:
            state["body_analysis"] = full_body_analysis
        
            # Extract body type if present
            if "body type:" in full_body_analysis.lower():
                body_type_part = full_body_analysis.lower().split("body type:")[1].strip()
                body_type = body_type_part.split("\n")[0].strip()
                state["user_profile"]["body_type"] = body_type
            elif "somatotype:" in full_body_analysis.lower():
                body_type_part = full_body_analysis.lower().split("somatotype:")[1].strip()
                body_type = body_type_part.split("\n")[0].strip()
                state["user_profile"]["body_type"] = body_type
        
        # Get previous data if available
        previous_profile_assessment = None
        previous_body_analysis = None
        if state.get("previous_sections"):
            previous_profile_assessment = state["previous_sections"].get("profile_assessment", "")
            previous_body_analysis = state["previous_sections"].get("body_analysis", "")
        
        # Create full profile analysis with previous data
        profile_prompt = f"""
            [Persona]
            You are an elite professional fitness coach and nutritionist with expertise in longitudinal fitness tracking. You specialize in creating personalized fitness assessments that track progress over time.

            [Task]
            Analyze the client's current profile data and body analysis, compare with any previous assessments, and create a comprehensive fitness profile that highlights changes, improvements, and areas needing attention.

            [Context]
            --- CURRENT CLIENT PROFILE ---
            {json.dumps(state["user_profile"])}

            --- CURRENT BODY ANALYSIS ---
            {full_body_analysis if full_body_analysis else "No body analysis available"}

            {f'''--- PREVIOUS ASSESSMENTS ---
            Previous Profile Assessment:
            {previous_profile_assessment}

            Previous Body Analysis:
            {previous_body_analysis}
            ''' if previous_profile_assessment else '--- NO PREVIOUS ASSESSMENTS AVAILABLE ---'}

            [Instructions]
            1. Create a clear, structured assessment with the following sections:
            - Current fitness status overview
            - Analysis of body composition and structure 
            - Alignment between client's goals and current physical state
            - Realistic timeframes for achieving stated goals
            - Key focus areas for improvement

            2. {f'''Progress Comparison Analysis:
            - Compare current measurements/stats with previous assessment
            - Identify specific improvements in body composition, posture, or muscle development
            - Quantify changes in weight, muscle mass, and body fat percentage when possible
            - Highlight areas showing good progress
            - Identify areas still requiring targeted work
            - Analyze if progress is aligned with previously stated goals
            ''' if previous_profile_assessment else 'Since this is the first assessment, establish clear baseline metrics for future comparison.'}

            3. Professional tone:
            - Be honest but encouraging
            - Use precise, measurable language
            - Avoid generic statements; be specific to this client
            - Balance constructive feedback with positive reinforcement

            4. Formatting guidelines:
            - Use "## Profile Assessment" as the main section header
            - Use only H3 ("###") headings for subsections
            - NEVER use level 2 headings (##) for any subsection - only for the main section title
            - Format all measurements and calculations in bold (e.g., **Weight: 81kg**)
            - Use horizontal rules (---) to separate major sections
            - Use bullet points for lists of recommendations
            - Keep paragraphs concise and easy to read
            - Always end your response with "---" (three dashes) as a separator, followed by two blank lines before the next section title.

            Format your response as a cohesive professional assessment that the client can use as a roadmap for their fitness journey.
        """
        
        response = await self.llm.ainvoke(profile_prompt)
        if full_body_analysis:
            state["user_profile"]["body_analysis"] = full_body_analysis
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
        
        # Get previous data if available
        previous_profile_assessment = None
        previous_body_analysis = None
        if state.get("previous_sections"):
            previous_profile_assessment = state["previous_sections"].get("profile_assessment", "")
            previous_body_analysis = state["previous_sections"].get("body_analysis", "")
        
        # Check if we have body analysis data in the state
        body_analysis = state.get("body_analysis")
    
        # If we have body photos but no analysis yet, we need to analyze
        # but _analyze_body_composition is now a streaming generator, 
        # so we use _call_with_body_analysis as a helper (which is synchronous)
        if has_body_photos and not body_analysis:
            # For synchronous contexts, use the helper method
            import asyncio
            try:
                # Use an event loop to run the async method
                loop = asyncio.get_event_loop()
                state = loop.run_until_complete(self._call_with_body_analysis(state))
                body_analysis = state.get("body_analysis")
            except Exception as e:
                print(f"Error in synchronous body analysis: {e}")
                # Continue without body analysis
        
        # Create profile analysis with or without body analysis
        profile_prompt = f"""
            [Persona]
            You are an elite professional fitness coach and nutritionist with expertise in longitudinal fitness tracking. You specialize in creating personalized fitness assessments that track progress over time.

            [Task]
            Analyze the client's current profile data{", including body analysis," if body_analysis else ""} compare with any previous assessments, and create a comprehensive fitness profile that highlights changes, improvements, and areas needing attention.

            [Context]
            --- CURRENT CLIENT PROFILE ---
            {json.dumps(state["user_profile"])}
            
            {f'''--- CURRENT BODY ANALYSIS ---
            {body_analysis}
            ''' if body_analysis else '--- NO BODY ANALYSIS AVAILABLE ---'}

            {f'''--- PREVIOUS ASSESSMENTS ---
            Previous Profile Assessment:
            {previous_profile_assessment}

            Previous Body Analysis:
            {previous_body_analysis}
            ''' if previous_profile_assessment else '--- NO PREVIOUS ASSESSMENTS AVAILABLE ---'}

            [Instructions]
            1. Begin your assessment with the section header "## Profile Assessment"
            
            2. Create a structured assessment with these subsections:
               - Current fitness status overview based on provided metrics
               - {("Analysis of body composition and structure based on the provided analysis" if body_analysis else "Analysis of client's fitness status based on provided profile information")}
               - Alignment between client's goals and current physical state
               - Realistic timeframes for achieving stated goals
               - Key focus areas for improvement

            3. {f'''Progress Comparison Analysis:
               - Compare current measurements/stats with previous assessment
               - {("Identify specific improvements in body composition, posture, or muscle development" if body_analysis else "Identify changes in weight, activity level, or other metrics")}
               - {("Quantify changes in weight, muscle mass, and body fat percentage when possible" if body_analysis else "Evaluate progress based on available information")}
               - Highlight areas showing good progress
               - Identify areas still requiring targeted work
               - Analyze if progress is aligned with previously stated goals
            ''' if previous_profile_assessment else 'Since this is the first assessment, establish clear baseline metrics for future comparison.'}

            4. Professional tone:
               - Be honest but encouraging
               - Use precise, measurable language
               - Avoid generic statements; be specific to this client
               - Balance constructive feedback with positive reinforcement

            5. Formatting guidelines:
               - Use "## Profile Assessment" as the ONLY level 2 (H2) heading in your response
               - Use level 3 headings (###) for all subsections (e.g., "### Current Fitness Status", "### Goal Alignment")
               - NEVER use level 2 headings (##) for any subsection - only for the main section title
               - Format all measurements and calculations in bold (e.g., **Weight: 81kg**)
               - Use horizontal rules (---) to separate major sections
               - Use bullet points for lists of recommendations
               - Keep paragraphs concise and easy to read
               - Always end your response with "---" (three dashes) as a separator, followed by two blank lines before the next section title.

            Format your response as a comprehensive professional assessment that the client can use as a roadmap for their fitness journey.
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
            model="gpt-4o",
            temperature=0.2,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    
    async def stream(self, state: WorkoutState) -> AsyncGenerator[str, None]:
        """Stream dietary recommendations"""
        # Get stream writer for emitting events
        writer = get_stream_writer()
        
        # Emit start step
        writer({"type": "step", "content": "Starting dietary plan generation..."})
        
        # Check if we have previous dietary plan and profile
        previous_dietary_plan = None
        previous_user_profile = None
        
        if state.get("previous_sections"):
            previous_dietary_plan = state["previous_sections"].get("dietary_plan", "")
        
        # Get previous structured profile if available
        if state.get("structured_user_profile") and state.get("previous_complete_response"):
            previous_user_profile = json.dumps(state.get("structured_user_profile"))
        
        diet_prompt = f"""
            [Persona]
            You are an elite nutritionist and dietary coach with expertise in personalized meal planning for fitness goals. You specialize in creating comprehensive nutrition plans that align with clients' training objectives.

            [Task]
            Create a detailed, personalized dietary plan based on the client's profile, body analysis, and fitness goals. If available, compare with previous dietary plans and profile data to ensure progression and appropriate adjustments.

            [Context]
            --- CURRENT CLIENT PROFILE ---
        {state["user_profile"]}
        
            {f'''--- PREVIOUS CLIENT PROFILE ---
            {previous_user_profile}
            ''' if previous_user_profile else ''}

            {f'''--- PREVIOUS DIETARY PLAN ---
            {previous_dietary_plan}
            ''' if previous_dietary_plan else '--- NO PREVIOUS DIETARY PLAN AVAILABLE ---'}

            [Instructions]
            1. Always start your response with 2 blank lines, then on a new line the heading "## Dietary Plan".
            
            2. Create a structured meal plan with these components:
            - Daily caloric and macronutrient targets based on client goals
            - 3+ breakfast options with nutritional breakdown
            - 3+ lunch options with nutritional breakdown
            - 3+ dinner options with nutritional breakdown
            - 2-3 healthy snack options
            - Hydration and electrolyte recommendations

            3. {f'''Profile and Plan Analysis:
            - Note any significant changes in the client's profile (weight, goals, activity level, etc.)
            - Analyze the effectiveness of the previous dietary plan in light of these changes
            - Identify which aspects should be maintained or modified based on profile changes
            - Recommend specific adjustments aligned with current goals and circumstances
            - Introduce new meal options for variety while maintaining nutritional integrity
            - Address any changed dietary preferences or restrictions
            ''' if previous_dietary_plan or previous_user_profile else 'Include clear guidance on meal timing and portion control for a first-time plan.'}

            4. Nutritional Insights:
            - Use "### Nutritional Insights" as a subsection heading
            - Explain how the plan supports specific fitness goals
            - Provide guidance on adaptation based on training intensity
            - Include information on key nutrients for recovery and performance
            - Offer practical tips for meal preparation and adherence

            5. Professional tone:
            - Use precise nutritional terminology
            - Provide specific portion sizes and measurements
            - Balance evidence-based recommendations with practical implementation
            - Be encouraging but realistic about dietary adherence
            - Use standard markdown for lists and formatting

            6. Formatting Guidelines:
            - Use "## Dietary Plan" as the ONLY level 2 (H2) heading in your response
            - Use level 3 headings (###) for all subsections (e.g., "### Meal Plan", "### Breakfast", "### Lunch", "### Profile and Plan Analysis")
            - NEVER use level 2 headings (##) for any subsection - only for the main section title
            - Include daily caloric and macronutrient targets based on client goals (use bold for calculations like **Protein = 150g**)
            - Format food items in bold: **Food Item Name**
            - Format macronutrient information in italics with this exact pattern: _Macros: Approximately 30g protein, 45g carbs, 15g fat_
            - Use horizontal rules (---) to separate major sections
            - Use bullet points for lists of recommendations
            - Always end your response with "---" (three dashes) as a separator, followed by two blank lines before the next section title.

            Format your response as a comprehensive dietary plan that supports the client's fitness journey and can be practically implemented in daily life.
        """
        
        state["dietary_state"].is_streaming = True
        state["dietary_state"].last_update = datetime.now().isoformat()
        
        # Emit progress update
        writer({"type": "step", "content": "Analyzing nutritional needs and preferences..."})
        
        # Stream the dietary recommendations
        async for chunk in self.llm.astream(diet_prompt):
            if chunk.content:
                # Emit dietary content for custom stream
                writer({"type": "dietary", "content": chunk.content})
                yield chunk.content
        
        state["dietary_state"].is_streaming = False
        
        # Signal completion
        writer({"type": "step", "content": "Dietary plan completed."})

    def __call__(self, state: WorkoutState) -> WorkoutState:
        """Generate a dietary plan based on user profile"""
        # Check if we have previous dietary plan and profile
        previous_dietary_plan = None
        previous_user_profile = None
        
        if state.get("previous_sections"):
            previous_dietary_plan = state["previous_sections"].get("dietary_plan", "")
        
        # Get previous structured profile if available
        if state.get("structured_user_profile") and state.get("previous_complete_response"):
            previous_user_profile = json.dumps(state.get("structured_user_profile"))
        
        diet_prompt = f"""
            [Persona]
            You are an elite nutritionist and dietary coach with expertise in personalized meal planning for fitness goals. You specialize in creating comprehensive nutrition plans that align with clients' training objectives.

            [Task]
            Create a detailed, personalized dietary plan based on the client's profile, body analysis, and fitness goals. If available, compare with previous dietary plans and profile data to ensure progression and appropriate adjustments.

            [Context]
            --- CURRENT CLIENT PROFILE ---
        {state["user_profile"]}
        
            {f'''--- PREVIOUS CLIENT PROFILE ---
            {previous_user_profile}
            ''' if previous_user_profile else ''}

            {f'''--- PREVIOUS DIETARY PLAN ---
            {previous_dietary_plan}
            ''' if previous_dietary_plan else '--- NO PREVIOUS DIETARY PLAN AVAILABLE ---'}

            [Instructions]
            1. Begin your response with the heading "## Dietary Plan"
            
            2. Create a structured meal plan with these components:
            - Daily caloric and macronutrient targets based on client goals
            - 3+ breakfast options with nutritional breakdown
            - 3+ lunch options with nutritional breakdown
            - 3+ dinner options with nutritional breakdown
            - 2-3 healthy snack options
            - Hydration and electrolyte recommendations

            3. {f'''Profile and Plan Analysis:
            - Note any significant changes in the client's profile (weight, goals, activity level, etc.)
            - Analyze the effectiveness of the previous dietary plan in light of these changes
            - Identify which aspects should be maintained or modified based on profile changes
            - Recommend specific adjustments aligned with current goals and circumstances
            - Introduce new meal options for variety while maintaining nutritional integrity
            - Address any changed dietary preferences or restrictions
            ''' if previous_dietary_plan or previous_user_profile else 'Include clear guidance on meal timing and portion control for a first-time plan.'}

            4. Nutritional Insights:
            - Use "### Nutritional Insights" as a subsection heading
            - Explain how the plan supports specific fitness goals
            - Provide guidance on adaptation based on training intensity
            - Include information on key nutrients for recovery and performance
            - Offer practical tips for meal preparation and adherence

            5. Professional tone:
            - Use precise nutritional terminology
            - Provide specific portion sizes and measurements
            - Balance evidence-based recommendations with practical implementation
            - Be encouraging but realistic about dietary adherence
            - Use standard markdown for lists and formatting

            6. Formatting Guidelines:
            - Use "## Dietary Plan" as the ONLY level 2 (H2) heading in your response
            - Use level 3 headings (###) for all subsections (e.g., "### Meal Plan", "### Breakfast", "### Lunch", "### Profile and Plan Analysis")
            - NEVER use level 2 headings (##) for any subsection - only for the main section title
            - Include daily caloric and macronutrient targets based on client goals (use bold for calculations like **Protein = 150g**)
            - Format food items in bold: **Food Item Name**
            - Format macronutrient information in italics with this exact pattern: _Macros: Approximately 30g protein, 45g carbs, 15g fat_
            - Use horizontal rules (---) to separate major sections
            - Use bullet points for lists of recommendations
            - Always end your response with "---" (three dashes) as a separator, followed by two blank lines before the next section title.

            Format your response as a comprehensive dietary plan that supports the client's fitness journey and can be practically implemented in daily life.
        """
        
        response = self.llm.invoke(diet_prompt)
        state["dietary_state"].content = response.content
        state["dietary_state"].last_update = datetime.now().isoformat()
        return state

class FitnessAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    
    async def stream(self, state: WorkoutState) -> AsyncGenerator[str, None]:
        """Stream fitness recommendations"""
        # Get stream writer for emitting events
        writer = get_stream_writer()
        
        # Emit start step
        writer({"type": "step", "content": "Starting fitness plan generation..."})
        
        # Check if we have previous fitness plan and profile
        previous_fitness_plan = None
        previous_user_profile = None
        
        if state.get("previous_sections"):
            previous_fitness_plan = state["previous_sections"].get("fitness_plan", "")
        
        # Get previous structured profile if available
        if state.get("structured_user_profile") and state.get("previous_complete_response"):
            previous_user_profile = json.dumps(state.get("structured_user_profile"))
        
        fitness_prompt = f"""
        [Persona]
        You are an elite fitness coach with expertise in periodization and progressive training programs. You specialize in creating personalized workout plans that adapt to clients' changing needs and level of progress.

        [Task]
        Design a comprehensive fitness program that addresses the client's specific goals, body composition, and fitness level. If previous data is available, build upon their progress with appropriate adaptations and progressions.

        [Context]
        --- CURRENT CLIENT PROFILE ---
        {state["user_profile"]}
        
        {f'''--- PREVIOUS CLIENT PROFILE ---
        {previous_user_profile}
        ''' if previous_user_profile else ''}

        {f'''--- PREVIOUS FITNESS PLAN ---
        {previous_fitness_plan}
        ''' if previous_fitness_plan else '--- NO PREVIOUS FITNESS PLAN AVAILABLE ---'}

        [Instructions]
        1. Begin your response with 2 blank lines, then on a new line the heading "## Fitness Plan".
        
        2. Create a structured fitness program with these components:
           - Detailed warm-up protocols (10-15 minutes)
           - Main workout plan with specific splits based on goals
           - Exercise selection with clear sets, reps, and intensity guidelines
           - Progressive overload strategy over 4-8 weeks
           - Recovery protocols and rest day recommendations
           - Cool-down and mobility work

        3. {f'''Program Progression Analysis:
           - Identify changes in the client's physical profile or goals
           - Evaluate which exercises produced the best results in the previous plan
           - Recommend appropriate progression in volume, intensity, or complexity
           - Introduce new exercise variations to prevent plateaus
           - Adjust training frequency or split based on progress and current capacity
           - Address any form corrections or technique improvements needed
        ''' if previous_fitness_plan or previous_user_profile else 'Include clear guidance on proper form and technique for beginners.'}

        4. Implementation Strategy:
           - Provide a weekly schedule template
           - Explain how to track progress (weights, reps, perceived exertion)
           - Include adaptation guidelines based on rate of progress
           - Offer alternatives for common equipment limitations
           - Address injury prevention specific to chosen exercises

        5. Professional tone:
           - Use precise exercise terminology
           - Provide specific measurements for progress tracking
           - Balance scientific principles with practical application
           - Be encouraging while emphasizing proper technique and safety

        6. Formatting Guidelines:
           - Use "## Fitness Plan" as the ONLY level 2 (H2) heading in your response
           - Use level 3 headings (###) for all subsections like "### Warm-Up Protocol", "### Main Workout Plan", etc.
           - NEVER use level 2 headings (##) for any subsection - only for the main section title
           - Format workout days in bold: **Day 1: Upper Body**
           - Use standard bullet points for exercises within each day
           - Format sets and reps consistently: "- Bench Press: 3 sets of 8-10 reps"
           - Format measurements and calculations in bold: **BMR = 1500 calories/day**
           - Use horizontal rules (---) to separate major sections
           - Include proper spacing between sections for readability
           - Always end your response with "---" (three dashes) as a separator, followed by two blank lines before the next section title.

        Format your response as a comprehensive fitness plan that supports the client's specific goals and can be realistically implemented with their available resources.
        """
        
        state["fitness_state"].is_streaming = True
        state["fitness_state"].last_update = datetime.now().isoformat()
        
        # Emit progress update
        writer({"type": "step", "content": "Analyzing fitness goals and creating personalized workout plan..."})
        
        # Stream the fitness recommendations
        async for chunk in self.llm.astream(fitness_prompt):
            if chunk.content:
                # Emit fitness content for custom stream
                writer({"type": "fitness", "content": chunk.content})
                yield chunk.content
        
        state["fitness_state"].is_streaming = False
        
        # Signal completion
        writer({"type": "step", "content": "Fitness plan completed."})

    def __call__(self, state: WorkoutState) -> WorkoutState:
        """Generate a fitness plan based on user profile"""
        # Check if we have previous fitness plan and profile
        previous_fitness_plan = None
        previous_user_profile = None
        
        if state.get("previous_sections"):
            previous_fitness_plan = state["previous_sections"].get("fitness_plan", "")
        
        # Get previous structured profile if available
        if state.get("structured_user_profile") and state.get("previous_complete_response"):
            previous_user_profile = json.dumps(state.get("structured_user_profile"))
        
        fitness_prompt = f"""
        [Persona]
        You are an elite fitness coach with expertise in periodization and progressive training programs. You specialize in creating personalized workout plans that adapt to clients' changing needs and level of progress.

        [Task]
        Design a comprehensive fitness program that addresses the client's specific goals, body composition, and fitness level. If previous data is available, build upon their progress with appropriate adaptations and progressions.

        [Context]
        --- CURRENT CLIENT PROFILE ---
        {state["user_profile"]}
        
        {f'''--- PREVIOUS CLIENT PROFILE ---
        {previous_user_profile}
        ''' if previous_user_profile else ''}

        {f'''--- PREVIOUS FITNESS PLAN ---
        {previous_fitness_plan}
        ''' if previous_fitness_plan else '--- NO PREVIOUS FITNESS PLAN AVAILABLE ---'}

        [Instructions]
        1. Begin your response with the heading "## Fitness Plan"
        
        2. Create a structured fitness program with these components:
           - Detailed warm-up protocols (10-15 minutes)
           - Main workout plan with specific splits based on goals
           - Exercise selection with clear sets, reps, and intensity guidelines
           - Progressive overload strategy over 4-8 weeks
           - Recovery protocols and rest day recommendations
           - Cool-down and mobility work

        3. {f'''Program Progression Analysis:
           - Identify changes in the client's physical profile or goals
           - Evaluate which exercises produced the best results in the previous plan
           - Recommend appropriate progression in volume, intensity, or complexity
           - Introduce new exercise variations to prevent plateaus
           - Adjust training frequency or split based on progress and current capacity
           - Address any form corrections or technique improvements needed
        ''' if previous_fitness_plan or previous_user_profile else 'Include clear guidance on proper form and technique for beginners.'}

        4. Implementation Strategy:
           - Provide a weekly schedule template
           - Explain how to track progress (weights, reps, perceived exertion)
           - Include adaptation guidelines based on rate of progress
           - Offer alternatives for common equipment limitations
           - Address injury prevention specific to chosen exercises

        5. Professional tone:
           - Use precise exercise terminology
           - Provide specific measurements for progress tracking
           - Balance scientific principles with practical application
           - Be encouraging while emphasizing proper technique and safety

        6. Formatting Guidelines:
           - Use "## Fitness Plan" as the ONLY level 2 (H2) heading in your response
           - Use level 3 headings (###) for all subsections like "### Warm-Up Protocol", "### Main Workout Plan", etc.
           - NEVER use level 2 headings (##) for any subsection - only for the main section title
           - Format workout days in bold: **Day 1: Upper Body**
           - Use standard bullet points for exercises within each day
           - Format sets and reps consistently: "- Bench Press: 3 sets of 8-10 reps"
           - Format measurements and calculations in bold: **BMR = 1500 calories/day**
           - Use horizontal rules (---) to separate major sections
           - Include proper spacing between sections for readability
           - Always end your response with "---" (three dashes) as a separator, followed by two blank lines before the next section title.

        Format your response as a comprehensive fitness plan that supports the client's specific goals and can be realistically implemented with their available resources.
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
        # Get stream writer for emitting events
        writer = get_stream_writer()
        
        if not state["current_query"]:
            return
        
        # Emit start of query processing
        writer({"type": "step", "content": f"Processing query: {state['current_query']}"})
        
        # Get previous sections if available
        previous_sections = state.get("previous_sections", {})
        
        # Simple query handling without routing logic
        query_prompt = f"""
        User Query: {state["current_query"]}
        
        Context:
        Dietary Plan: {state["dietary_state"].content}
        Fitness Plan: {state["fitness_state"].content}
        User Profile: {state["user_profile"]}
        
        {f'''Previous Data (for reference):
        Previous Profile Assessment: {previous_sections.get("profile_assessment", "")}
        Previous Dietary Plan: {previous_sections.get("dietary_plan", "")}
        Previous Fitness Plan: {previous_sections.get("fitness_plan", "")}
        ''' if previous_sections else ''}
        
        Provide a detailed, personalized response to the user's question.
        Ensure the answer is:
        - Specific to their health and fitness plan
        - Actionable
        - Based on the generated plans
        - Considers all relevant aspects of their fitness and dietary plans
        
        {f'''If the user's question relates to changes over time or progress:
        - Compare their previous and current plans
        - Highlight improvements or changes
        - Explain how their approach has evolved based on their progress
        ''' if previous_sections else ''}
        """
        
        # Emit analyzing step
        writer({"type": "step", "content": "Analyzing query against your personalized plans..."})
        
        # Stream the response
        async for chunk in self.llm.astream(query_prompt):
            if chunk.content:
                # Emit response content 
                writer({"type": "response", "content": chunk.content})
                yield chunk.content
        
        # Signal completion
        writer({"type": "step", "content": "Query processing completed."})

    def __call__(self, state: WorkoutState) -> WorkoutState:
        """Handle user queries about their fitness and dietary plans"""
        if not state["current_query"]:
            return state
        
        # Get previous sections if available
        previous_sections = state.get("previous_sections", {})
        
        # Simple query handling without routing logic
        query_prompt = f"""
        User Query: {state["current_query"]}
        
        Context:
        Dietary Plan: {state["dietary_state"].content}
        Fitness Plan: {state["fitness_state"].content}
        User Profile: {state["user_profile"]}
        
        {f'''Previous Data (for reference):
        Previous Profile Assessment: {previous_sections.get("profile_assessment", "")}
        Previous Dietary Plan: {previous_sections.get("dietary_plan", "")}
        Previous Fitness Plan: {previous_sections.get("fitness_plan", "")}
        ''' if previous_sections else ''}
        
        Provide a detailed, personalized response to the user's question.
        Ensure the answer is:
        - Specific to their health and fitness plan
        - Actionable
        - Based on the generated plans
        - Considers all relevant aspects of their fitness and dietary plans
        
        {f'''If the user's question relates to changes over time or progress:
        - Compare their previous and current plans
        - Highlight improvements or changes
        - Explain how their approach has evolved based on their progress
        ''' if previous_sections else ''}
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
        # Get stream writer for emitting events
        writer = get_stream_writer()
        
        # If complete_response is already generated, stream it directly
        if state.get("complete_response"):
            # Split by sections for more natural streaming
            sections = state["complete_response"].split("\n\n")
            for section in sections:
                writer({"type": "response", "content": section + "\n\n"})
                yield section + "\n\n"
            return
        
        # If not already generated, create the structured output with all components
        writer({"type": "response", "content": "# Your Personalized Fitness Plan\n\n"})
        yield "# Your Personalized Fitness Plan\n\n"
        
        # Add profile assessment (no need to add header, it's in the content)
        profile_content = state["user_profile"] if isinstance(state["user_profile"], str) else json.dumps(state["user_profile"], indent=2)
        writer({"type": "profile", "content": profile_content})
        yield profile_content
        yield "\n\n"
        
        # Include body analysis if available (no need to add header, it's in the content)
        if state.get("body_analysis") and len(state.get("body_analysis", "")) > 50:
            writer({"type": "profile", "content": state["body_analysis"]})
            yield state["body_analysis"]
            yield "\n\n"
        
        # Add dietary plan (no need to add the title since it's in the content)
        writer({"type": "dietary", "content": state["dietary_state"].content})
        yield state["dietary_state"].content
        yield "\n\n"
        
        # Add fitness plan (no need to add the title since it's in the content)
        writer({"type": "fitness", "content": state["fitness_state"].content})
        yield state["fitness_state"].content
        yield "\n\n"

        # Add progress comparison if available
        if state.get("progress_comparison"):
            writer({"type": "progress", "content": state["progress_comparison"]})
            yield state["progress_comparison"]
            yield "\n\n"
        
        # Final message
        final_message = "---\n\nYour personalized fitness and dietary plans have been created. You can now ask specific questions about your plans."
        writer({"type": "response", "content": final_message})
        yield final_message

    async def compare_responses(self, previous_overview, current_overview, user_profile):
        """Compare previous and current fitness overviews to identify progress, streaming token by token."""
        if not previous_overview or not current_overview:
            yield "## Progress Tracking\n\n"
            yield "This is your first fitness assessment. Future assessments will include progress tracking."
            return
        comparison_prompt = f"""
        [Persona]
        You are an elite fitness coach who specializes in analyzing client progress over time. You have exceptional abilities in recognizing changes in fitness metrics, body composition, dietary habits, and workout performance.

        [Task]
        Compare and analyze the client's previous fitness assessment with their current assessment and create a detailed progress comparison that highlights improvements, changes, and areas that still need work.

        [Context]
        --- PREVIOUS ASSESSMENT ---
        {previous_overview}
        
        --- CURRENT ASSESSMENT ---
        {current_overview}
        
        --- USER PROFILE ---
        {json.dumps(user_profile) if isinstance(user_profile, dict) else user_profile}

        [Instructions]
        
        1. Always start your response with 2 blank lines, then on a new line the heading "## Progress Tracking".

        2. Compare these specific aspects between the two assessments:
           - Body metrics (weight, BMI, body fat percentage)
           - Dietary compliance and nutrition habits
           - Workout performance and consistency
           - Progress toward stated fitness goals
           - Changes in recommendations or approach

        3. Highlight improvements:
           - Format specific improvements in bold (e.g., **Weight decreased by 2kg**)
           - Quantify progress whenever possible (e.g., "Increased workout frequency from 2 to 4 days per week")
           - Acknowledge both major and minor positive changes

        4. Identify remaining challenges:
           - Areas where progress has been slow or nonexistent
           - New issues that have emerged since the last assessment
           - Potential barriers to further improvement

        5. Provide context for the progress:
           - Whether progress is aligned with expected timeline
           - If the rate of improvement is appropriate for the client's situation
           - How the progress compares to typical results

        6. Formatting Guidelines:
        - Use "## Progress Tracking" as the ONLY level 2 (H2) heading in your response
        - Use level 3 headings (###) for all other (sub)sections.
        - NEVER use level 2 headings (##) for any subsection - only for the main section title
        - Use consistent formatting for all sections.
        - Use ### for all section titles and #### for any subsections if needed.
        - Group your analysis into 3-5 main sections with clear H3 headings.
        - Always end your response with "---" (three dashes) as a separator

        Format your response as an insightful progress analysis that motivates the client while providing an honest assessment of their fitness journey.
        """
        print("\n\n==========================================")
        print(f"Generating progress comparison")
        print(f"Previous overview length: {len(previous_overview)}")
        print(f"Current overview length: {len(current_overview)}")
        print("==========================================\n\n")
        async for chunk in self.llm.astream(comparison_prompt):
            if chunk.content:
                yield chunk.content
        print("\n\n==========================================")
        print(f"Completed progress comparison streaming.")
        print("==========================================\n\n")
    
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
        
        # Start building the complete response
        complete_response = ""
        
        # First, generate progress comparison if we have previous data
        progress_comparison = None
        try:
            # Use the complete previous response for comparison - this is more reliable than parsed sections
            if "previous_complete_response" in state and state["previous_complete_response"]:
                print("\n\n==========================================")
                print(f"Generating progress comparison")
                print(f"Previous overview length: {len(state['previous_complete_response'])}")
                
                # Build current overview for comparison
                current_overview = ""
                
                # Add profile assessment
                current_overview += state["user_profile"] if isinstance(state["user_profile"], str) else json.dumps(state["user_profile"], indent=2) + "\n\n"
                
                # Add body analysis if available
                if state.get("body_analysis") and len(state.get("body_analysis", "")) > 50:
                    current_overview += state["body_analysis"] + "\n\n"
                
                # Add dietary plan
                current_overview += state["dietary_state"].content + "\n\n"
                
                # Add fitness plan
                current_overview += state["fitness_state"].content + "\n\n"
                
                print(f"Current overview length: {len(current_overview)}")
                print("==========================================\n\n")
                
                # Generate progress comparison using complete previous response
                progress_comparison = self.compare_responses(
                    state["previous_complete_response"],
                    current_overview,
                    user_profile_data
                )
                
                print(f"\n\n==========================================")
                print(f"Completed progress comparison. Result length: {len(progress_comparison)}")
                print(f"==========================================\n\n")
            
            # Add progress comparison at the beginning if available
            if progress_comparison:
                print("\n\n==========================================")
                print(f"Adding progress comparison to the beginning (length: {len(progress_comparison)})")
                print("==========================================\n\n")
                complete_response += progress_comparison
                complete_response += "\n\n---\n\n"
                state["progress_comparison"] = progress_comparison
            else:
                # No previous response available
                print("\n\n==========================================")
                print("No previous response data found for comparison")
                print("==========================================\n\n")
                state["progress_comparison"] = "This is your first fitness assessment. Future assessments will include progress tracking."
                complete_response += "## Progress Tracking\n\n"
                complete_response += state["progress_comparison"]
                complete_response += "\n\n---\n\n"
        except Exception as e:
            print("\n\n==========================================")
            print(f"Error generating progress comparison: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            print("==========================================\n\n")
            # Continue without the comparison
            complete_response += "# Your Personalized Fitness Plan\n\n"
        
        # Now add the rest of the sections (no need to add headers, they're in the content)
        complete_response += state["user_profile"] if isinstance(state["user_profile"], str) else json.dumps(state["user_profile"], indent=2)
        
        # Include body analysis if available
        if state.get("body_analysis") and len(state.get("body_analysis", "")) > 50:
            complete_response += "\n\n"
            complete_response += state["body_analysis"]
        
        # Add dietary plan (no need to add the title since it's in the content)
        complete_response += "\n\n"
        complete_response += state["dietary_state"].content
        
        # Add fitness plan (no need to add the title since it's in the content)
        complete_response += "\n\n"
        complete_response += state["fitness_state"].content
        
        # Final message
        complete_response += "\n\n---\n\n"
        complete_response += "Your personalized fitness and dietary plans have been created. You can now ask specific questions about your plans."
        
        # Store the complete response and original user profile data
        state["complete_response"] = complete_response
        state["user_profile_data"] = user_profile_data
        
        return state 