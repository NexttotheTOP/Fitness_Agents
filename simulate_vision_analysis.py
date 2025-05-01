#!/usr/bin/env python
"""
Simulation file to test body composition analysis with Anthropic's Claude vision model.
This file replicates the functionality in ProfileAgent._analyze_body_composition
but uses Claude instead of OpenAI GPT-4o.
"""

import os
import re
import json
import base64
import requests
from datetime import datetime
from dotenv import load_dotenv
# Replace OpenAI imports with Anthropic
import anthropic
from anthropic import Anthropic

# Load environment variables
load_dotenv()

class VisionAnalysisSimulator:
    def __init__(self):
        # Initialize the Anthropic client instead of OpenAI
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable must be set")
        
        self.client = Anthropic(api_key=self.anthropic_api_key)
        
        # Model to use - Claude Sonnet has vision capabilities
        self.model = "claude-3-sonnet-20240229"
        
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
                
            # Check if image size is too large for API (typically 5MB limit for Claude)
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
            
            # Print sample of base64 data for debugging (first 20 chars)
            data_sample = image_data[:20] + "..." if len(image_data) > 20 else image_data
            print(f"Successfully encoded image as base64 ({mime_type}, {len(image_data)} chars)")
            
            # For Claude, we need just the base64 string and media type, not a data URL
            return {"data": image_data, "media_type": mime_type}
        except Exception as e:
            print(f"Error in _get_image_as_base64 for {image_url}: {str(e)}")
            return None
    
    def _format_body_photos(self, user_profile: dict) -> list:
        """Format body photos for Claude vision model input"""
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
                # Store the image data in the formatted images list
                # For Claude, we need to use its specific format
                formatted_images.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": base64_image["media_type"],
                        "data": base64_image["data"]
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
                # Print summary info about each image
                if img["source"]["type"] == "base64":
                    print(f"  Image {i+1}: type={img['type']}, media_type={img['source']['media_type']}")
        
        # Store timestamps in user profile for later use
        user_profile["image_timestamps"] = image_timestamps
        
        return formatted_images
    
    async def analyze_body_composition(self, user_profile):
        """Analyze body composition using Claude vision model and user input"""
        print("Starting body composition analysis with Claude...")
        
        # Exit early if no imagePaths found
        image_paths = user_profile.get("imagePaths", {})
        if not image_paths or not any(image_paths.values()):
            print("No body photos found in profile, cannot analyze body composition")
            return None

        # Format the images for Claude API call
        formatted_images = self._format_body_photos(user_profile)
        
        if not formatted_images:
            print("No valid body photos could be processed, skipping analysis")
            return None
            
        print(f"Formatted {len(formatted_images)} images for Claude API")
        
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
               - Always end your response with "---" (three dashes) as a separator
"""
        
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
        
        # Print the structure of content for debugging (without the actual image data)
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
            
            # Make the API call to Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=messages
            )
            
            # Extract the analysis from the response
            analysis = response.content[0].text
            
            # Print a sample of the response for debugging
            print(f"Analysis response received, {len(analysis)} chars")
            print(f"Sample of analysis: {analysis[:150]}...")
            
            return analysis
        except Exception as e:
            print(f"Error during body composition analysis with Claude: {str(e)}")
            # Print more detailed error information
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            return None

async def main():
    # Example user profile with image paths
    user_profile = {
        "user_id": "test_user_123",
        "thread_id": "44357faf-3358-4f89-b901-f55c36fda5b5",
        "age": 30,
        "gender": "male",
        "height": "180cm",
        "weight": "80kg",
        "activity_level": "extra",
        "fitness_goals": ["weight loss", "muscle gain"],
        "imagePaths": {
            "front": [
            "e8ead177-52a7-4309-a25e-fdf6d75776fa/1df2d6ed-196e-4960-8157-f81ad003c687/front-1746117495132-494817565_451423291385792_7182736232947609688_n.jpg"
            ],
            "side": [
            "e8ead177-52a7-4309-a25e-fdf6d75776fa/1df2d6ed-196e-4960-8157-f81ad003c687/side-1746117501075-494688360_627447523626838_4237786259470242201_n.jpg"
            ],
            "back": [
            "e8ead177-52a7-4309-a25e-fdf6d75776fa/1df2d6ed-196e-4960-8157-f81ad003c687/back-1746117506932-494818989_593550529693737_8992669121793502361_n.jpg"
            ]
        }
    }
    
    # To simulate with your actual images, replace with your real URLs or file paths
    # Example:
    # user_profile["imagePaths"] = {
    #     "front": ["your-actual-front-image-url.jpg"],
    #     "side": ["your-actual-side-image-url.jpg"],
    #     "back": ["your-actual-back-image-url.jpg"]
    # }
    
    simulator = VisionAnalysisSimulator()
    analysis_result = await simulator.analyze_body_composition(user_profile)
    
    if analysis_result:
        print("\n============== ANALYSIS RESULT ==============")
        print(analysis_result)
        print("============================================\n")
    else:
        print("\n============== ERROR ==============")
        print("No analysis result was generated")
        print("====================================\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 