#!/usr/bin/env python3
import os
import json
import time
import random
import logging
import argparse
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound, 
    TranscriptsDisabled
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transcript_extraction.log"),
        logging.StreamHandler()
    ]
)

def get_transcript(video_id, languages=['en']):
    """
    Get transcript for a video with fallback to other languages if needed.
    
    Args:
        video_id (str): YouTube video ID
        languages (list): List of language codes to try, in order of preference
        
    Returns:
        str or None: Transcript text if found, None otherwise
    """
    try:
        # First try with specified languages
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        transcript_text = " ".join([item["text"] for item in transcript_list])
        return transcript_text, languages[0]
    except NoTranscriptFound:
        # If no transcript in requested languages, try to find available languages
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to get any available transcript and translate it to English if possible
            for transcript in transcript_list:
                try:
                    # First try to get an English translation if available
                    if transcript.is_translatable:
                        translated = transcript.translate('en')
                        translated_list = translated.fetch()
                        transcript_text = " ".join([item["text"] for item in translated_list])
                        return transcript_text, f"translated-{transcript.language_code}-en"
                    
                    # Otherwise get the original transcript
                    original_list = transcript.fetch()
                    transcript_text = " ".join([item["text"] for item in original_list])
                    return transcript_text, transcript.language_code
                except Exception as e:
                    logging.warning(f"Error getting transcript for {video_id} in language {transcript.language_code}: {str(e)}")
                    continue
                    
            # If we got here, no transcript could be retrieved
            return None, None
            
        except (TranscriptsDisabled, Exception) as e:
            logging.error(f"Error fetching transcript for video {video_id}: {str(e)}")
            return None, None
    except Exception as e:
        logging.error(f"Error fetching transcript for video {video_id}: {str(e)}")
        return None, None

def load_videos_metadata(input_file):
    """Load video metadata from JSON file"""
    try:
        with open(input_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading metadata from {input_file}: {str(e)}")
        return []

def load_existing_transcripts(file_path):
    """Load existing transcripts if the file exists"""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading existing transcripts from {file_path}: {str(e)}")
            # If there's an error, create a backup of the potentially corrupted file
            backup_path = f"{file_path}.backup-{int(time.time())}"
            try:
                import shutil
                shutil.copy2(file_path, backup_path)
                logging.info(f"Created backup of transcript file at {backup_path}")
            except Exception as backup_error:
                logging.error(f"Failed to create backup: {str(backup_error)}")
    return []

def save_transcripts_progress(transcripts, output_file):
    """Save transcript extraction progress to file"""
    try:
        with open(output_file, 'w') as f:
            json.dump(transcripts, f, indent=2)
        logging.info(f"Saved {len(transcripts)} transcripts to {output_file}")
        return True
    except Exception as e:
        logging.error(f"Error saving transcripts to {output_file}: {str(e)}")
        # Create an emergency backup file if the main save fails
        try:
            emergency_file = f"{output_file}.emergency-{int(time.time())}.json"
            with open(emergency_file, 'w') as f:
                json.dump(transcripts, f, indent=2)
            logging.info(f"Created emergency save at {emergency_file}")
        except Exception as backup_error:
            logging.error(f"Emergency save also failed: {str(backup_error)}")
        return False

def extract_transcripts(
    video_data, 
    output_file, 
    save_interval=20,
    rate_limit_delay=2.0,
    languages=['en'],
    continue_from_existing=True
):
    """
    Extract transcripts for videos and save to JSON with periodic saving
    
    Args:
        video_data (list): List of video metadata dictionaries
        output_file (str): Path to output JSON file
        save_interval (int): Save progress after this many videos processed
        rate_limit_delay (float): Delay between API calls in seconds
        languages (list): List of language codes to try, in order of preference
        continue_from_existing (bool): If True, load existing transcripts and continue
    
    Returns:
        list: List of videos with transcripts
    """
    # Load existing transcripts if requested
    data_with_transcripts = []
    processed_video_ids = set()
    
    if continue_from_existing:
        data_with_transcripts = load_existing_transcripts(output_file)
        processed_video_ids = {item.get("video_id") for item in data_with_transcripts}
        logging.info(f"Loaded {len(data_with_transcripts)} existing transcripts, continuing extraction...")
    
    # Find videos that need processing
    videos_to_process = [v for v in video_data if v.get("video_id") not in processed_video_ids]
    logging.info(f"Found {len(videos_to_process)} videos that need transcript extraction")
    
    # Skip if nothing to process
    if not videos_to_process:
        logging.info("No new videos to process")
        return data_with_transcripts
    
    # Process each video
    language_stats = {}
    videos_with_transcripts = 0
    videos_without_transcripts = 0
    
    for i, video in enumerate(tqdm(videos_to_process, desc="Extracting transcripts")):
        video_id = video.get("video_id")
        if not video_id:
            continue
            
        transcript_text, lang_code = get_transcript(video_id, languages)
        
        if transcript_text:
            video_with_transcript = video.copy()
            video_with_transcript["transcript"] = transcript_text
            video_with_transcript["transcript_language"] = lang_code
            data_with_transcripts.append(video_with_transcript)
            videos_with_transcripts += 1
            
            # Update language statistics
            if lang_code in language_stats:
                language_stats[lang_code] += 1
            else:
                language_stats[lang_code] = 1
        else:
            # Still add the video but with an empty transcript to mark it as processed
            video_with_no_transcript = video.copy()
            video_with_no_transcript["transcript"] = ""
            video_with_no_transcript["transcript_language"] = None
            data_with_transcripts.append(video_with_no_transcript)
            videos_without_transcripts += 1
        
        # Save progress at intervals
        if (i + 1) % save_interval == 0:
            save_transcripts_progress(data_with_transcripts, output_file)
            logging.info(f"Progress: {i + 1}/{len(videos_to_process)} videos processed")
            
        # Sleep to avoid rate limiting
        time.sleep(random.uniform(rate_limit_delay * 0.8, rate_limit_delay * 1.2))
    
    # Final save
    save_transcripts_progress(data_with_transcripts, output_file)
    
    # Log summary
    logging.info(f"\nTranscript Extraction Summary:")
    logging.info(f"---------------------------")
    logging.info(f"Total processed: {len(data_with_transcripts)}")
    logging.info(f"Videos with transcripts: {videos_with_transcripts}")
    logging.info(f"Videos without transcripts: {videos_without_transcripts}")
    logging.info(f"Language statistics: {language_stats}")
    
    return data_with_transcripts

def main():
    parser = argparse.ArgumentParser(description="Extract YouTube transcripts from video metadata")
    parser.add_argument("--input", type=str, default="fitness_data/AthleanX_videos.json",
                      help="Input JSON file containing video metadata")
    parser.add_argument("--output", type=str, default="fitness_data/AthleanX_transcripts.json",
                      help="Output JSON file for transcripts")
    parser.add_argument("--save-interval", type=int, default=20,
                      help="Save progress after this many videos")
    parser.add_argument("--delay", type=float, default=2.0,
                      help="Delay between transcript requests in seconds")
    parser.add_argument("--languages", nargs="+", default=["en"],
                      help="Language codes to try, in order of preference")
    parser.add_argument("--no-continue", action="store_true",
                      help="Don't continue from existing transcript file")
    
    args = parser.parse_args()
    
    # Load video metadata
    video_data = load_videos_metadata(args.input)
    if not video_data:
        logging.error(f"No video metadata found in {args.input}")
        return
        
    logging.info(f"Loaded {len(video_data)} videos from {args.input}")
    
    # Extract transcripts
    data_with_transcripts = extract_transcripts(
        video_data,
        args.output,
        save_interval=args.save_interval,
        rate_limit_delay=args.delay,
        languages=args.languages,
        continue_from_existing=not args.no_continue
    )
    
    print(f"\nExtraction Complete:")
    print(f"-------------------")
    print(f"Total videos with transcripts: {len(data_with_transcripts)}")
    print(f"Transcript data saved to {args.output}")
    
    # Return non-empty transcripts count for statistics
    non_empty_transcripts = sum(1 for item in data_with_transcripts if item.get("transcript"))
    print(f"Videos with non-empty transcripts: {non_empty_transcripts}")
    print(f"Transcript coverage: {non_empty_transcripts/len(video_data)*100:.1f}%")

if __name__ == "__main__":
    main() 