from dotenv import load_dotenv
import os
import argparse
import json
from pytube import YouTube, Search
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
import time
import random
from tqdm import tqdm
import logging
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("youtube_collection.log"),
        logging.StreamHandler()
    ]
)

load_dotenv()

# Helper functions
def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    elif "youtube.com" in url:
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
    return None

def get_video_details(url):
    """Get video details using pytube"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        video_id = extract_video_id(url)
        if not video_id:
            return None
            
        # Use requests to get video info
        response = requests.get(f"https://www.youtube.com/watch?v={video_id}", headers=headers)
        if response.status_code != 200:
            logging.error(f"Error fetching video page: {response.status_code}")
            return None
            
        # Parse title and author with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('meta', property='og:title')
        author = soup.find('meta', property='og:site_name')
        
        return {
            "title": title.get('content', 'Unknown') if title else 'Unknown',
            "author": author.get('content', 'YouTube') if author else 'YouTube',
            "url": url,
            "video_id": video_id
        }
    except Exception as e:
        logging.error(f"Error fetching video details for {url}: {e}")
        return None

def get_transcript(video_id):
    """Get transcript using youtube_transcript_api"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([item["text"] for item in transcript_list])
        return transcript_text
    except Exception as e:
        logging.error(f"Error fetching transcript for video {video_id}: {e}")
        return None

def get_channel_videos_by_scraping(channel_url, limit=50):
    """Get videos from a YouTube channel by scraping the channel page"""
    videos = []
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Convert /c/ URLs to /channel/ URLs if needed
        if '/c/' in channel_url or '/user/' in channel_url:
            response = requests.get(channel_url, headers=headers)
            if response.status_code == 200:
                # Extract the channel ID from meta tags
                soup = BeautifulSoup(response.text, 'html.parser')
                channel_name = soup.find('meta', property='og:title')
                channel_name = channel_name.get('content', 'Unknown Channel') if channel_name else 'Unknown Channel'
                logging.info(f"Found channel: {channel_name}")
                
                # Find channel ID if available
                canonical = soup.find('link', rel='canonical')
                if canonical and 'channel/' in canonical['href']:
                    channel_url = canonical['href']
                    logging.info(f"Updated channel URL: {channel_url}")
            else:
                logging.error(f"Error accessing channel {channel_url}: {response.status_code}")
                return videos
                
        # Get the videos tab
        videos_url = f"{channel_url}/videos"
        response = requests.get(videos_url, headers=headers)
        
        if response.status_code != 200:
            logging.error(f"Error accessing videos for {channel_url}: {response.status_code}")
            return videos
            
        # Parse the page with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract video links from the page
        # This is a basic implementation that may need adjustments
        video_links = soup.find_all('a', href=re.compile(r'/watch\?v='))
        
        seen_videos = set()
        for link in video_links:
            href = link.get('href', '')
            if '/watch?v=' in href and 'list=' not in href:
                video_id = extract_video_id(f"https://youtube.com{href}")
                if video_id and video_id not in seen_videos:
                    seen_videos.add(video_id)
                    videos.append(f"https://youtube.com/watch?v={video_id}")
                    
                    if len(videos) >= limit:
                        break
                        
            # Sleep to avoid rate limiting
            time.sleep(random.uniform(0.1, 0.3))
            
        logging.info(f"Found {len(videos)} videos from {channel_url}")
        
    except Exception as e:
        logging.error(f"Error getting videos from channel {channel_url}: {e}")
    
    return videos

def get_playlist_videos_by_scraping(playlist_url, limit=50):
    """Get videos from a YouTube playlist by scraping the playlist page"""
    videos = []
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(playlist_url, headers=headers)
        
        if response.status_code != 200:
            logging.error(f"Error accessing playlist {playlist_url}: {response.status_code}")
            return videos
            
        # Parse the page with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract playlist title
        title = soup.find('meta', property='og:title')
        playlist_title = title.get('content', 'Unknown Playlist') if title else 'Unknown Playlist'
        logging.info(f"Found playlist: {playlist_title}")
        
        # Extract video links from the page
        video_links = soup.find_all('a', href=re.compile(r'/watch\?v='))
        
        seen_videos = set()
        for link in video_links:
            href = link.get('href', '')
            if '/watch?v=' in href:
                video_id = extract_video_id(f"https://youtube.com{href}")
                if video_id and video_id not in seen_videos:
                    seen_videos.add(video_id)
                    videos.append(f"https://youtube.com/watch?v={video_id}")
                    
                    if len(videos) >= limit:
                        break
                        
            # Sleep to avoid rate limiting
            time.sleep(random.uniform(0.1, 0.3))
            
        logging.info(f"Found {len(videos)} videos from playlist {playlist_title}")
        
    except Exception as e:
        logging.error(f"Error getting videos from playlist {playlist_url}: {e}")
    
    return videos

def search_youtube(query, limit=20):
    """Search YouTube for videos"""
    videos = []
    try:
        s = Search(query)
        logging.info(f"Searching for: {query}")
        
        # Collect videos more carefully
        for result in tqdm(s.results[:limit], desc=f"Searching for {query}"):
            try:
                video_url = result.watch_url
                if video_url and extract_video_id(video_url):
                    videos.append(video_url)
                # Sleep to avoid rate limiting
                time.sleep(random.uniform(0.5, 1.5))
            except Exception as inner_e:
                logging.warning(f"Error processing search result: {inner_e}")
                continue
            
    except Exception as e:
        logging.error(f"Error searching YouTube for {query}: {e}")
    
    return videos

def save_videos_metadata(videos, output_file="fitness_videos.json"):
    """Save video metadata to JSON file"""
    video_data = []
    
    for url in tqdm(videos, desc="Getting video details"):
        details = get_video_details(url)
        if details:
            video_data.append(details)
        # Sleep to avoid rate limiting
        time.sleep(random.uniform(0.5, 1.0))
    
    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(video_data, f, indent=2)
    
    logging.info(f"Saved {len(video_data)} video details to {output_file}")
    return video_data

def extract_transcripts(video_data, output_file="fitness_transcripts.json"):
    """Extract transcripts for videos and save to JSON"""
    data_with_transcripts = []
    
    for video in tqdm(video_data, desc="Extracting transcripts"):
        video_id = video.get("video_id")
        if not video_id:
            continue
            
        transcript = get_transcript(video_id)
        if transcript:
            video_with_transcript = video.copy()
            video_with_transcript["transcript"] = transcript
            data_with_transcripts.append(video_with_transcript)
        
        # Sleep to avoid rate limiting
        time.sleep(random.uniform(1.0, 2.0))
    
    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(data_with_transcripts, f, indent=2)
    
    logging.info(f"Saved {len(data_with_transcripts)} transcripts to {output_file}")
    return data_with_transcripts

# Main function
def main():
    parser = argparse.ArgumentParser(description="Collect YouTube fitness videos and transcripts")
    parser.add_argument("--channels", nargs="*", help="YouTube channel URLs to collect videos from")
    parser.add_argument("--playlists", nargs="*", help="YouTube playlist URLs to collect videos from")
    parser.add_argument("--search", nargs="*", help="Search terms to find YouTube videos")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of videos to collect per source")
    parser.add_argument("--output", type=str, default="fitness_data", help="Output directory for collected data")
    parser.add_argument("--max-videos", type=int, default=100, help="Maximum total number of videos to collect")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    collected_videos = []
    
    # Default creators if none specified
    default_channels = [
        "https://www.youtube.com/channel/UCe0TLA0EsQbE-MjuHXevj2A",  # AthleanX
        "https://www.youtube.com/channel/UC68TLK0mAEzUyHx5x5k-S1Q",  # Jeff Nippard
        "https://www.youtube.com/channel/UC1gyFDswGnNKVIhWQ3KIXnw"   # Renaissance Periodization
    ]
    
    default_searches = [
        "best workout for beginners",
        "how to build muscle scientific",
        "proper exercise form",
        "weight loss workout routine",
        "strength training program",
        "fitness nutrition guide",
        "workout recovery techniques"
    ]
    
    max_videos = args.max_videos
    
    # Get videos from channels
    if args.channels:
        for channel_url in args.channels:
            videos = get_channel_videos_by_scraping(channel_url, args.limit)
            collected_videos.extend(videos)
            logging.info(f"Collected {len(videos)} videos from channel {channel_url}")
            
            if len(collected_videos) >= max_videos:
                logging.info(f"Reached maximum video limit of {max_videos}")
                break
    elif not args.playlists and not args.search:
        for channel_url in default_channels:
            videos = get_channel_videos_by_scraping(channel_url, args.limit)
            collected_videos.extend(videos)
            logging.info(f"Collected {len(videos)} videos from default channel {channel_url}")
            
            if len(collected_videos) >= max_videos:
                logging.info(f"Reached maximum video limit of {max_videos}")
                break
    
    # Get videos from playlists if we still need more videos
    if args.playlists and len(collected_videos) < max_videos:
        for playlist_url in args.playlists:
            videos = get_playlist_videos_by_scraping(playlist_url, args.limit)
            collected_videos.extend(videos)
            logging.info(f"Collected {len(videos)} videos from playlist {playlist_url}")
            
            if len(collected_videos) >= max_videos:
                logging.info(f"Reached maximum video limit of {max_videos}")
                break
    
    # Search for videos if we still need more videos
    if len(collected_videos) < max_videos:
        search_queries = args.search if args.search else default_searches
        videos_needed = max_videos - len(collected_videos)
        videos_per_search = max(1, min(args.limit, videos_needed // len(search_queries)))
        
        for query in search_queries:
            videos = search_youtube(query, videos_per_search)
            collected_videos.extend(videos)
            logging.info(f"Collected {len(videos)} videos from search: {query}")
            
            if len(collected_videos) >= max_videos:
                logging.info(f"Reached maximum video limit of {max_videos}")
                break
    
    # Remove duplicates
    collected_videos = list(set(collected_videos))
    logging.info(f"Total unique videos collected: {len(collected_videos)}")
    
    # Save video metadata
    metadata_file = os.path.join(args.output, "fitness_videos.json")
    video_data = save_videos_metadata(collected_videos, metadata_file)
    
    # Extract transcripts
    transcript_file = os.path.join(args.output, "fitness_transcripts.json")
    data_with_transcripts = extract_transcripts(video_data, transcript_file)
    
    print(f"Collected {len(collected_videos)} unique videos")
    print(f"Extracted metadata for {len(video_data)} videos")
    print(f"Extracted {len(data_with_transcripts)} transcripts")
    print(f"Data saved to {args.output} directory")

if __name__ == "__main__":
    main() 