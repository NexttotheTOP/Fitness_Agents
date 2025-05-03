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
import sys

# Try to import Selenium dependencies
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Selenium not installed. For full channel scraping, install with: pip install selenium")
    print("You will also need ChromeDriver: https://chromedriver.chromium.org/")

import traceback

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
    """Get videos from a YouTube channel by scraping the channel page with Selenium to handle infinite scroll"""
    videos = []
    
    # If Selenium is not available, use the basic scraping method
    if not SELENIUM_AVAILABLE:
        logging.warning("Selenium not available. Using basic scraping method which may return fewer videos.")
        return get_channel_videos_basic_scraping(channel_url, limit)
        
    try:
        # Set up Chrome browser
        chrome_options = Options()
        if HEADLESS_MODE:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36")
        
        # Add preferences to handle consent automatically if possible
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # Try to avoid detection
        
        # Add additional experimental options
        chrome_prefs = {
            "profile.default_content_settings.popups": 0,
            "profile.managed_default_content_settings.images": 2,  # Don't load images in headless mode
            "profile.cookie_controls_mode": 0,  # Allow all cookies
        }
        
        # Only disable images in headless mode
        if HEADLESS_MODE:
            chrome_prefs["profile.managed_default_content_settings.images"] = 2
        else:
            chrome_prefs["profile.managed_default_content_settings.images"] = 0  # Load images in visible mode
            
        chrome_options.add_experimental_option("prefs", chrome_prefs)
        
        driver = webdriver.Chrome(options=chrome_options)
        
        # Handle YouTube's country/cookies consent screen
        def handle_consent_screen():
            try:
                logging.info("Checking for consent screen...")
                # Check if we're on a consent page
                if "consent" in driver.current_url:
                    logging.info("Detected consent screen, attempting to accept...")
                    # Look for the accept button using different possible selectors
                    consent_buttons = [
                        "//button[@aria-label='Accept all']",
                        "//button[contains(text(), 'Accept all')]",
                        "//button[contains(text(), 'I agree')]",
                        "//button[contains(@class, 'VfPpkd-LgbsSe')]",  # General material button class
                    ]
                    
                    for button_xpath in consent_buttons:
                        try:
                            button = WebDriverWait(driver, 3).until(
                                EC.element_to_be_clickable((By.XPATH, button_xpath))
                            )
                            button.click()
                            logging.info("Clicked consent button")
                            # Wait for redirect
                            time.sleep(2)
                            return True
                        except:
                            continue
                            
                    logging.warning("Could not find or click consent button")
                    return False
            except Exception as e:
                logging.error(f"Error handling consent: {e}")
                return False
            return True  # No consent needed or already handled
        
        # Convert /c/ URLs to /channel/ URLs if needed
        if '/c/' in channel_url or '/user/' in channel_url:
            logging.info(f"Resolving channel URL: {channel_url}")
            driver.get(channel_url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            # Handle consent if needed
            handle_consent_screen()
            
            # Get channel title
            try:
                channel_name = driver.title.replace(" - YouTube", "")
                logging.info(f"Found channel: {channel_name}")
            except:
                channel_name = "Unknown Channel"
            
            # Get current URL which should be resolved
            channel_url = driver.current_url
            logging.info(f"Resolved channel URL: {channel_url}")
        
        # Navigate to the videos tab
        videos_url = f"{channel_url}/videos"
        logging.info(f"Navigating to videos page: {videos_url}")
        driver.get(videos_url)
        
        # Handle consent screen if needed
        handle_consent_screen()
        
        # Wait for the page to load
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.ID, "contents"))
            )
        except TimeoutException:
            logging.error("Timed out waiting for video contents to load")
            # Try an alternative selector
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "ytd-grid-video-renderer, ytd-video-renderer"))
                )
                logging.info("Found video elements with alternative selector")
            except TimeoutException:
                logging.error("Could not find any video elements on the page")
                # Take a screenshot for debugging
                try:
                    driver.save_screenshot("youtube_error.png")
                    logging.info("Saved screenshot to youtube_error.png")
                except:
                    pass
                driver.quit()
                return videos
        
        # Get the channel name from the page title
        channel_name = driver.title.replace(" - Videos - YouTube", "").strip()
        logging.info(f"Scraping videos from channel: {channel_name}")
        
        # Initialize progress bar for scrolling
        pbar = tqdm(total=min(limit, 1000), desc=f"Scraping {channel_name} videos")  # Cap at 1000 to avoid infinite loop
        previous_count = 0
        same_count_iterations = 0
        max_same_count = 5  # Stop if no new videos found after 5 scrolls
        
        seen_videos = set()
        
        # Scroll down to load more videos
        while len(videos) < limit and same_count_iterations < max_same_count:
            # Extract video links
            try:
                # Try multiple selectors to find video links
                video_elements = driver.find_elements(By.CSS_SELECTOR, "a#video-title-link, a#video-title")
                
                if not video_elements:
                    # Try alternative selector
                    video_elements = driver.find_elements(By.CSS_SELECTOR, "ytd-grid-video-renderer a#thumbnail, ytd-video-renderer a#thumbnail")
                
                for element in video_elements:
                    href = element.get_attribute('href')
                    if href and '/watch?v=' in href:
                        video_id = extract_video_id(href)
                        if video_id and video_id not in seen_videos:
                            seen_videos.add(video_id)
                            videos.append(f"https://youtube.com/watch?v={video_id}")
                            pbar.update(1)
                
                # Check if we found new videos
                if len(videos) > previous_count:
                    previous_count = len(videos)
                    same_count_iterations = 0
                    pbar.set_description(f"Scraping {channel_name} videos ({len(videos)})")
                else:
                    same_count_iterations += 1
                
                if len(videos) >= limit:
                    break
                    
                # Scroll down to load more
                driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                time.sleep(random.uniform(1.5, 2.5))  # Wait for content to load
                
            except StaleElementReferenceException:
                # Handle stale elements by refreshing references
                logging.warning("Encountered stale elements, continuing...")
                time.sleep(1)
                continue
            except Exception as e:
                logging.error(f"Error during scrolling: {e}")
                logging.error(traceback.format_exc())
                break
        
        pbar.close()
        driver.quit()
        
        logging.info(f"Found {len(videos)} videos from {channel_name}")
        
    except Exception as e:
        logging.error(f"Error getting videos from channel {channel_url}: {e}")
        logging.error(traceback.format_exc())
        try:
            if 'driver' in locals() and driver:
                driver.quit()
        except:
            pass
    
    return videos

def get_channel_videos_basic_scraping(channel_url, limit=50):
    """Basic method to get videos from a YouTube channel without Selenium"""
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
    """Search YouTube for videos with error handling and fallback"""
    videos = []
    try:
        logging.info(f"Searching for: {query}")
        
        # First try with pytube
        try:
            s = Search(query)
            
            # Collect videos more carefully
            for i, result in enumerate(tqdm(s.results[:limit], desc=f"Searching for {query}")):
                try:
                    video_url = result.watch_url
                    if video_url and extract_video_id(video_url):
                        videos.append(video_url)
                    # Sleep to avoid rate limiting
                    time.sleep(random.uniform(0.5, 1.5))
                    
                    if len(videos) >= limit:
                        break
                except Exception as inner_e:
                    logging.warning(f"Error processing search result: {inner_e}")
                    continue
        except Exception as e:
            logging.warning(f"PyTube search failed, trying fallback method: {e}")
            
            # Fallback to Selenium search if pytube fails
            if SELENIUM_AVAILABLE:
                videos = search_youtube_selenium(query, limit)
                
    except Exception as e:
        logging.error(f"Error searching YouTube for {query}: {e}")
    
    return videos

def search_youtube_selenium(query, limit=20):
    """Search YouTube using Selenium as a fallback"""
    videos = []
    driver = None
    try:
        # Set up Chrome browser
        chrome_options = Options()
        if HEADLESS_MODE:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # Try to avoid detection
        
        # Add experimental options for accepting cookies/consent
        chrome_prefs = {
            "profile.default_content_settings.popups": 0,
            "profile.cookie_controls_mode": 0,  # Allow all cookies
        }
        
        # Only disable images in headless mode
        if HEADLESS_MODE:
            chrome_prefs["profile.managed_default_content_settings.images"] = 2  # Don't load images in headless mode
        else:
            chrome_prefs["profile.managed_default_content_settings.images"] = 0  # Load images in visible mode
            
        chrome_options.add_experimental_option("prefs", chrome_prefs)
        
        driver = webdriver.Chrome(options=chrome_options)
        
        # Format the search URL
        search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        logging.info(f"Navigating to search URL: {search_url}")
        driver.get(search_url)
        
        # Handle consent screen if present
        try:
            if "consent" in driver.current_url:
                logging.info("Detected consent screen in search, attempting to accept...")
                # Look for the accept button
                consent_buttons = [
                    "//button[@aria-label='Accept all']",
                    "//button[contains(text(), 'Accept all')]",
                    "//button[contains(text(), 'I agree')]",
                    "//button[contains(@class, 'VfPpkd-LgbsSe')]",
                ]
                
                for button_xpath in consent_buttons:
                    try:
                        button = WebDriverWait(driver, 3).until(
                            EC.element_to_be_clickable((By.XPATH, button_xpath))
                        )
                        button.click()
                        logging.info("Clicked consent button")
                        # Wait for redirect
                        time.sleep(2)
                        break
                    except:
                        continue
        except Exception as e:
            logging.error(f"Error handling consent in search: {e}")
        
        # Wait for the search results to load
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.ID, "contents"))
            )
        except TimeoutException:
            logging.error("Timed out waiting for search results to load")
            try:
                # Try alternative selector
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "ytd-video-renderer"))
                )
                logging.info("Found video elements with alternative selector")
            except:
                logging.error("Could not find any video elements on search page")
                if driver:
                    driver.quit()
                return videos
        
        # Initialize progress bar
        pbar = tqdm(total=limit, desc=f"Fetching search results for {query}")
        seen_videos = set()
        scroll_count = 0
        max_scroll = 10  # Limit scrolling in case of issues
        
        # Scroll and collect videos
        while len(videos) < limit and scroll_count < max_scroll:
            # Find video elements
            video_elements = driver.find_elements(By.CSS_SELECTOR, "a#video-title")
            if not video_elements:
                # Try alternative selector
                video_elements = driver.find_elements(By.CSS_SELECTOR, "ytd-video-renderer a#thumbnail")
            
            # Process found elements
            for element in video_elements:
                href = element.get_attribute('href')
                if href and '/watch?v=' in href and '&list=' not in href:  # Exclude playlist links
                    video_id = extract_video_id(href)
                    if video_id and video_id not in seen_videos:
                        seen_videos.add(video_id)
                        videos.append(f"https://youtube.com/watch?v={video_id}")
                        pbar.update(1)
                        
                        if len(videos) >= limit:
                            break
            
            if len(videos) >= limit:
                break
                
            # Scroll down to load more results
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            scroll_count += 1
            time.sleep(random.uniform(1.0, 2.0))
        
        pbar.close()
        logging.info(f"Found {len(videos)} videos from search: {query}")
        
    except Exception as e:
        logging.error(f"Error during Selenium search: {e}")
    finally:
        if driver:
            driver.quit()
    
    return videos

def save_videos_metadata(videos, output_file="RP_Mike_videos.json"):
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

def extract_transcripts(video_data, output_file="RP_Mike_transcripts.json"):
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
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of videos to collect per source (default: 20)")
    parser.add_argument("--output", type=str, default="fitness_data", help="Output directory for collected data")
    parser.add_argument("--max-videos", type=int, default=100, help="Maximum total number of videos to collect (default: 100)")
    parser.add_argument("--full-channel", action="store_true", help="Attempt to scrape all videos from a channel (requires Selenium)")
    parser.add_argument("--no-headless", action="store_true", help="Run browser in visible mode (not headless) for debugging")
    
    args = parser.parse_args()
    
    # Set global for headless mode
    global HEADLESS_MODE
    HEADLESS_MODE = not args.no_headless
    
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
    
    # Check if we're trying to do a full channel scrape
    if args.full_channel and args.channels:
        if not SELENIUM_AVAILABLE:
            print("WARNING: Full channel scraping requires Selenium. Please install it with:")
            print("pip install selenium")
            print("You will also need to download ChromeDriver: https://chromedriver.chromium.org/")
            print("Continuing with basic scraping which will get fewer videos...")
        else:
            print(f"Full channel scraping mode enabled. Will try to get all videos up to limit: {args.limit}")
            if not HEADLESS_MODE:
                print("Running in visible browser mode for debugging")
    
    # Get videos from channels
    if args.channels:
        for channel_url in args.channels:
            print(f"\nScraping channel: {channel_url}")
            # Set high limit for channel collection
            channel_limit = args.limit if args.full_channel else min(args.limit, max_videos - len(collected_videos))
            videos = get_channel_videos_by_scraping(channel_url, channel_limit)
            
            if videos:
                prev_count = len(collected_videos)
                collected_videos.extend(videos)
                logging.info(f"Collected {len(videos)} videos from channel {channel_url}")
                print(f"Added {len(collected_videos) - prev_count} videos from this channel")
            else:
                logging.warning(f"No videos collected from channel {channel_url}")
                print(f"No videos collected from this channel")
            
            if len(collected_videos) >= max_videos:
                logging.info(f"Reached maximum video limit of {max_videos}")
                break
    elif not args.playlists and not args.search:
        # Use default channels
        for channel_url in default_channels:
            videos = get_channel_videos_by_scraping(channel_url, min(args.limit, max_videos - len(collected_videos)))
            collected_videos.extend(videos)
            logging.info(f"Collected {len(videos)} videos from default channel {channel_url}")
            
            if len(collected_videos) >= max_videos:
                logging.info(f"Reached maximum video limit of {max_videos}")
                break
    
    # Get videos from playlists if we still need more videos
    # if args.playlists and len(collected_videos) < max_videos:
    #     for playlist_url in args.playlists:
    #         videos = get_playlist_videos_by_scraping(playlist_url, min(args.limit, max_videos - len(collected_videos)))
    #         
    #         if videos:
    #             prev_count = len(collected_videos)
    #             collected_videos.extend(videos)
    #             logging.info(f"Collected {len(videos)} videos from playlist {playlist_url}")
    #             print(f"Added {len(collected_videos) - prev_count} videos from this playlist")
    #         else:
    #             logging.warning(f"No videos collected from playlist {playlist_url}")
    #             print(f"No videos collected from this playlist")
    #         
    #         if len(collected_videos) >= max_videos:
    #             logging.info(f"Reached maximum video limit of {max_videos}")
    #             break
    
    # Search for videos if we still need more videos
    # if len(collected_videos) < max_videos:
    #     search_queries = args.search if args.search else default_searches
    #     videos_needed = max_videos - len(collected_videos)
    #     videos_per_search = max(1, min(args.limit, videos_needed // max(1, len(search_queries))))
    #     
    #     for query in search_queries:
    #         videos = search_youtube(query, videos_per_search)
    #         
    #         if videos:
    #             prev_count = len(collected_videos)
    #             collected_videos.extend(videos)
    #             logging.info(f"Collected {len(videos)} videos from search: {query}")
    #             print(f"Added {len(collected_videos) - prev_count} videos from search: {query}")
    #         else:
    #             logging.warning(f"No videos collected from search: {query}")
    #             print(f"No videos collected from search: {query}")
    #         
    #         if len(collected_videos) >= max_videos:
    #             logging.info(f"Reached maximum video limit of {max_videos}")
    #             break
    
    # Remove duplicates
    pre_dedup_count = len(collected_videos)
    collected_videos = list(set(collected_videos))
    post_dedup_count = len(collected_videos)
    logging.info(f"Removed {pre_dedup_count - post_dedup_count} duplicate videos")
    logging.info(f"Total unique videos collected: {post_dedup_count}")
    
    if not collected_videos:
        print("No videos were collected. Please check your inputs and try again.")
        return
    
    # Save video metadata
    metadata_file = os.path.join(args.output, "RP_Mike_videos.json")
    video_data = save_videos_metadata(collected_videos, metadata_file)
    
    # Extract transcripts
    transcript_file = os.path.join(args.output, "RP_Mike_transcripts.json")
    data_with_transcripts = extract_transcripts(video_data, transcript_file)
    
    print(f"\nCollection Summary:")
    print(f"------------------")
    print(f"Collected {len(collected_videos)} unique videos")
    print(f"Extracted metadata for {len(video_data)} videos")
    print(f"Extracted {len(data_with_transcripts)} transcripts")
    print(f"Data saved to {args.output} directory")
    
    # Print Jeff Nippard example if no channel was specified
    if not args.channels:
        print("\nExample to scrape Jeff Nippard's entire channel:")
        print("python youtube_fitness_collector.py --channels https://www.youtube.com/c/JeffNippard --limit 600 --max-videos 600 --full-channel --output fitness_data")

# Add global variable for headless mode
HEADLESS_MODE = True

if __name__ == "__main__":
    main() 