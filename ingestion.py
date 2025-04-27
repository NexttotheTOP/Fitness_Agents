# In this file we are simply going to load articles into the langchain document format, 
# chunk them to smaller pieces, embed them and save them to a vector database.

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import os
import time
from langchain_core.documents import Document

load_dotenv()

# Function to extract video ID from YouTube URL
def extract_video_id(url):
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    elif "youtube.com" in url:
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
    return None

# Function to get video details using pytube
def get_video_details(url):
    try:
        yt = YouTube(url)
        return {
            "title": yt.title,
            "author": yt.author,
            "description": yt.description,
            "length": yt.length,
            "publish_date": yt.publish_date,
        }
    except Exception as e:
        print(f"Error fetching video details for {url}: {e}")
        return None

# Function to get transcript using youtube_transcript_api
def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([item["text"] for item in transcript_list])
        return transcript_text
    except Exception as e:
        print(f"Error fetching transcript for video {video_id}: {e}")
        return None

# YouTube channels/creators to use
creators = [
    "AthleanX",
    "Jeff Nippard", 
    "Renaissance Periodization"
]

# Example URLs for each creator - replace with actual playlists or specific videos
youtube_urls = [
    # AthleanX
    "https://www.youtube.com/watch?v=rSskXJr4Wrg",  # Sample: Full Body Workout
    "https://www.youtube.com/watch?v=JOAQgBtl_QI",  # Sample: Chest workout
    # Jeff Nippard
    "https://www.youtube.com/watch?v=iSGztQPRsto",  # Sample: Push Pull Legs science
    "https://www.youtube.com/watch?v=V88zHocMYYE",  # Sample: Full body 5 day split
    # Renaissance Periodization
    "https://www.youtube.com/watch?v=D0qDkq2aVNE",  # Sample: Top 5 exercise mistakes
    "https://www.youtube.com/watch?v=NLuQmiOVm7Q",  # Sample: How many exercises per muscle
]

def collect_and_prepare_documents():
    """Collect and prepare documents from YouTube videos - ADMIN ONLY FUNCTION"""
    # Prepare documents
    documents = []

    for url in youtube_urls:
        print(f"Processing: {url}")
        video_id = extract_video_id(url)
        if not video_id:
            print(f"Couldn't extract video ID from {url}")
            continue
        
        # Get video details
        details = get_video_details(url)
        if not details:
            continue
        
        # Get transcript
        transcript = get_transcript(video_id)
        if not transcript:
            continue
        
        # Create document with metadata
        doc = Document(
            page_content=transcript,
            metadata={
                "source": url,
                "title": details["title"],
                "author": details["author"],
                "content_type": "youtube_transcript",
                "video_length": details["length"],
                "publish_date": str(details["publish_date"]),
            }
        )
        documents.append(doc)
        
        # Sleep to avoid rate limiting
        time.sleep(1)

    print(f"Processed {len(documents)} videos")

    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=50  # Increased chunk size and added overlap for context
    )
    doc_splits = text_splitter.split_documents(documents)

    print(f"Created {len(doc_splits)} chunks")
    
    return doc_splits

def create_vectorstore():
    """Admin function to create the vectorstore - only called directly by admin tools"""
    doc_splits = collect_and_prepare_documents()
    
    # Save to vector database
    persist_directory = "./.chroma"
    os.makedirs(persist_directory, exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="fitness-coach-chroma",
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_directory,
    )
    
    print(f"Vector database created and saved to {persist_directory}")
    return vectorstore

def get_vectorstore():
    """Get the existing vectorstore - used by the main application"""
    persist_directory = "./.chroma"
    try:
        return Chroma(
            collection_name="fitness-coach-chroma",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=persist_directory
        )
    except Exception as e:
        print(f"Error accessing vectorstore: {e}")
        return None

def get_retriever():
    """Get a retriever from the existing vectorstore"""
    vs = get_vectorstore()
    if vs is not None:
        return vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10}
        )
    return None

# Export the retriever for use in the application
retriever = get_retriever()

# This code only runs when directly executing this script
if __name__ == "__main__":
    create_vectorstore()






