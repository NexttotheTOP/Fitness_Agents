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
import json

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

def load_fitness_youtubers_data():
    """
    Load and process fitness data from the three YouTubers' JSON files.
    Returns documents ready for embedding.
    """
    documents = []
    
    # Define the YouTubers and their corresponding files
    youtubers = [
        {"name": "Jeff Nippard", 
         "videos_file": "fitness_data/Jeff_Nippard_videos.json", 
         "transcripts_file": "fitness_data/Jeff_Nippard_transcripts.json"},
        {"name": "AthleanX", 
         "videos_file": "fitness_data/AthleanX_videos.json", 
         "transcripts_file": "fitness_data/AthleanX_transcripts.json"},
        {"name": "Renaissance Periodization", 
         "videos_file": "fitness_data/RP_Mike_videos.json", 
         "transcripts_file": "fitness_data/RP_Mike_transcripts.json"}
    ]
    
    for youtuber in youtubers:
        print(f"Processing data for {youtuber['name']}...")
        
        # Load transcript data
        try:
            with open(youtuber["transcripts_file"], "r") as f:
                transcripts_data = json.load(f)
            print(f"  Loaded {len(transcripts_data)} transcript entries")
        except Exception as e:
            print(f"  Error loading {youtuber['transcripts_file']}: {e}")
            transcripts_data = []
        
        # Process transcript data
        for item in transcripts_data:
            try:
                # Create document with metadata
                doc = Document(
                    page_content=item.get("transcript", ""),
                    metadata={
                        "source": item.get("url", ""),
                        "title": item.get("title", "Unknown"),
                        "author": item.get("author", youtuber["name"]),
                        "content_type": "youtube_transcript",
                        "video_id": item.get("video_id", ""),
                        "collection": youtuber["name"]
                    }
                )
                documents.append(doc)
            except Exception as e:
                print(f"  Error processing transcript item: {e}")
    
    print(f"Total documents processed: {len(documents)}")
    return documents

def delete_existing_vectorstore():
    """Delete the existing vector database to start fresh"""
    persist_directory = "./.chroma"
    try:
        import shutil
        if os.path.exists(persist_directory):
            print(f"Deleting existing vector database at {persist_directory}...")
            shutil.rmtree(persist_directory)
            print("Vector database deleted successfully")
        return True
    except Exception as e:
        print(f"Error deleting vector database: {e}")
        return False

def create_fitness_vector_database(batch_size=250):
    """
    Create or update vector database with fitness YouTubers' content.
    This function processes all collected fitness data JSON files.
    
    Args:
        batch_size: Number of documents to process in each batch to avoid token limits
    """
    # Clear existing vector database
    delete_existing_vectorstore()
    
    # Load all fitness YouTuber documents
    documents = load_fitness_youtubers_data()
    
    if not documents:
        print("No documents found to process!")
        return None
    
    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(documents)
    
    print(f"Created {len(doc_splits)} chunks from {len(documents)} documents")
    
    # Save to vector database
    persist_directory = "./.chroma"
    os.makedirs(persist_directory, exist_ok=True)
    embedding = OpenAIEmbeddings()
    
    # Process in batches to avoid token limits
    total_chunks = len(doc_splits)
    for i in range(0, total_chunks, batch_size):
        end_idx = min(i + batch_size, total_chunks)
        current_batch = doc_splits[i:end_idx]
        print(f"Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}: chunks {i} to {end_idx-1}")
        
        try:
            # On first batch, create the collection
            if i == 0:
                vectorstore = Chroma.from_documents(
                    documents=current_batch,
                    collection_name="fitness-coach-chroma",
                    embedding=embedding,
                    persist_directory=persist_directory,
                )
                vectorstore.persist()
            else:
                # Add subsequent batches to existing collection
                vectorstore = Chroma(
                    collection_name="fitness-coach-chroma",
                    embedding_function=embedding,
                    persist_directory=persist_directory
                )
                vectorstore.add_documents(current_batch)
                vectorstore.persist()
            
            print(f"Successfully processed batch {i//batch_size + 1}")
        
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Still try to continue with next batch
    
    print(f"Vector database created with fitness YouTuber content")
    
    # Update the global retriever to use the new database
    global retriever
    new_vectorstore = get_vectorstore()
    if new_vectorstore:
        retriever = new_vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10}
        )
        print("Global retriever updated with new vector database")
    
    return new_vectorstore

# This code only runs when directly executing this script
if __name__ == "__main__":
    # Comment/uncomment the appropriate function to run
    # create_vectorstore()  # Original function for sample videos
    create_fitness_vector_database()  # New function for fitness YouTubers data






