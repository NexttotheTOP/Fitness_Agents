# In this file we are simply going to load articles into the langchain document format, 
# chunk them to smaller pieces, embed them and save them to a vector database.

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import os
import time
from langchain_core.documents import Document
import json
import logging
import shutil

# Add Supabase vector support
try:
    from supabase_retriever import (
        get_supabase_retriever, 
        check_supabase_vectorstore,
        SupabaseVectorRetriever
    )
    SUPABASE_AVAILABLE = True
    logger = logging.getLogger("ingestion")
    logger.info("✅ Supabase vector retriever available")
except ImportError as e:
    SUPABASE_AVAILABLE = False
    logger = logging.getLogger("ingestion")
    logger.warning(f"⚠️ Supabase vector retriever not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vector_db_debug.log"),
        logging.StreamHandler()
    ]
)

# Disable excessive logging from HTTP libraries
for noisy_logger in ["httpx", "httpcore", "hpack", "httpcore.http2"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

load_dotenv()

# Environment variable to control which vector store to use
USE_SUPABASE = os.getenv("USE_SUPABASE_VECTOR", "true").lower() == "true"

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

def get_retriever():
    """
    Get a retriever from the available vectorstore.
    Only Supabase is supported.
    """
    logger.info("🔍 Creating retriever from Supabase vectorstore...")
    if USE_SUPABASE and SUPABASE_AVAILABLE:
        logger.info("🚀 Attempting to use Supabase vector retriever...")
        try:
            exists, supabase_retriever = check_supabase_vectorstore()
            if exists and supabase_retriever is not None:
                logger.info("✅ Supabase retriever created successfully")
                print("Supabase Retriever created ================================")
                return supabase_retriever
            else:
                logger.error("❌ Supabase vectorstore not available.")
        except Exception as e:
            logger.error(f"❌ Supabase retriever failed: {e}")
    logger.error("❌ Failed to create any retriever - no Supabase vectorstore available")
    return None

def check_vectorstore_availability():
    """
    Check if Supabase vector store is available and return status
    """
    status = {
        "supabase": {"available": False, "enabled": USE_SUPABASE, "count": 0}
    }
    if SUPABASE_AVAILABLE and USE_SUPABASE:
        try:
            exists, retriever = check_supabase_vectorstore()
            if exists:
                try:
                    from graph.memory_store import get_supabase_client
                    supabase = get_supabase_client()
                    result = supabase.table('fitness_documents').select('id', count='exact').limit(1).execute()
                    status["supabase"]["count"] = getattr(result, 'count', 0)
                    status["supabase"]["available"] = True
                except:
                    status["supabase"]["available"] = True
        except Exception as e:
            logger.warning(f"Supabase check failed: {e}")
    return status

def get_retriever_info():
    """Get information about the current retriever setup (Supabase only)"""
    info = {
        "current_retriever": None,
        "supabase_enabled": USE_SUPABASE,
        "supabase_available": SUPABASE_AVAILABLE,
        "status": check_vectorstore_availability()
    }
    if USE_SUPABASE and SUPABASE_AVAILABLE and info["status"]["supabase"]["available"]:
        info["current_retriever"] = "supabase"
    else:
        info["current_retriever"] = "none"
    return info

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
    persist_directory = "./.fitness_chroma"
    try:
        # logger.info(f"Attempting to delete existing vector database at {persist_directory}")
        if os.path.exists(persist_directory):
            # Check write permissions before attempting to delete
            if os.access(persist_directory, os.W_OK):
                # logger.info(f"Directory has write permissions, proceeding with delete")
                pass
            else:
                # logger.error(f"No write permission for {persist_directory}")
                return False
                
            # Check for lock files
            lock_files = []
            for root, dirs, files in os.walk(persist_directory):
                for file in files:
                    if file.endswith('.lock'):
                        lock_path = os.path.join(root, file)
                        lock_files.append(lock_path)
                        # logger.warning(f"Found lock file: {lock_path}")
            
            # Delete directory
            # logger.info(f"Deleting directory {persist_directory}")
            shutil.rmtree(persist_directory)
            # logger.info("Vector database deleted successfully")
            
            # Verify deletion
            if not os.path.exists(persist_directory):
                # logger.info("Verified directory deletion was successful")
                pass
            else:
                # logger.error("Directory still exists after deletion attempt")
                return False
                
            # Create empty directory to ensure fresh start
            # logger.info(f"Creating fresh directory at {persist_directory}")
            os.makedirs(persist_directory, exist_ok=True)
            
            # Test write access to new directory
            test_file = os.path.join(persist_directory, "test_write.txt")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                # logger.info(f"Successfully wrote test file to {test_file}")
                os.remove(test_file)
                # logger.info("Test file removed")
            except Exception as e:
                # logger.error(f"Failed write test to new directory: {e}", exc_info=True)
                return False
                
        return True
    except Exception as e:
        # logger.error(f"Error deleting vector database: {e}", exc_info=True)
        return False

def create_fitness_vector_database(batch_size=250):
    """
    Create or update vector database with fitness YouTubers' content.
    This function processes all collected fitness data JSON files.
    
    Args:
        batch_size: Number of documents to process in each batch to avoid token limits
    """
    # logger.info("Starting fitness vector database creation")
    
    # Clear existing vector database
    if not delete_existing_vectorstore():
        # logger.error("Failed to prepare directory for new vector database")
        return None
    
    # Load all fitness YouTuber documents
    # logger.info("Loading fitness YouTuber documents")
    documents = load_fitness_youtubers_data()
    
    if not documents:
        # logger.error("No documents found to process!")
        return None
    
    # Chunk documents
    # logger.info(f"Chunking {len(documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(documents)
    
    #logger.info(f"Created {len(doc_splits)} chunks from {len(documents)} documents")
    
    # Save to vector database
    persist_directory = "./.fitness_chroma"
    # logger.info(f"Preparing directory {persist_directory}")
    
    # Ensure directory exists and is writable
    os.makedirs(persist_directory, exist_ok=True)
    
    # Create consistent client settings to use throughout
    import chromadb
    client_settings = chromadb.config.Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
    # logger.info("Created ChromaDB client settings")
    
    # Process first batch separately to create the collection
    try:
        # logger.info("Creating new collection with first batch")
        batch_end = min(batch_size, len(doc_splits))
        first_batch = doc_splits[0:batch_end]
        
        vectorstore = Chroma.from_documents(
            documents=first_batch,
            collection_name="fitness-coach-chroma-new",
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory,
            client_settings=client_settings
        )
        vectorstore.persist()
        logger.info(f"Successfully created collection with {batch_end} documents")
    except Exception as e:
        # logger.error(f"Error creating collection: {e}", exc_info=True)
        return None
    
    # Process remaining batches
    if len(doc_splits) > batch_size:
        try:
            # Use the same vectorstore instance for all subsequent batches
            for i in range(batch_size, len(doc_splits), batch_size):
                end_idx = min(i + batch_size, len(doc_splits))
                current_batch = doc_splits[i:end_idx]
                # logger.info(f"Processing batch {i//batch_size + 1}/{(len(doc_splits) + batch_size - 1)//batch_size}: chunks {i} to {end_idx-1}")
                
                # Add documents to the existing collection
                vectorstore.add_documents(current_batch)
                vectorstore.persist()
                # logger.info(f"Successfully added batch {i//batch_size + 1}")
                
        except Exception as e:
            # logger.error(f"Error adding documents: {e}", exc_info=True)
            # Continue with what we have
            pass
    
    # Verify the final count
    try:
        doc_count = vectorstore._collection.count()
        logger.info(f"Vector database creation complete with {doc_count} documents")
    except Exception as e:
        # logger.error(f"Could not get final document count: {e}", exc_info=True)
        pass
    
    # Update the global retriever
    # logger.info("Updating global retriever")
    try:
        global retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"score_threshold": 0.5, "k": 15}
        )
        # logger.info("Global retriever updated with new vector database")
    except Exception as e:
        # logger.error(f"Failed to update global retriever: {e}", exc_info=True)
        pass
    
    return vectorstore


# This code only runs when directly executing this script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fitness Vector Database Management")
    parser.add_argument("--create", action="store_true", help="Create or recreate the vector database")
    parser.add_argument("--check", action="store_true", help="Check vector database status")
    
    args = parser.parse_args()
    
    if args.create:
        # logger.info("Creating fitness vector database")
        create_fitness_vector_database()
    elif args.check:
        # logger.info("Checking vector database status")
        check_vectorstore()
    else:
        # logger.info("Running with default action: create fitness vector database")
        create_fitness_vector_database() 






