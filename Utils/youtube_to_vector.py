from dotenv import load_dotenv
import os
import json
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import logging
from tqdm import tqdm
# Import functions from ingestion.py
from ingestion import get_vectorstore, create_vectorstore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vector_db_creation.log"),
        logging.StreamHandler()
    ]
)

load_dotenv()

def load_transcripts(input_file):
    """Load transcripts from JSON file"""
    try:
        with open(input_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading transcripts from {input_file}: {e}")
        return []

def prepare_documents(transcripts_data):
    """Convert transcript data to Document objects"""
    documents = []
    
    for item in tqdm(transcripts_data, desc="Preparing documents"):
        if "transcript" not in item or not item["transcript"]:
            continue
            
        # Create a document with metadata
        doc = Document(
            page_content=item["transcript"],
            metadata={
                "source": item.get("url", ""),
                "title": item.get("title", "Unknown"),
                "author": item.get("author", "Unknown"),
                "content_type": "youtube_transcript",
                "video_id": item.get("video_id", ""),
            }
        )
        documents.append(doc)
    
    return documents

def create_vector_database(documents, collection_name="fitness-coach-chroma", persist_dir="./.chroma"):
    """Create or update a vector database from documents"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(persist_dir, exist_ok=True)
        
        # Chunk the documents for better retrieval
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500,
            chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(documents)
        
        logging.info(f"Created {len(doc_splits)} chunks from {len(documents)} documents")
        
        # Create or update the vector database
        embedding = OpenAIEmbeddings()
        
        # Check if collection exists
        try:
            existing_db = get_vectorstore()
            if existing_db:
                existing_count = existing_db._collection.count()
                logging.info(f"Found existing collection with {existing_count} entries")
            
            # Add new documents to existing collection
            vectorstore = Chroma.from_documents(
                documents=doc_splits,
                collection_name=collection_name,
                embedding=embedding,
                persist_directory=persist_dir
            )
            
            # Persist the database
            vectorstore.persist()
            final_count = vectorstore._collection.count()
            
            logging.info(f"Updated vector database. Now contains {final_count} entries")
            return vectorstore, final_count
            
        except Exception as e:
            logging.warning(f"Could not load existing collection: {e}")
            
            # Create new collection
            vectorstore = Chroma.from_documents(
                documents=doc_splits,
                collection_name=collection_name,
                embedding=embedding,
                persist_directory=persist_dir
            )
            
            # Persist the database
            vectorstore.persist()
            
            logging.info(f"Created new vector database with {len(doc_splits)} entries")
            return vectorstore, len(doc_splits)
            
    except Exception as e:
        logging.error(f"Error creating vector database: {e}")
        return None, 0

def main():
    parser = argparse.ArgumentParser(description="Convert YouTube transcripts to vector database")
    parser.add_argument("--input", type=str, default="fitness_data/fitness_transcripts.json",
                      help="Input JSON file containing transcripts")
    parser.add_argument("--collection", type=str, default="fitness-coach-chroma",
                      help="Name of the vector database collection")
    parser.add_argument("--persist_dir", type=str, default="./.chroma",
                      help="Directory to persist the vector database")
    
    args = parser.parse_args()
    
    # Load transcripts
    transcripts_data = load_transcripts(args.input)
    if not transcripts_data:
        logging.error(f"No transcripts found in {args.input}")
        return
        
    logging.info(f"Loaded {len(transcripts_data)} transcripts from {args.input}")
    
    # Prepare documents
    documents = prepare_documents(transcripts_data)
    logging.info(f"Prepared {len(documents)} documents")
    
    # Create or update vector database
    vectorstore, count = create_vector_database(
        documents,
        collection_name=args.collection,
        persist_dir=args.persist_dir
    )
    
    if vectorstore:
        print(f"Vector database created/updated successfully with {count} entries")
        print(f"Database stored in {args.persist_dir}")
    else:
        print("Failed to create/update vector database")

if __name__ == "__main__":
    main() 