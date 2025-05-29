#!/usr/bin/env python3
"""
Supabase Vector Retriever
Replaces ChromaDB with Supabase vector search using pgvector
"""

import numpy as np
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from graph.memory_store import get_supabase_client
import logging
from pydantic import Field

logger = logging.getLogger("supabase_retriever")

class SupabaseVectorRetriever(BaseRetriever):
    """
    A LangChain-compatible retriever that uses Supabase vector search
    instead of ChromaDB. Maintains the same interface as ChromaDB retriever.
    """
    
    # Define fields with proper Pydantic field definitions
    embedding_function: OpenAIEmbeddings = Field(default_factory=OpenAIEmbeddings)
    search_type: str = Field(default="similarity_score_threshold")
    search_kwargs: Dict[str, Any] = Field(default_factory=lambda: {"score_threshold": 0.5, "k": 15})
    supabase_client: Any = Field(default=None, exclude=True)  # Exclude from serialization
    
    def __init__(
        self, 
        embedding_function: OpenAIEmbeddings = None,
        search_type: str = "similarity_score_threshold",
        search_kwargs: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Initialize the Supabase vector retriever
        
        Args:
            embedding_function: OpenAI embeddings function (same as ChromaDB)
            search_type: Type of search ("similarity_score_threshold")
            search_kwargs: Search parameters (score_threshold, k, etc.)
        """
        # Set up fields before calling super().__init__
        if embedding_function is None:
            embedding_function = OpenAIEmbeddings()
        if search_kwargs is None:
            search_kwargs = {"score_threshold": 0.5, "k": 15}
            
        # Initialize with proper field values
        super().__init__(
            embedding_function=embedding_function,
            search_type=search_type,
            search_kwargs=search_kwargs,
            **kwargs
        )
        
        # Set up Supabase client (not a Pydantic field)
        object.__setattr__(self, 'supabase_client', get_supabase_client())
        
        logger.info(f"SupabaseVectorRetriever initialized with search_type='{search_type}', kwargs={search_kwargs}")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents from Supabase using vector similarity search.
        This replaces ChromaDB's similarity search.
        
        Args:
            query: The user's search query
            
        Returns:
            List of Document objects with similarity scores
        """
        try:
            # Step 1: Embed the query (same as ChromaDB process)
            logger.info(f"🔍 Embedding query: '{query[:50]}...'")
            query_embedding = self.embedding_function.embed_query(query)
            
            # Convert to list if needed (ensure compatibility)
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
            elif isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Step 2: Call Supabase vector search
            return self._search_supabase_vectors(query_embedding, query)
            
        except Exception as e:
            logger.error(f"❌ Error in vector retrieval: {e}")
            # Return empty list instead of failing completely
            return []
    
    def _search_supabase_vectors(self, query_embedding: List[float], original_query: str) -> List[Document]:
        """
        Perform the actual vector search in Supabase using the match_documents function
        
        Args:
            query_embedding: The embedded query vector
            original_query: Original query string for logging
            
        Returns:
            List of Document objects
        """
        try:
            # Extract search parameters
            score_threshold = self.search_kwargs.get("score_threshold", 0.5)
            match_count = self.search_kwargs.get("k", 15)
            
            logger.info(f"🔍 Searching Supabase vectors with threshold={score_threshold}, count={match_count}")
            
            # Call the Supabase vector search function
            # This calls the match_documents function we'll create in Supabase
            result = self.supabase_client.rpc(
                'match_documents',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': score_threshold,
                    'match_count': match_count
                }
            ).execute()
            
            if not result.data:
                logger.info("ℹ️ No matching documents found")
                return []
            
            # Convert Supabase results to LangChain Documents
            documents = []
            for item in result.data:
                # Create Document object (same format as ChromaDB)
                doc = Document(
                    page_content=item.get('content', ''),
                    metadata={
                        'source': item.get('source', ''),
                        'title': item.get('title', ''),
                        'author': item.get('author', ''),
                        'content_type': item.get('content_type', 'youtube_transcript'),
                        'video_id': item.get('video_id', ''),
                        'collection': item.get('collection', ''),
                        'chunk_index': item.get('chunk_index', 0),
                        'similarity_score': item.get('similarity', 0.0)  # Add similarity score
                    }
                )
                documents.append(doc)
            
            logger.info(f"✅ Found {len(documents)} matching documents")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Supabase vector search error: {e}")
            
            # Fallback: try a simpler approach if the RPC function doesn't exist
            return self._fallback_search(query_embedding, original_query)
    
    def _fallback_search(self, query_embedding: List[float], original_query: str) -> List[Document]:
        """
        Fallback search method if the custom RPC function isn't available
        Uses direct SQL with vector operations
        """
        try:
            logger.warning("🔄 Using fallback search method")
            
            # Simple approach: get all documents and calculate similarity in Python
            # This is less efficient but works without custom SQL functions
            
            # Get a reasonable sample of documents (can't get all 21k at once)
            result = self.supabase_client.table('fitness_documents')\
                .select('id, content, embedding, source, title, author, content_type, video_id, collection, chunk_index')\
                .limit(1000)\
                .execute()
            
            if not result.data:
                return []
            
            # Calculate similarities in Python
            documents_with_scores = []
            query_vector = np.array(query_embedding)
            
            for item in result.data:
                if not item.get('embedding'):
                    continue
                    
                try:
                    # Handle different embedding formats
                    embedding_data = item['embedding']
                    
                    if isinstance(embedding_data, str):
                        # Parse JSON string to list
                        import json
                        doc_vector = np.array(json.loads(embedding_data))
                    elif isinstance(embedding_data, list):
                        # Already a list
                        doc_vector = np.array(embedding_data)
                    else:
                        # Unknown format, skip
                        logger.warning(f"⚠️ Unknown embedding format: {type(embedding_data)}")
                        continue
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
                    
                    # Apply threshold
                    if similarity >= self.search_kwargs.get("score_threshold", 0.5):
                        doc = Document(
                            page_content=item.get('content', ''),
                            metadata={
                                'source': item.get('source', ''),
                                'title': item.get('title', ''),
                                'author': item.get('author', ''),
                                'content_type': item.get('content_type', 'youtube_transcript'),
                                'video_id': item.get('video_id', ''),
                                'collection': item.get('collection', ''),
                                'chunk_index': item.get('chunk_index', 0),
                                'similarity_score': float(similarity)
                            }
                        )
                        documents_with_scores.append((doc, similarity))
                        
                except Exception as embedding_error:
                    logger.warning(f"⚠️ Error processing embedding for document {item.get('id', 'unknown')}: {embedding_error}")
                    continue
            
            # Sort by similarity and return top k
            documents_with_scores.sort(key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, score in documents_with_scores[:self.search_kwargs.get("k", 15)]]
            
            logger.info(f"✅ Fallback search found {len(top_docs)} documents")
            return top_docs
            
        except Exception as e:
            logger.error(f"❌ Fallback search failed: {e}")
            return []

def get_supabase_retriever(
    search_type: str = "similarity_score_threshold",
    search_kwargs: Dict[str, Any] = None
) -> SupabaseVectorRetriever:
    """
    Create a Supabase vector retriever (replaces ChromaDB retriever)
    
    Args:
        search_type: Type of search
        search_kwargs: Search parameters
        
    Returns:
        SupabaseVectorRetriever instance
    """
    if search_kwargs is None:
        search_kwargs = {"score_threshold": 0.5, "k": 15}
    
    return SupabaseVectorRetriever(
        embedding_function=OpenAIEmbeddings(),
        search_type=search_type,
        search_kwargs=search_kwargs
    )

def check_supabase_vectorstore() -> tuple[bool, Optional[SupabaseVectorRetriever]]:
    """
    Check if Supabase vectorstore is available (replaces ChromaDB check)
    
    Returns:
        (exists, retriever) tuple
    """
    try:
        supabase = get_supabase_client()
        
        # Check if the fitness_documents table exists and has data
        result = supabase.table('fitness_documents').select('id', count='exact').limit(1).execute()
        
        if hasattr(result, 'count') and result.count > 0:
            logger.info(f"✅ Supabase vectorstore available with {result.count} documents")
            # Create retriever
            retriever = get_supabase_retriever()
            return True, retriever
        else:
            logger.warning("⚠️ Supabase vectorstore table exists but is empty")
            return False, None
            
    except Exception as e:
        logger.error(f"❌ Error checking Supabase vectorstore: {e}")
        return False, None 