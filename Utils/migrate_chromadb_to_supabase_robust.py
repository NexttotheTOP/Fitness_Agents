#!/usr/bin/env python3
"""
Robust ChromaDB to Supabase Migration Script
Handles SSL errors, connection issues, and prevents duplicates
"""

import os
import json
import time
from typing import List, Dict, Any, Set
from datetime import datetime
import logging
from dotenv import load_dotenv
import numpy as np
import hashlib

# Import existing functions
from ingestion import check_vectorstore, get_vectorstore
from graph.memory_store import get_supabase_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("migration")

load_dotenv()

def get_content_hash(content: str) -> str:
    """Generate a hash for content to detect duplicates"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def get_existing_hashes(supabase) -> Set[str]:
    """Get all existing content hashes from Supabase to detect duplicates"""
    try:
        logger.info("🔍 Checking for existing documents...")
        # Get all existing documents in small batches to avoid timeout
        existing_hashes = set()
        page_size = 1000
        page = 0
        
        while True:
            result = supabase.table('fitness_documents')\
                .select('content')\
                .range(page * page_size, (page + 1) * page_size - 1)\
                .execute()
            
            if not result.data:
                break
                
            for doc in result.data:
                content_hash = get_content_hash(doc['content'])
                existing_hashes.add(content_hash)
            
            page += 1
            if len(result.data) < page_size:
                break
                
        logger.info(f"📊 Found {len(existing_hashes)} existing document hashes")
        return existing_hashes
        
    except Exception as e:
        logger.warning(f"⚠️ Could not get existing hashes: {e}")
        return set()

def investigate_skipped_documents():
    """Investigate why documents are being skipped"""
    logger.info("🔍 INVESTIGATION MODE: Analyzing skipped documents...")
    
    # Get ChromaDB data
    exists, vs = check_vectorstore()
    if not exists or vs is None:
        logger.error("❌ No ChromaDB vectorstore found!")
        return
    
    collection = vs._collection
    chromadb_data = collection.get(include=['documents', 'metadatas'])
    
    # Get Supabase data
    supabase = get_supabase_client()
    existing_hashes = get_existing_hashes(supabase)
    
    # Check for duplicates within ChromaDB itself
    chromadb_hashes = {}
    chromadb_duplicates = []
    
    for i, doc in enumerate(chromadb_data['documents']):
        content_hash = get_content_hash(doc)
        if content_hash in chromadb_hashes:
            chromadb_duplicates.append({
                'hash': content_hash,
                'original_index': chromadb_hashes[content_hash],
                'duplicate_index': i,
                'content_preview': doc[:100] + '...',
                'metadata': chromadb_data['metadatas'][i]
            })
        else:
            chromadb_hashes[content_hash] = i
    
    # Analyze skipped documents
    skipped_analysis = []
    for i, doc in enumerate(chromadb_data['documents']):
        content_hash = get_content_hash(doc)
        if content_hash in existing_hashes:
            skipped_analysis.append({
                'index': i,
                'hash': content_hash,
                'content_preview': doc[:100] + '...',
                'metadata': chromadb_data['metadatas'][i],
                'reason': 'Already exists in Supabase'
            })
    
    # Report findings
    logger.info(f"📊 INVESTIGATION RESULTS:")
    logger.info(f"   ChromaDB total documents: {len(chromadb_data['documents'])}")
    logger.info(f"   ChromaDB unique hashes: {len(chromadb_hashes)}")
    logger.info(f"   ChromaDB internal duplicates: {len(chromadb_duplicates)}")
    logger.info(f"   Supabase existing hashes: {len(existing_hashes)}")
    logger.info(f"   Would be skipped: {len(skipped_analysis)}")
    
    if chromadb_duplicates:
        logger.info(f"\n🔍 ChromaDB Internal Duplicates (first 5):")
        for dup in chromadb_duplicates[:5]:
            logger.info(f"   Hash: {dup['hash'][:12]}... Original idx: {dup['original_index']}, Duplicate idx: {dup['duplicate_index']}")
            logger.info(f"   Content: {dup['content_preview']}")
            logger.info(f"   Metadata: {dup['metadata']}")
    
    if skipped_analysis:
        logger.info(f"\n🔍 Would Be Skipped (first 5):")
        for skip in skipped_analysis[:5]:
            logger.info(f"   Hash: {skip['hash'][:12]}... Index: {skip['index']}")
            logger.info(f"   Content: {skip['content_preview']}")
            logger.info(f"   Metadata: {skip['metadata']}")

def clear_supabase_safely(supabase):
    """Safely clear Supabase data in small batches"""
    try:
        logger.info("🗑️ Clearing existing Supabase data...")
        
        # Get total count first
        count_result = supabase.table('fitness_documents').select('id', count='exact').execute()
        total_count = count_result.count if hasattr(count_result, 'count') else 0
        
        if total_count == 0:
            logger.info("✅ No existing data to clear")
            return True
            
        logger.info(f"📊 Found {total_count} documents to clear")
        
        # Delete in small batches
        batch_size = 100
        deleted_total = 0
        
        while True:
            # Get a batch of IDs
            result = supabase.table('fitness_documents')\
                .select('id')\
                .limit(batch_size)\
                .execute()
            
            if not result.data:
                break
                
            # Delete this batch
            ids_to_delete = [doc['id'] for doc in result.data]
            delete_result = supabase.table('fitness_documents')\
                .delete()\
                .in_('id', ids_to_delete)\
                .execute()
            
            deleted_count = len(ids_to_delete)
            deleted_total += deleted_count
            logger.info(f"🗑️ Deleted {deleted_count} documents (Total: {deleted_total}/{total_count})")
            
            time.sleep(0.5)  # Small delay to prevent timeout
            
        logger.info(f"✅ Cleared {deleted_total} documents from Supabase")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to clear Supabase data: {e}")
        return False

def migrate_data_with_resume(mode="normal"):
    """Main migration function with resume capability and duplicate detection
    
    Args:
        mode: "normal" for regular migration, "force_all" to migrate everything including duplicates
    """
    logger.info("🚀 Starting ChromaDB to Supabase migration...")
    
    # 1. Get ChromaDB data
    logger.info("📥 Loading ChromaDB data...")
    exists, vs = check_vectorstore()
    if not exists or vs is None:
        logger.error("❌ No ChromaDB vectorstore found!")
        return False
    
    collection = vs._collection
    total_count = collection.count()
    logger.info(f"📊 Found {total_count} documents in ChromaDB")
    
    # Get all data from ChromaDB
    logger.info("🔄 Extracting embeddings and metadata...")
    chromadb_data = collection.get(
        include=['documents', 'metadatas', 'embeddings']
    )
    
    # 2. Setup Supabase client
    logger.info("🔗 Connecting to Supabase...")
    supabase = get_supabase_client()
    
    # 3. Check for existing data and ask user what to do
    try:
        existing_result = supabase.table('fitness_documents').select('id', count='exact').execute()
        existing_count = existing_result.count if hasattr(existing_result, 'count') else 0
        logger.info(f"📊 Found {existing_count} existing documents in Supabase")
        
        if existing_count > 0 and mode == "normal":
            logger.info("⚠️ Found existing data in Supabase. Options:")
            logger.info("1. Clear all existing data and start fresh")
            logger.info("2. Skip duplicates and add only new documents")
            
            # For automation, let's skip duplicates by default
            logger.info("🔄 Proceeding with duplicate detection...")
            existing_hashes = get_existing_hashes(supabase)
        else:
            existing_hashes = set()
            
    except Exception as e:
        logger.warning(f"⚠️ Could not check existing documents: {e}")
        existing_hashes = set()
    
    # 4. Transform and insert data with duplicate detection
    batch_size = 25  # Smaller batches for free tier
    documents = chromadb_data['documents']
    metadatas = chromadb_data['metadatas'] 
    embeddings = chromadb_data['embeddings']
    
    total_inserted = 0
    total_skipped = 0
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_embeds = embeddings[i:i+batch_size]
        
        batch_num = i//batch_size + 1
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                # Prepare batch for Supabase with duplicate checking
                supabase_batch = []
                batch_skipped = 0
                
                for j, (doc, meta, embed) in enumerate(zip(batch_docs, batch_metas, batch_embeds)):
                    # Check for duplicates (skip in normal mode, force in force_all mode)
                    content_hash = get_content_hash(doc)
                    if mode == "normal" and content_hash in existing_hashes:
                        batch_skipped += 1
                        continue
                    elif mode == "force_all" and content_hash in existing_hashes:
                        # Add a suffix to make it unique for force mode
                        doc = doc + f" [DUPLICATE_MIGRATION_{datetime.now().strftime('%Y%m%d_%H%M%S')}]"
                        logger.info(f"🔄 Force migrating duplicate with suffix")
                    
                    # FIX: Proper embedding conversion - ensure it's a list of floats
                    if hasattr(embed, 'tolist'):
                        embedding_list = embed.tolist()
                    elif isinstance(embed, list):
                        embedding_list = embed
                    else:
                        # Convert numpy array or other formats to list
                        embedding_list = np.array(embed).flatten().tolist()
                    
                    # Validate embedding dimensions (should be 1536 for OpenAI)
                    if len(embedding_list) != 1536:
                        logger.warning(f"⚠️ Unexpected embedding dimension: {len(embedding_list)} (expected 1536)")
                        continue
                    
                    # Validate all values are finite
                    if not all(np.isfinite(val) for val in embedding_list):
                        logger.warning(f"⚠️ Invalid embedding values detected, skipping document")
                        continue
                    
                    supabase_doc = {
                        'content': doc,
                        'embedding': embedding_list,
                        'metadata': json.dumps(meta) if meta else '{}',
                        'source': meta.get('source', '') if meta else '',
                        'title': meta.get('title', '') if meta else '',
                        'author': meta.get('author', '') if meta else '',
                        'content_type': meta.get('content_type', 'youtube_transcript') if meta else 'youtube_transcript',
                        'video_id': meta.get('video_id', '') if meta else '',
                        'collection': meta.get('collection', '') if meta else '',
                        'chunk_index': meta.get('chunk_index', 0) if meta else 0,
                        'created_at': datetime.now().isoformat()
                    }
                    supabase_batch.append(supabase_doc)
                    # Add to existing hashes to prevent duplicates within this migration
                    if mode == "normal":
                        existing_hashes.add(content_hash)
                
                # Skip if no new documents in this batch
                if not supabase_batch:
                    total_skipped += batch_skipped
                    logger.info(f"⏭️ Batch {batch_num}: Skipped {batch_skipped} duplicates (Total skipped: {total_skipped})")
                    break
                
                # Insert batch to Supabase
                result = supabase.table('fitness_documents').insert(supabase_batch).execute()
                inserted_count = len(result.data) if result.data else len(supabase_batch)
                total_inserted += inserted_count
                total_skipped += batch_skipped
                
                logger.info(f"✅ Batch {batch_num}: Inserted {inserted_count}, Skipped {batch_skipped} duplicates (Total: {total_inserted} inserted, {total_skipped} skipped)")
                
                # Delay for free tier
                time.sleep(1)
                break  # Success, break retry loop
                
            except Exception as e:
                logger.warning(f"⚠️ Batch {batch_num} attempt {retry + 1} failed: {e}")
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 3  # Longer backoff for free tier
                    logger.info(f"🔄 Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    # Reconnect to Supabase
                    supabase = get_supabase_client()
                else:
                    logger.error(f"❌ Batch {batch_num} failed after {max_retries} attempts")
                    logger.info(f"📝 Resume from batch {batch_num} (index {i}) if needed")
                    return False
    
    # 5. Verify migration
    logger.info("🔍 Verifying migration...")
    try:
        result = supabase.table('fitness_documents').select('id', count='exact').execute()
        migrated_count = result.count if hasattr(result, 'count') else 0
        logger.info(f"📊 Migration complete! {migrated_count} total documents in Supabase")
        logger.info(f"📊 This session: {total_inserted} inserted, {total_skipped} skipped (duplicates)")
        
        if total_inserted > 0:
            logger.info("✅ Migration successful!")
            return True
        else:
            logger.info("ℹ️ No new documents were added (all were duplicates)")
            return True
            
    except Exception as e:
        logger.error(f"❌ Verification error: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "investigate":
            investigate_skipped_documents()
        elif command == "force_all":
            logger.info("🔄 FORCE MODE: Migrating all documents including duplicates...")
            success = migrate_data_with_resume(mode="force_all")
        elif command == "clear":
            supabase = get_supabase_client()
            clear_supabase_safely(supabase)
        else:
            print("Usage:")
            print("  python migrate_chromadb_to_supabase_robust.py                    # Normal migration")
            print("  python migrate_chromadb_to_supabase_robust.py investigate       # Investigate skipped docs")
            print("  python migrate_chromadb_to_supabase_robust.py force_all         # Force migrate all docs")
            print("  python migrate_chromadb_to_supabase_robust.py clear             # Clear Supabase data")
    else:
        success = migrate_data_with_resume()
        if success:
            print("\n🎉 Migration completed successfully!")
            print("Your ChromaDB data is now available in Supabase!")
        else:
            print("\n❌ Migration failed. Check logs for details.")
            print("You can restart the script to resume migration.") 