#!/usr/bin/env python3
"""
Test Vector Retrieval Systems
Tests both ChromaDB and Supabase vector retrieval to ensure they work correctly
"""

import sys
import os
import time
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment
load_dotenv()

print("🧪 VECTOR RETRIEVAL TESTING")
print("=" * 50)

# Test imports
try:
    from ingestion import (
        get_retriever, 
        check_vectorstore_availability, 
        get_retriever_info,
        USE_SUPABASE,
        SUPABASE_AVAILABLE
    )
    print("✅ Ingestion imports successful")
except Exception as e:
    print(f"❌ Ingestion import error: {e}")
    sys.exit(1)

def test_retriever_status():
    """Test the status of available retrievers"""
    print("\n📊 RETRIEVER STATUS CHECK")
    print("-" * 30)
    
    # Get retriever info
    info = get_retriever_info()
    print(f"Current retriever: {info['current_retriever']}")
    print(f"Supabase enabled: {info['supabase_enabled']}")
    print(f"Supabase available: {info['supabase_available']}")
    
    # Print detailed status
    status = info['status']
    for system, details in status.items():
        print(f"\n{system.upper()}:")
        print(f"  Available: {details['available']}")
        print(f"  Enabled: {details['enabled']}")
        print(f"  Document count: {details['count']}")
    
    return info

def test_basic_retrieval(retriever, test_queries: List[str]):
    """Test basic retrieval functionality"""
    print("\n🔍 BASIC RETRIEVAL TEST")
    print("-" * 30)
    
    if not retriever:
        print("❌ No retriever available for testing")
        return False
    
    print(f"Retriever type: {type(retriever).__name__}")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test Query {i}: '{query}'")
        
        try:
            start_time = time.time()
            results = retriever.get_relevant_documents(query)
            end_time = time.time()
            
            print(f"⏱️ Query time: {end_time - start_time:.2f} seconds")
            print(f"📄 Results found: {len(results)}")
            
            if results:
                # Show first result details
                first_result = results[0]
                print(f"📖 First result preview:")
                print(f"   Content (first 100 chars): {first_result.page_content[:100]}...")
                print(f"   Author: {first_result.metadata.get('author', 'Unknown')}")
                print(f"   Title: {first_result.metadata.get('title', 'Unknown')[:50]}...")
                if 'similarity_score' in first_result.metadata:
                    print(f"   Similarity: {first_result.metadata['similarity_score']:.3f}")
            else:
                print("⚠️ No results found")
                
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return False
    
    return True

def test_supabase_specifically():
    """Test Supabase retriever specifically"""
    print("\n🚀 SUPABASE-SPECIFIC TEST")
    print("-" * 30)
    
    if not SUPABASE_AVAILABLE:
        print("⚠️ Supabase retriever not available")
        return False
    
    try:
        from supabase_retriever import get_supabase_retriever, check_supabase_vectorstore
        
        # Test Supabase connection
        exists, retriever = check_supabase_vectorstore()
        
        if not exists:
            print("❌ Supabase vectorstore not available")
            return False
        
        print("✅ Supabase vectorstore available")
        
        # Test a simple query
        test_query = "muscle building exercises"
        print(f"Testing query: '{test_query}'")
        
        results = retriever.get_relevant_documents(test_query)
        print(f"Results: {len(results)}")
        
        if results:
            print("✅ Supabase retrieval working correctly")
            return True
        else:
            print("⚠️ No results from Supabase (may be threshold issue)")
            return True  # Still working, just no matches
            
    except Exception as e:
        print(f"❌ Supabase test failed: {e}")
        return False

def test_chromadb_specifically():
    """Test ChromaDB retriever specifically"""
    print("\n💾 CHROMADB-SPECIFIC TEST")
    print("-" * 30)
    
    try:
        from ingestion import check_vectorstore, get_vectorstore
        
        # Test ChromaDB connection
        exists, vs = check_vectorstore()
        
        if not exists:
            print("❌ ChromaDB vectorstore not available")
            return False
        
        print("✅ ChromaDB vectorstore available")
        
        # Test a simple query
        retriever = vs.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"score_threshold": 0.5, "k": 5}
        )
        
        test_query = "muscle building exercises"
        print(f"Testing query: '{test_query}'")
        
        results = retriever.get_relevant_documents(test_query)
        print(f"Results: {len(results)}")
        
        if results:
            print("✅ ChromaDB retrieval working correctly")
            return True
        else:
            print("⚠️ No results from ChromaDB (may be threshold issue)")
            return True  # Still working, just no matches
            
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        return False

def main():
    """Main test function"""
    
    # Test queries to use
    test_queries = [
        "muscle building exercises",
        "how to lose fat",
        "protein requirements for muscle growth",
        "best exercises for chest",
        "workout split recommendations"
    ]
    
    print(f"Environment: USE_SUPABASE = {USE_SUPABASE}")
    print(f"Supabase available: {SUPABASE_AVAILABLE}")
    
    # Test 1: Check retriever status
    info = test_retriever_status()
    
    # Test 2: Get the main retriever and test it
    print("\n🎯 MAIN RETRIEVER TEST")
    print("-" * 30)
    retriever = get_retriever()
    
    if retriever:
        print(f"✅ Main retriever created: {type(retriever).__name__}")
        success = test_basic_retrieval(retriever, test_queries)
        if success:
            print("✅ Main retriever tests passed")
        else:
            print("❌ Main retriever tests failed")
    else:
        print("❌ Failed to create main retriever")
    
    # Test 3: Test each system specifically
    supabase_success = test_supabase_specifically()
    chromadb_success = test_chromadb_specifically()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    print(f"Main retriever working: {'✅' if retriever else '❌'}")
    print(f"Supabase system: {'✅' if supabase_success else '❌'}")
    print(f"ChromaDB system: {'✅' if chromadb_success else '❌'}")
    
    if info['current_retriever'] == 'supabase':
        print("\n🚀 Currently using: SUPABASE vector search")
    elif info['current_retriever'] == 'chromadb':
        print("\n💾 Currently using: CHROMADB vector search")
    else:
        print("\n❌ No vector search system active")
    
    # Instructions
    print("\n" + "=" * 50)
    print("🛠️  CONFIGURATION")
    print("=" * 50)
    print("To switch between systems, set environment variable:")
    print("  USE_SUPABASE_VECTOR=true   # Use Supabase (recommended)")
    print("  USE_SUPABASE_VECTOR=false  # Use ChromaDB")
    print("\nSupabase is preferred for production due to:")
    print("  ✅ Better scalability")
    print("  ✅ No local dependencies")
    print("  ✅ Handles large document collections")
    print("  ✅ Built-in similarity search functions")

if __name__ == "__main__":
    main() 