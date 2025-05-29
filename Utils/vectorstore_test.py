from ingestion import get_vectorstore
import os

def test_vector_database():
    # Test if vector store exists and is accessible
    print("Testing vector store access...")
    vectorstore = get_vectorstore()
    
    if vectorstore is None:
        print("ERROR: Vector store not accessible!")
        return False
    
    # Check collection info (size)
    collection_info = vectorstore._collection.count()
    print(f"Vector store contains {collection_info} documents")
    
    # Test simple queries to see if results are returned
    test_queries = [
        "What does Jeff Nippard recommend for building muscle?",
        "AthleanX tips for shoulder pain",
        "Renaissance Periodization diet advice"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: {query}")
        results = vectorstore.similarity_search(query, k=2)
        
        if not results:
            print(f"No results found for query: {query}")
        else:
            print(f"Found {len(results)} results")
            for i, doc in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"Content snippet: {doc.page_content[:100]}...")
                print(f"Source: {doc.metadata.get('title', 'Unknown')} by {doc.metadata.get('author', 'Unknown')}")
    
    return True

if __name__ == "__main__":
    test_vector_database()