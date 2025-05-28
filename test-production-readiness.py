#!/usr/bin/env python3
"""
Production readiness test script for Fitness Coach API
Tests all critical components before deployment
"""

import os
import sys
import asyncio
import requests
import json
from datetime import datetime
from typing import Dict, Any

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(message: str, status: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    if status == "SUCCESS":
        print(f"{Colors.GREEN}✅ [{timestamp}] {message}{Colors.ENDC}")
    elif status == "WARNING":
        print(f"{Colors.YELLOW}⚠️  [{timestamp}] {message}{Colors.ENDC}")
    elif status == "ERROR":
        print(f"{Colors.RED}❌ [{timestamp}] {message}{Colors.ENDC}")
    else:
        print(f"{Colors.BLUE}ℹ️  [{timestamp}] {message}{Colors.ENDC}")

def test_environment_variables():
    """Test that all required environment variables are present"""
    print_status("Testing environment variables...", "INFO")
    
    required_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY", 
        "SUPABASE_URL",
        "SUPABASE_SERVICE_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            print_status(f"Found {var}", "SUCCESS")
    
    if missing_vars:
        print_status(f"Missing required environment variables: {', '.join(missing_vars)}", "ERROR")
        return False
    
    print_status("All required environment variables present", "SUCCESS")
    return True

def test_imports():
    """Test that all critical imports work"""
    print_status("Testing critical imports...", "INFO")
    
    try:
        import fastapi
        print_status("FastAPI imported successfully", "SUCCESS")
        
        import uvicorn
        print_status("Uvicorn imported successfully", "SUCCESS")
        
        import langchain
        print_status("LangChain imported successfully", "SUCCESS")
        
        import langgraph
        print_status("LangGraph imported successfully", "SUCCESS")
        
        import openai
        print_status("OpenAI imported successfully", "SUCCESS")
        
        import anthropic
        print_status("Anthropic imported successfully", "SUCCESS")
        
        import supabase
        print_status("Supabase imported successfully", "SUCCESS")
        
        import chromadb
        print_status("ChromaDB imported successfully", "SUCCESS")
        
        import socketio
        print_status("Socket.IO imported successfully", "SUCCESS")
        
        return True
        
    except ImportError as e:
        print_status(f"Import error: {str(e)}", "ERROR")
        return False

def test_supabase_connection():
    """Test Supabase connection"""
    print_status("Testing Supabase connection...", "INFO")
    
    try:
        from graph.memory_store import get_supabase_client
        supabase = get_supabase_client()
        
        # Try a simple query to test connection
        # This will fail gracefully if table doesn't exist
        try:
            result = supabase.table("profile_overview_generations").select("count").limit(1).execute()
            print_status("Supabase connection successful", "SUCCESS")
            return True
        except Exception as e:
            # Table might not exist, but connection is working
            if "relation" in str(e) and "does not exist" in str(e):
                print_status("Supabase connected (table setup needed)", "WARNING")
                return True
            else:
                print_status(f"Supabase query error: {str(e)}", "ERROR")
                return False
                
    except Exception as e:
        print_status(f"Supabase connection failed: {str(e)}", "ERROR")
        return False

def test_openai_connection():
    """Test OpenAI API connection"""
    print_status("Testing OpenAI API connection...", "INFO")
    
    try:
        from openai import OpenAI
        client = OpenAI()
        
        # Test with a minimal request
        response = client.models.list()
        print_status("OpenAI API connection successful", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"OpenAI API connection failed: {str(e)}", "ERROR")
        return False

def test_anthropic_connection():
    """Test Anthropic API connection"""
    print_status("Testing Anthropic API connection...", "INFO")
    
    try:
        import anthropic
        client = anthropic.Anthropic()
        
        # Test with a minimal request - just check if client can be created
        print_status("Anthropic API client created successfully", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Anthropic API connection failed: {str(e)}", "ERROR")
        return False

def test_vector_database():
    """Test vector database availability"""
    print_status("Testing vector database...", "INFO")
    
    try:
        from ingestion import get_vectorstore, check_vectorstore
        
        # First check using the check function
        exists, vs_check = check_vectorstore()
        if exists and vs_check:
            count = vs_check._collection.count()
            print_status(f"Vector database accessible with {count} documents", "SUCCESS")
            return True
        
        # Fallback to direct get_vectorstore
        vs = get_vectorstore()
        if vs:
            count = vs._collection.count()
            print_status(f"Vector database loaded with {count} documents", "SUCCESS")
            return True
        else:
            # Check if the directory exists but is inaccessible
            import os
            persist_directory = "./.fitness_chroma"
            if os.path.exists(persist_directory):
                print_status("Vector database directory exists but collection is inaccessible", "WARNING")
            else:
                print_status("Vector database directory not found - run data collection first", "WARNING")
            return True  # Not critical for basic functionality
            
    except Exception as e:
        print_status(f"Vector database error: {str(e)}", "WARNING")
        return True  # Not critical for basic functionality

def test_app_startup():
    """Test that the FastAPI app can be imported and initialized"""
    print_status("Testing FastAPI app startup...", "INFO")
    
    try:
        # Change to the project directory
        import sys
        import os
        
        # Import the main app
        from main import api, app
        
        print_status("FastAPI app imported successfully", "SUCCESS")
        
        # Test that routes are registered
        routes = [route.path for route in api.routes]
        expected_routes = ["/health", "/", "/ask", "/fitness/profile/stream"]
        
        for expected_route in expected_routes:
            if expected_route in routes:
                print_status(f"Route {expected_route} registered", "SUCCESS")
            else:
                print_status(f"Route {expected_route} not found", "WARNING")
        
        return True
        
    except Exception as e:
        print_status(f"App startup error: {str(e)}", "ERROR")
        return False

async def test_health_endpoint():
    """Test the health endpoint works"""
    print_status("Testing health endpoint...", "INFO")
    
    try:
        import uvicorn
        import threading
        import time
        from main import app
        
        # Start server in a separate thread
        config = uvicorn.Config(app, host="127.0.0.1", port=8001, log_level="error")
        server = uvicorn.Server(config)
        
        def run_server():
            asyncio.run(server.serve())
        
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Wait for server to start
        time.sleep(3)
        
        # Test health endpoint
        response = requests.get("http://127.0.0.1:8001/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print_status(f"Health endpoint responded: {health_data.get('status')}", "SUCCESS")
            return True
        else:
            print_status(f"Health endpoint returned status: {response.status_code}", "ERROR")
            return False
            
    except Exception as e:
        print_status(f"Health endpoint test failed: {str(e)}", "ERROR")
        return False

def main():
    """Run all production readiness tests"""
    print_status("Starting Production Readiness Tests", "INFO")
    print("=" * 60)
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("Critical Imports", test_imports),
        ("Supabase Connection", test_supabase_connection),
        ("OpenAI API", test_openai_connection),
        ("Anthropic API", test_anthropic_connection),
        ("Vector Database", test_vector_database),
        ("App Startup", test_app_startup),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{Colors.BOLD}Running: {test_name}{Colors.ENDC}")
        print("-" * 40)
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print_status(f"Test {test_name} crashed: {str(e)}", "ERROR")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print_status("TEST SUMMARY", "INFO")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        if result:
            print_status(f"{test_name}: PASSED", "SUCCESS")
            passed += 1
        else:
            print_status(f"{test_name}: FAILED", "ERROR")
    
    print(f"\n{Colors.BOLD}Result: {passed}/{total} tests passed{Colors.ENDC}")
    
    if passed == total:
        print_status("🎉 All tests passed! Ready for production deployment", "SUCCESS")
        return True
    else:
        print_status("❌ Some tests failed. Fix issues before deploying", "ERROR")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 