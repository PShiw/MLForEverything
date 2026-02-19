#!/usr/bin/env python3
"""
Test Script for RAG Chat Agent System
Run this to verify your installation is working correctly
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import chromadb
        print("  ✓ chromadb")
    except ImportError as e:
        print(f"  ✗ chromadb - {e}")
        return False
    
    try:
        import sentence_transformers
        print("  ✓ sentence-transformers")
    except ImportError as e:
        print(f"  ✗ sentence-transformers - {e}")
        return False
    
    try:
        import langchain
        print("  ✓ langchain")
    except ImportError as e:
        print(f"  ✗ langchain - {e}")
        return False
    
    try:
        import transformers
        print("  ✓ transformers")
    except ImportError as e:
        print(f"  ✗ transformers - {e}")
        return False
    
    try:
        import torch
        print("  ✓ torch")
    except ImportError as e:
        print(f"  ✗ torch - {e}")
        return False
    
    print("✓ All imports successful!\n")
    return True


def test_modules():
    """Test that all custom modules can be imported"""
    print("Testing custom modules...")
    
    try:
        from document_processor import DocumentProcessor
        print("  ✓ document_processor")
    except ImportError as e:
        print(f"  ✗ document_processor - {e}")
        return False
    
    try:
        from vector_store import VectorStoreManager
        print("  ✓ vector_store")
    except ImportError as e:
        print(f"  ✗ vector_store - {e}")
        return False
    
    try:
        from rag_system import RAGSystem, SimpleRAGSystem
        print("  ✓ rag_system")
    except ImportError as e:
        print(f"  ✗ rag_system - {e}")
        return False
    
    try:
        from chat_agent import ChatAgent, SimpleChatAgent
        print("  ✓ chat_agent")
    except ImportError as e:
        print(f"  ✗ chat_agent - {e}")
        return False
    
    try:
        from fine_tuning import ModelFineTuner, DatasetCreator
        print("  ✓ fine_tuning")
    except ImportError as e:
        print(f"  ✗ fine_tuning - {e}")
        return False
    
    try:
        from config import SystemConfig
        print("  ✓ config")
    except ImportError as e:
        print(f"  ✗ config - {e}")
        return False
    
    try:
        from main import RAGChatApp
        print("  ✓ main")
    except ImportError as e:
        print(f"  ✗ main - {e}")
        return False
    
    print("✓ All modules loaded successfully!\n")
    return True


def test_basic_functionality():
    """Test basic functionality"""
    print("Testing basic functionality...")
    
    try:
        from main import RAGChatApp
        import os
        
        # Create test document
        test_dir = "./test_data"
        os.makedirs(test_dir, exist_ok=True)
        
        test_file = os.path.join(test_dir, "test_doc.txt")
        with open(test_file, 'w') as f:
            f.write("This is a test document about artificial intelligence and machine learning.")
        
        print("  ✓ Created test document")
        
        # Initialize app
        app = RAGChatApp()
        print("  ✓ Initialized RAGChatApp")
        
        # Setup with simple RAG
        app.setup(use_simple_rag=True)
        print("  ✓ Setup complete")
        
        # Ingest document
        app.ingest_documents(test_file)
        print("  ✓ Document ingestion successful")
        
        # Test query
        response = app.query("What is this document about?", k=2)
        print("  ✓ Query successful")
        
        # Check response
        if 'documents' in response and len(response['documents']) > 0:
            print("  ✓ Retrieved documents")
        else:
            print("  ✗ No documents retrieved")
            return False
        
        # Get stats
        stats = app.get_stats()
        if 'document_count' in stats:
            print(f"  ✓ Vector store contains {stats['document_count']} chunks")
        
        # Cleanup
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        print("  ✓ Cleaned up test files")
        
        print("✓ Basic functionality test passed!\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_processing():
    """Test document processing"""
    print("Testing document processing...")
    
    try:
        from document_processor import DocumentProcessor
        import os
        
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        print("  ✓ Created DocumentProcessor")
        
        # Create test file
        test_file = "./test_doc_processing.txt"
        with open(test_file, 'w') as f:
            f.write("Test content " * 50)  # Long enough to create multiple chunks
        
        docs = processor.process_documents(test_file)
        print(f"  ✓ Processed document into {len(docs)} chunks")
        
        # Cleanup
        os.remove(test_file)
        
        print("✓ Document processing test passed!\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Document processing test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("RAG CHAT AGENT SYSTEM - TEST SUITE")
    print("="*80 + "\n")
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test modules
    results.append(("Modules", test_modules()))
    
    # Test document processing
    results.append(("Document Processing", test_document_processing()))
    
    # Test basic functionality
    results.append(("Basic Functionality", test_basic_functionality()))
    
    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("\n🎉 All tests passed! Your RAG Chat Agent is ready to use.")
        print("\nNext steps:")
        print("  1. Read QUICKSTART.md for a quick introduction")
        print("  2. Run: python examples.py 1")
        print("  3. Check README.md for full documentation")
    else:
        print("\n⚠️  Some tests failed. Please check your installation.")
        print("\nTroubleshooting:")
        print("  1. Make sure all dependencies are installed: pip install -r ../requirement.txt")
        print("  2. Check that you're using Python 3.8 or higher")
        print("  3. Verify you have internet connection for downloading models")
    
    print("\n")
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
