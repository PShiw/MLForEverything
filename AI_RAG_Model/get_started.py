#!/usr/bin/env python3
"""
Getting Started Script - Interactive Setup for RAG Chat Agent

This script helps you get started quickly with your RAG chat agent.
"""

import os
import sys

def print_header():
    print("\n" + "="*80)
    print("🤖 RAG CHAT AGENT - GETTING STARTED")
    print("="*80 + "\n")

def print_step(step_num, step_title):
    print(f"\n{'='*80}")
    print(f"STEP {step_num}: {step_title}")
    print("="*80 + "\n")

def check_installation():
    print_step(1, "Checking Installation")
    
    print("Checking if dependencies are installed...")
    
    missing = []
    
    try:
        import chromadb
        print("  ✓ chromadb")
    except ImportError:
        missing.append("chromadb")
        print("  ✗ chromadb")
    
    try:
        import sentence_transformers
        print("  ✓ sentence-transformers")
    except ImportError:
        missing.append("sentence-transformers")
        print("  ✗ sentence-transformers")
    
    try:
        import langchain
        print("  ✓ langchain")
    except ImportError:
        missing.append("langchain")
        print("  ✗ langchain")
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nPlease install dependencies first:")
        print("  pip install -r ../requirement.txt")
        return False
    
    print("\n✅ All required packages are installed!")
    return True

def create_sample_documents():
    print_step(2, "Creating Sample Documents")
    
    data_dir = "./data/sample_docs"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create sample document 1
    with open(f"{data_dir}/ai_basics.txt", 'w') as f:
        f.write("""
Artificial Intelligence Overview

Artificial Intelligence (AI) refers to computer systems that can perform tasks 
that typically require human intelligence. These tasks include visual perception, 
speech recognition, decision-making, and language translation.

Key Areas of AI:
1. Machine Learning - Systems that learn from data
2. Natural Language Processing - Understanding human language
3. Computer Vision - Interpreting visual information
4. Robotics - Intelligent physical systems

AI is transforming industries including healthcare, finance, transportation, 
and entertainment.
""")
    
    # Create sample document 2
    with open(f"{data_dir}/machine_learning.txt", 'w') as f:
        f.write("""
Machine Learning Fundamentals

Machine learning is a subset of AI that focuses on developing algorithms 
that enable computers to learn from and make predictions based on data.

Types of Machine Learning:
1. Supervised Learning - Learning from labeled data
2. Unsupervised Learning - Finding patterns in unlabeled data
3. Reinforcement Learning - Learning through trial and error

Common algorithms include linear regression, decision trees, neural networks, 
and support vector machines.
""")
    
    # Create sample document 3
    with open(f"{data_dir}/data_science.txt", 'w') as f:
        f.write("""
Data Science Essentials

Data science combines statistics, mathematics, programming, and domain 
expertise to extract insights from data.

Core Skills:
- Statistical analysis and hypothesis testing
- Programming (Python, R)
- Data visualization
- Machine learning
- Database management
- Communication skills

Popular tools include Python, Jupyter notebooks, Pandas, NumPy, Scikit-learn, 
and visualization libraries like Matplotlib and Seaborn.
""")
    
    print(f"✅ Created sample documents in: {data_dir}")
    print(f"   - ai_basics.txt")
    print(f"   - machine_learning.txt")
    print(f"   - data_science.txt")
    
    return data_dir

def setup_and_ingest(data_dir):
    print_step(3, "Setting Up RAG System and Ingesting Documents")
    
    try:
        from main import RAGChatApp
        
        print("Initializing RAG Chat App...")
        app = RAGChatApp()
        
        print("Setting up system (using Simple RAG mode)...")
        app.setup(use_simple_rag=True)
        
        print(f"Ingesting documents from {data_dir}...")
        app.ingest_documents(data_dir, is_directory=True)
        
        print("\n✅ System setup complete!")
        
        return app
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_queries(app):
    print_step(4, "Testing with Sample Queries")
    
    test_questions = [
        "What is artificial intelligence?",
        "What are the types of machine learning?",
        "What skills are needed for data science?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 80)
        
        try:
            response = app.query(question, k=2)
            
            if 'documents' in response and response['documents']:
                doc = response['documents'][0]
                print(f"   Relevance Score: {doc['score']:.4f}")
                print(f"   Answer: {doc['content'][:200]}...")
            else:
                print("   No results found")
                
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n✅ Test queries complete!")

def show_next_steps():
    print_step(5, "Next Steps")
    
    print("""
Congratulations! Your RAG Chat Agent is working! 🎉

Now you can:

1. Start Interactive Chat:
   >>> app.start_chat()
   
2. Add Your Own Documents:
   >>> app.ingest_documents("./path/to/your/documents", is_directory=True)
   
3. Try More Examples:
   $ python examples.py 1
   $ python examples.py 2
   $ python examples.py all
   
4. Read the Documentation:
   - QUICKSTART.md - Quick introduction (5 min)
   - TUTORIAL.md - Complete step-by-step guide (30 min)
   - README.md - Full reference documentation
   
5. Use Command Line:
   $ python main.py --mode chat --simple
   
6. Customize Configuration:
   - Edit config.py or pass custom config to RAGChatApp()
   
7. Try Full RAG with LLM:
   >>> app = RAGChatApp()
   >>> app.setup(use_simple_rag=False)

---

Quick Reference Commands:
   
# Test installation
$ python test_installation.py

# Run example
$ python examples.py 1

# Ingest documents
$ python main.py --mode ingest --documents ./data

# Start chat
$ python main.py --mode chat --simple

# Show stats
$ python main.py --mode stats

---

For help, check:
- README.md for full documentation
- TUTORIAL.md for step-by-step guide
- examples.py for code examples

Happy chatting! 🚀
""")

def main():
    print_header()
    
    # Step 1: Check installation
    if not check_installation():
        print("\n❌ Please install dependencies first and try again.")
        sys.exit(1)
    
    input("\nPress Enter to continue...")
    
    # Step 2: Create sample documents
    data_dir = create_sample_documents()
    
    input("\nPress Enter to continue...")
    
    # Step 3: Setup and ingest
    app = setup_and_ingest(data_dir)
    
    if app is None:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
    
    input("\nPress Enter to continue...")
    
    # Step 4: Test queries
    test_queries(app)
    
    input("\nPress Enter to continue...")
    
    # Step 5: Show next steps
    show_next_steps()
    
    # Optional: Start interactive chat
    print("\n" + "="*80)
    response = input("\nWould you like to start the interactive chat now? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        print("\nStarting interactive chat...")
        print("Type 'quit' to exit, 'help' for commands\n")
        try:
            app.start_chat()
        except KeyboardInterrupt:
            print("\n\nChat ended. Goodbye!")
    else:
        print("\nYou can start chat anytime with:")
        print("  python main.py --mode chat --simple")
        print("\nGoodbye! 👋")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted. Goodbye!")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
