"""
Example Usage Scripts
Demonstrates various use cases of the RAG Chat Agent
"""

import os
from main import RAGChatApp
from config import SystemConfig
from fine_tuning import DatasetCreator


def example_1_basic_setup():
    """
    Example 1: Basic Setup and Document Ingestion
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Setup and Document Ingestion")
    print("="*80 + "\n")
    
    # Initialize app
    config = SystemConfig()
    app = RAGChatApp(config)
    
    # Setup with simple RAG (no LLM needed)
    app.setup(use_simple_rag=True)
    
    # Create a sample document
    sample_doc_path = "./data/sample_document.txt"
    os.makedirs("./data", exist_ok=True)
    
    with open(sample_doc_path, 'w') as f:
        f.write("""
Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems 
to learn and improve from experience without being explicitly programmed.

Types of Machine Learning:
1. Supervised Learning: Learning from labeled data
2. Unsupervised Learning: Finding patterns in unlabeled data
3. Reinforcement Learning: Learning through trial and error

Common Algorithms:
- Linear Regression
- Decision Trees
- Neural Networks
- Support Vector Machines

Machine learning is used in many applications including image recognition,
natural language processing, recommendation systems, and autonomous vehicles.
""")
    
    # Ingest the document
    app.ingest_documents(sample_doc_path)
    
    # Query the system
    response = app.query("What are the types of machine learning?")
    
    print("\nQuery Results:")
    print("-" * 80)
    for i, doc in enumerate(response['documents'], 1):
        print(f"\nDocument {i} (Score: {doc['score']:.4f}):")
        print(doc['content'][:300] + "...")
    
    print("\n✓ Example 1 complete!")


def example_2_multiple_documents():
    """
    Example 2: Ingesting Multiple Documents from Directory
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Multiple Documents from Directory")
    print("="*80 + "\n")
    
    # Create sample documents
    docs_dir = "./data/documents"
    os.makedirs(docs_dir, exist_ok=True)
    
    # Document 1: Python basics
    with open(f"{docs_dir}/python_basics.txt", 'w') as f:
        f.write("""
Python Programming

Python is a high-level, interpreted programming language known for its simplicity
and readability. It was created by Guido van Rossum and first released in 1991.

Key Features:
- Easy to learn and read
- Extensive standard library
- Support for multiple programming paradigms
- Large community and ecosystem

Popular uses include web development, data science, machine learning, and automation.
""")
    
    # Document 2: Data Science
    with open(f"{docs_dir}/data_science.txt", 'w') as f:
        f.write("""
Data Science Overview

Data science combines statistics, mathematics, programming, and domain expertise
to extract insights from data.

Key Skills:
- Statistical analysis
- Programming (Python, R)
- Data visualization
- Machine learning
- Database management

Common tools include Jupyter, Pandas, NumPy, Scikit-learn, and TensorFlow.
""")
    
    # Initialize and ingest
    app = RAGChatApp()
    app.setup(use_simple_rag=True)
    app.ingest_documents(docs_dir, is_directory=True)
    
    # Test queries
    queries = [
        "What is Python?",
        "What skills are needed for data science?",
        "Tell me about machine learning tools"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        response = app.query(query, k=2)
        for i, doc in enumerate(response['documents'][:2], 1):
            print(f"{i}. {doc['content'][:150]}...")
    
    print("\n✓ Example 2 complete!")


def example_3_updating_documents():
    """
    Example 3: Updating Documents with New Information
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Updating Documents")
    print("="*80 + "\n")
    
    # Initialize app
    app = RAGChatApp()
    app.setup(use_simple_rag=True)
    
    # Create initial document
    doc_path = "./data/product_info.txt"
    with open(doc_path, 'w') as f:
        f.write("Product X: Version 1.0 - Released January 2024")
    
    # Ingest
    app.ingest_documents(doc_path, metadata={"source": "product_info", "version": "1.0"})
    
    print("Initial document ingested")
    response = app.query("What is the current version?", k=1)
    print(f"Response: {response['documents'][0]['content']}")
    
    # Update document
    with open(doc_path, 'w') as f:
        f.write("Product X: Version 2.0 - Released February 2024 - New features added")
    
    # Update in system
    app.update_documents(
        doc_path,
        source_filter={"source": "product_info"}
    )
    
    print("\nDocument updated")
    response = app.query("What is the current version?", k=1)
    print(f"Response: {response['documents'][0]['content']}")
    
    print("\n✓ Example 3 complete!")


def example_4_custom_configuration():
    """
    Example 4: Custom Configuration
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Configuration")
    print("="*80 + "\n")
    
    # Create custom configuration
    config = SystemConfig()
    config.document.chunk_size = 500
    config.document.chunk_overlap = 100
    config.vector_store.collection_name = "custom_collection"
    config.vector_store.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    config.chat.max_history = 20
    
    print("Custom configuration:")
    print(f"  Chunk size: {config.document.chunk_size}")
    print(f"  Collection: {config.vector_store.collection_name}")
    print(f"  Embedding model: {config.vector_store.embedding_model}")
    print(f"  Max history: {config.chat.max_history}")
    
    # Initialize with custom config
    app = RAGChatApp(config)
    app.setup(use_simple_rag=True)
    
    stats = app.get_stats()
    print(f"\nSystem stats: {stats}")
    
    print("\n✓ Example 4 complete!")


def example_5_fine_tuning():
    """
    Example 5: Fine-tuning a Model
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Fine-tuning (Setup Only - Training Requires More Resources)")
    print("="*80 + "\n")
    
    # Create training data
    questions = [
        "What is machine learning?",
        "What is deep learning?",
        "What is natural language processing?"
    ]
    
    answers = [
        "Machine learning is a subset of AI that learns from data.",
        "Deep learning uses neural networks with multiple layers.",
        "NLP enables computers to understand and process human language."
    ]
    
    contexts = [
        "Machine learning enables computers to learn patterns from data.",
        "Deep learning models can learn hierarchical representations.",
        "NLP combines linguistics and machine learning."
    ]
    
    # Create dataset
    training_data = DatasetCreator.create_qa_dataset(questions, answers, contexts)
    
    print(f"Created training dataset with {len(training_data)} examples")
    print(f"\nSample training example:")
    print(training_data[0]['text'])
    
    print("\nTo actually fine-tune:")
    print("  app = RAGChatApp()")
    print("  app.setup_fine_tuning(base_model='gpt2')")
    print("  model_path = app.fine_tune_model(training_data)")
    
    print("\n✓ Example 5 complete!")


def example_6_chat_history():
    """
    Example 6: Working with Chat History
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Chat History Management")
    print("="*80 + "\n")
    
    # Setup
    app = RAGChatApp()
    app.setup(use_simple_rag=True)
    
    # Add some sample data
    sample_doc = "./data/history_test.txt"
    with open(sample_doc, 'w') as f:
        f.write("AI can be used for image recognition, text generation, and data analysis.")
    
    app.ingest_documents(sample_doc)
    
    # Simulate conversation
    queries = [
        "What can AI do?",
        "Tell me about image recognition",
        "What about text generation?"
    ]
    
    for query in queries:
        app.query(query)
    
    # Show history
    history = app.chat_agent.get_history()
    print(f"Conversation has {len(history)} turns")
    
    # Save history
    app.chat_agent.save_conversation("./data/conversation_history.json")
    print("Conversation saved to ./data/conversation_history.json")
    
    # Export to markdown
    app.chat_agent.export_conversation_markdown("./data/conversation.md")
    print("Conversation exported to ./data/conversation.md")
    
    print("\n✓ Example 6 complete!")


def run_all_examples():
    """Run all examples"""
    print("\n" + "="*80)
    print("RUNNING ALL EXAMPLES")
    print("="*80)
    
    try:
        example_1_basic_setup()
        example_2_multiple_documents()
        example_3_updating_documents()
        example_4_custom_configuration()
        example_5_fine_tuning()
        example_6_chat_history()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        
        examples = {
            '1': example_1_basic_setup,
            '2': example_2_multiple_documents,
            '3': example_3_updating_documents,
            '4': example_4_custom_configuration,
            '5': example_5_fine_tuning,
            '6': example_6_chat_history,
            'all': run_all_examples
        }
        
        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Unknown example: {example_num}")
            print("Available: 1, 2, 3, 4, 5, 6, all")
    else:
        print("\nUsage: python examples.py <example_number>")
        print("\nAvailable examples:")
        print("  1 - Basic setup and document ingestion")
        print("  2 - Multiple documents from directory")
        print("  3 - Updating documents")
        print("  4 - Custom configuration")
        print("  5 - Fine-tuning setup")
        print("  6 - Chat history management")
        print("  all - Run all examples")
        print("\nExample: python examples.py 1")
