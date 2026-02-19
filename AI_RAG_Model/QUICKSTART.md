# Quick Start Guide - RAG Chat Agent

This guide will walk you through setting up your first RAG chat agent in 5 minutes!

## Step-by-Step Setup

### Step 1: Install Dependencies

First, make sure you've installed all required packages:

```bash
pip install -r ../requirement.txt
```

### Step 2: Create Your First Agent

```python
from main import RAGChatApp

# Initialize the application
app = RAGChatApp()

# Setup with Simple RAG (no LLM needed - perfect for testing!)
app.setup(use_simple_rag=True)

print("✓ RAG Chat Agent initialized!")
```

### Step 3: Add Your Documents

Create a sample document and ingest it:

```python
# Create a sample document
import os
os.makedirs("./data", exist_ok=True)

with open("./data/sample.txt", "w") as f:
    f.write("""
    Artificial Intelligence (AI) is the simulation of human intelligence by machines.
    
    Key areas of AI include:
    - Machine Learning: Systems that learn from data
    - Natural Language Processing: Understanding human language
    - Computer Vision: Interpreting visual information
    - Robotics: Intelligent physical systems
    
    AI is used in many applications today including virtual assistants,
    recommendation systems, autonomous vehicles, and medical diagnosis.
    """)

# Ingest the document
app.ingest_documents("./data/sample.txt")
print("✓ Document ingested!")
```

### Step 4: Ask Questions!

```python
# Query the system
response = app.query("What is AI?")

print("\n" + "="*80)
print("ANSWER:")
print("="*80)
for i, doc in enumerate(response['documents'], 1):
    print(f"\nDocument {i} (Relevance: {doc['score']:.4f}):")
    print(doc['content'])
```

### Step 5: Start Interactive Chat (Optional)

```python
# For interactive chat session
# app.start_chat()
# 
# Commands:
# - Type your questions
# - 'history' to view conversation
# - 'stats' for system statistics
# - 'quit' to exit
```

## What's Next?

### Option 1: Add More Documents

```python
# Add multiple documents
app.ingest_documents("./path/to/documents", is_directory=True)
```

### Option 2: Use Full LLM Mode

```python
# For more sophisticated responses with natural language generation
app2 = RAGChatApp()
app2.setup(use_simple_rag=False)  # This will load a language model
```

### Option 3: Customize Configuration

```python
from config import SystemConfig

config = SystemConfig()
config.document.chunk_size = 500
config.rag.retrieval_k = 6

custom_app = RAGChatApp(config)
custom_app.setup(use_simple_rag=True)
```

### Option 4: Update Documents

```python
# When you have new versions of documents
app.update_documents(
    "./data/updated_sample.txt",
    source_filter={"source": "sample.txt"}
)
```

## Common Use Cases

### 1. Personal Knowledge Base

```python
# Ingest your notes, articles, bookmarks
app.ingest_documents("./my_notes", is_directory=True)
response = app.query("What did I learn about Python decorators?")
```

### 2. Project Documentation

```python
# Index your project docs
app.ingest_documents("./project_docs", is_directory=True)
response = app.query("How do I configure the database?")
```

### 3. Research Assistant

```python
# Add research papers
app.ingest_documents("./research_papers", is_directory=True)
response = app.query("What methods are used for text classification?")
```

## Running Examples

The `examples.py` file contains complete working examples:

```bash
# Basic setup
python examples.py 1

# Multiple documents
python examples.py 2

# Updating documents
python examples.py 3

# Custom configuration
python examples.py 4

# Fine-tuning preparation
python examples.py 5

# Chat history
python examples.py 6

# Run all
python examples.py all
```

## Command-Line Usage

```bash
# Ingest documents
python main.py --mode ingest --documents ./data

# Start chat
python main.py --mode chat

# Simple mode (no LLM)
python main.py --mode chat --simple

# Update documents
python main.py --mode update --documents ./data

# Show statistics
python main.py --mode stats
```

## Tips for Best Results

1. **Document Quality**: Clean, well-structured documents work best
2. **Chunk Size**: 500-1000 characters is optimal for most documents
3. **Query Clarity**: Be specific in your questions
4. **Retrieval K**: Start with 4, increase if you need more context
5. **Test Queries**: Try different phrasings to find what works best

## Troubleshooting Quick Fixes

### If you get memory errors:
```python
# Use simple mode
app.setup(use_simple_rag=True)
```

### If responses are slow:
```python
config = SystemConfig()
config.document.chunk_size = 500  # Smaller chunks
config.rag.retrieval_k = 2  # Fewer documents
app = RAGChatApp(config)
```

### If answers are not relevant:
```python
# Increase documents retrieved
config = SystemConfig()
config.rag.retrieval_k = 6
app = RAGChatApp(config)
```

## Next Steps

1. Read the full README.md for detailed documentation
2. Explore examples.py for more use cases
3. Check individual module files for advanced features
4. Try fine-tuning for your specific domain

---

Happy chatting! 🚀

For more help, check:
- README.md - Full documentation
- examples.py - Working examples
- Individual module files for API details
