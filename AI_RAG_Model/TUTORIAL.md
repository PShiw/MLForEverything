# Step-by-Step Guide: Building Your RAG Chat Agent

This guide will walk you through every step of creating a production-ready RAG (Retrieval-Augmented Generation) chat agent.

---

## 📖 Table of Contents

1. [Understanding the System](#step-1-understanding-the-system)
2. [Installation](#step-2-installation)
3. [Document Preparation](#step-3-document-preparation)
4. [Creating Your First Agent](#step-4-creating-your-first-agent)
5. [Querying the System](#step-5-querying-the-system)
6. [Interactive Chat](#step-6-interactive-chat)
7. [Updating Documents](#step-7-updating-documents)
8. [Fine-Tuning](#step-8-fine-tuning-optional)
9. [Production Deployment](#step-9-production-deployment)
10. [Troubleshooting](#step-10-troubleshooting)

---

## Step 1: Understanding the System

### What is RAG?

RAG (Retrieval-Augmented Generation) combines:
- **Retrieval**: Finding relevant documents from your knowledge base
- **Generation**: Using AI to create natural language responses

### System Components

```
Your Documents → Document Processor → Vector Database → RAG System → Chat Agent
                                                            ↓
                                                      Fine-Tuning (Optional)
```

**Document Processor**: Loads and chunks your documents
**Vector Database**: Stores document embeddings for fast retrieval
**RAG System**: Retrieves relevant docs and generates answers
**Chat Agent**: Provides conversational interface
**Fine-Tuning**: Improves model for your specific domain

### Two Modes Available

1. **Simple RAG**: Fast, retrieval-only (no LLM needed)
   - Perfect for testing and low-resource environments
   - Returns relevant document chunks

2. **Full RAG**: Complete LLM-powered responses
   - Natural language generation
   - Context-aware answers

---

## Step 2: Installation

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- GPU (optional, for faster performance)

### Install Dependencies

```bash
cd AI_RAG_Model
pip install -r ../requirement.txt
```

This will install:
- **chromadb**: Vector database
- **sentence-transformers**: Text embeddings
- **langchain**: RAG framework
- **transformers**: Language models
- **peft**: Fine-tuning capabilities
- And more...

### Verify Installation

```bash
python test_installation.py
```

If all tests pass, you're ready to proceed! ✓

---

## Step 3: Document Preparation

### Supported Formats

Your documents can be in any of these formats:
- Text files (`.txt`)
- PDF documents (`.pdf`)
- Word documents (`.docx`, `.doc`)
- HTML files (`.html`)
- CSV files (`.csv`)

### Organizing Documents

Create a directory structure:

```
data/
├── product_docs/
│   ├── user_manual.pdf
│   └── faq.txt
├── company_policies/
│   ├── hr_policy.docx
│   └── security_policy.pdf
└── research/
    ├── paper1.pdf
    └── paper2.pdf
```

### Document Best Practices

1. **Clean Text**: Remove unnecessary formatting
2. **Structure**: Use headings and sections
3. **Context**: Include relevant metadata
4. **Size**: Break very large documents into smaller files
5. **Updates**: Keep a versioning system

---

## Step 4: Creating Your First Agent

### Option A: Quick Start (Command Line)

```bash
# 1. Ingest documents
python main.py --mode ingest --documents ./data

# 2. Start chat
python main.py --mode chat --simple
```

### Option B: Python Script

Create `my_agent.py`:

```python
from main import RAGChatApp

# Step 1: Initialize
app = RAGChatApp()

# Step 2: Setup (use simple mode for testing)
app.setup(use_simple_rag=True)

# Step 3: Ingest documents
app.ingest_documents("./data", is_directory=True)

# Step 4: Test a query
response = app.query("What information do you have?")

# Step 5: Print results
for i, doc in enumerate(response['documents'], 1):
    print(f"\n{i}. Score: {doc['score']:.4f}")
    print(f"Content: {doc['content'][:200]}...")

print("\n✓ Agent created successfully!")
```

Run it:
```bash
python my_agent.py
```

### Option C: Interactive Notebook

For experimentation, use the examples:

```bash
python examples.py 1  # Basic setup example
```

---

## Step 5: Querying the System

### Single Queries

```python
from main import RAGChatApp

app = RAGChatApp()
app.setup(use_simple_rag=True)
app.ingest_documents("./data", is_directory=True)

# Ask a question
response = app.query("How do I reset my password?", k=4)

# Print answer
for doc in response['documents']:
    print(f"Relevance: {doc['score']:.4f}")
    print(f"Content: {doc['content']}\n")
```

### Batch Queries

```python
questions = [
    "What are the company holidays?",
    "How do I submit expenses?",
    "What is the remote work policy?"
]

for question in questions:
    print(f"\nQ: {question}")
    response = app.query(question, k=2)
    print(f"A: {response['documents'][0]['content'][:200]}...")
```

### Custom Retrieval Parameters

```python
# Retrieve more documents for complex questions
response = app.query("Explain the complete onboarding process", k=8)

# Get top result only for simple lookups
response = app.query("What is the support email?", k=1)
```

---

## Step 6: Interactive Chat

### Starting Chat Session

```python
from main import RAGChatApp

app = RAGChatApp()
app.setup(use_simple_rag=True)
app.ingest_documents("./data", is_directory=True)

# Start interactive chat
app.start_chat()
```

### Chat Commands

While in chat:
- Type your questions normally
- `history` - View conversation history
- `clear` - Clear conversation history
- `stats` - Show system statistics
- `quit` or `q` - Exit

### Example Chat Session

```
You: What products do we offer?
Assistant: [Returns relevant documents about products]

You: Tell me more about Product X
Assistant: [Returns specific information about Product X]

You: history
[Shows conversation history]

You: quit
Goodbye!
```

### Saving Conversations

```python
# After chatting
app.chat_agent.save_conversation("conversation_2024_02_19.json")
app.chat_agent.export_conversation_markdown("conversation.md")
```

---

## Step 7: Updating Documents

### Scenario: New Policy Released

```python
from main import RAGChatApp

app = RAGChatApp()
app.setup(use_simple_rag=True)

# Initial ingestion
app.ingest_documents("./data/policies", is_directory=True)

# Later: policy updated
app.update_documents(
    "./data/policies/remote_work_policy.pdf",
    source_filter={"source": "remote_work_policy.pdf"}
)

print("✓ Policy updated in knowledge base")
```

### Scenario: Adding New Documents

```python
# Add new product documentation
app.ingest_documents(
    "./data/new_products",
    is_directory=True,
    metadata={"category": "products", "date": "2024-02-19"}
)
```

### Scenario: Removing Outdated Information

```python
from vector_store import VectorStoreManager

vector_store = VectorStoreManager()

# Remove all docs from 2023
vector_store.delete_documents({"year": "2023"})

# Remove specific category
vector_store.delete_documents({"category": "deprecated"})
```

### Best Practices for Updates

1. **Version Control**: Keep track of document versions
2. **Metadata**: Add dates and categories
3. **Testing**: Test queries after updates
4. **Backup**: Keep backups of vector database
5. **Monitoring**: Track which documents are retrieved

---

## Step 8: Fine-Tuning (Optional)

### When to Fine-Tune?

Fine-tune when:
- You have domain-specific terminology
- You want more consistent responses
- Standard models don't perform well
- You have sufficient training data (100+ examples)

### Creating Training Data

```python
from fine_tuning import DatasetCreator

# Method 1: Q&A pairs
questions = [
    "What is our return policy?",
    "How long is shipping?",
    "Do you ship internationally?"
]

answers = [
    "We offer 30-day returns on all items.",
    "Standard shipping takes 3-5 business days.",
    "Yes, we ship to over 50 countries worldwide."
]

training_data = DatasetCreator.create_qa_dataset(questions, answers)

# Method 2: From JSON file
# training_data = DatasetCreator.load_from_json("training_data.json")
```

### Fine-Tuning Process

```python
from main import RAGChatApp

app = RAGChatApp()

# Setup fine-tuning
app.setup_fine_tuning(base_model="gpt2")  # Or other models

# Fine-tune (this may take a while!)
model_path = app.fine_tune_model(
    training_data,
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4
)

print(f"✓ Model fine-tuned and saved to: {model_path}")
```

### Using Fine-Tuned Model

```python
from fine_tuning import ModelFineTuner

# Load your fine-tuned model
fine_tuner = ModelFineTuner()
fine_tuner.load_fine_tuned_model("./fine_tuned_models/your_model")

# Use it in RAG system
# (Replace the standard model with your fine-tuned one)
```

### Fine-Tuning Tips

- Start with 3-5 epochs
- Use batch size 4-8 (depends on GPU memory)
- Monitor for overfitting
- Keep validation set for testing
- LoRA uses only ~1% of parameters (efficient!)

---

## Step 9: Production Deployment

### Configuration for Production

```python
from config import SystemConfig

config = SystemConfig()

# Optimize for production
config.document.chunk_size = 800
config.document.chunk_overlap = 150
config.rag.retrieval_k = 5
config.vector_store.embedding_model = "sentence-transformers/all-mpnet-base-v2"
config.chat.max_history = 20

# Initialize with production config
app = RAGChatApp(config)
```

### Persistent Storage

Your vector database is automatically persisted in:
```
chroma_db/
```

To use a different location:
```python
config.vector_store.persist_directory = "/path/to/persistent/storage"
```

### API Wrapper (Example)

Create `api.py`:

```python
from flask import Flask, request, jsonify
from main import RAGChatApp

app = Flask(__name__)
rag_app = RAGChatApp()
rag_app.setup(use_simple_rag=True)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question', '')
    
    response = rag_app.query(question, k=4)
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Monitoring

```python
# Get system statistics
stats = app.get_stats()
print(f"Documents in DB: {stats['document_count']}")
print(f"Conversation turns: {stats.get('conversation_turns', 0)}")

# Log queries
import logging
logging.basicConfig(filename='rag_queries.log', level=logging.INFO)
```

### Performance Optimization

1. **Use GPU**: Set `config.model.device = "cuda"`
2. **Batch Processing**: Process multiple queries together
3. **Caching**: Cache frequent queries
4. **Async**: Use async operations for I/O
5. **Load Balancing**: Distribute across multiple instances

---

## Step 10: Troubleshooting

### Issue: Import Errors

**Problem**: Cannot import modules

**Solution**:
```bash
pip install -r ../requirement.txt --upgrade
```

### Issue: Out of Memory

**Problem**: System runs out of RAM

**Solutions**:
1. Use Simple RAG mode:
   ```python
   app.setup(use_simple_rag=True)
   ```

2. Use smaller embedding model:
   ```python
   config.vector_store.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
   ```

3. Reduce batch size:
   ```python
   config.fine_tuning.batch_size = 2
   ```

### Issue: Slow Performance

**Problem**: Queries take too long

**Solutions**:
1. Reduce retrieval k:
   ```python
   config.rag.retrieval_k = 2
   ```

2. Smaller chunk size:
   ```python
   config.document.chunk_size = 500
   ```

3. Use GPU:
   ```python
   config.model.device = "cuda"
   ```

### Issue: Poor Quality Answers

**Problem**: Answers not relevant or accurate

**Solutions**:
1. Increase retrieval:
   ```python
   config.rag.retrieval_k = 6
   ```

2. Better embedding model:
   ```python
   config.vector_store.embedding_model = "sentence-transformers/all-mpnet-base-v2"
   ```

3. Improve document quality:
   - Clean text
   - Better structure
   - Remove noise

4. Fine-tune the model on your domain

### Issue: Documents Not Found

**Problem**: System can't find relevant documents

**Solutions**:
1. Check document ingestion:
   ```python
   stats = app.get_stats()
   print(f"Documents: {stats['document_count']}")
   ```

2. Verify file formats supported

3. Check for encoding issues:
   ```python
   processor = DocumentProcessor()
   docs = processor.load_document("file.txt")  # Test individual files
   ```

### Getting Help

1. Check `test_installation.py` output
2. Review examples in `examples.py`
3. Read module docstrings
4. Check logs in `./logs/` directory

---

## 🎉 Congratulations!

You now have a complete RAG chat agent system!

### Quick Reference

```bash
# Test installation
python test_installation.py

# Run examples
python examples.py 1

# Ingest documents
python main.py --mode ingest --documents ./data

# Start chat
python main.py --mode chat --simple

# Check stats
python main.py --mode stats
```

### Next Steps

1. **Experiment**: Try different configurations
2. **Customize**: Adapt to your use case
3. **Scale**: Add more documents
4. **Fine-tune**: Improve for your domain
5. **Deploy**: Put it in production!

### Resources

- **README.md**: Full documentation
- **QUICKSTART.md**: Quick introduction  
- **examples.py**: Working code examples
- **Module files**: API documentation

---

**Happy Building! 🚀**

Need help? Review the troubleshooting section or check the examples!
