# RAG Chat Agent System

A complete, production-ready Retrieval-Augmented Generation (RAG) chat agent system built with open-source tools. This system allows you to create an AI assistant that answers questions based on your documents.

## 🌟 Features

- **Document Processing**: Support for multiple formats (PDF, DOCX, TXT, HTML, CSV)
- **Vector Database**: Persistent storage using ChromaDB with easy updates
- **Flexible RAG**: Full LLM-powered or Simple retrieval-only modes
- **Chat Interface**: Interactive conversational agent with history
- **Fine-tuning**: Built-in support for LoRA/QLoRA fine-tuning
- **Open Source**: 100% open-source packages, no API keys required
- **Extensible**: Modular architecture for easy customization

## 📋 Prerequisites

- Python 3.8 or higher
- GPU (optional, for faster inference and fine-tuning)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd AI_RAG_Model
pip install -r ../requirement.txt
```

### 2. Basic Usage

#### Option A: Using the Command-Line Interface

```bash
# Ingest documents
python main.py --mode ingest --documents ./path/to/documents

# Start chat
python main.py --mode chat

# Use simple mode (no LLM, just retrieval)
python main.py --mode chat --simple
```

#### Option B: Using Python API

```python
from main import RAGChatApp

# Initialize
app = RAGChatApp()
app.setup(use_simple_rag=True)  # Set False to use full LLM

# Ingest documents
app.ingest_documents("./data/documents", is_directory=True)

# Query
response = app.query("What is machine learning?")
print(response)

# Start interactive chat
app.start_chat()
```

## 📚 Detailed Guide

### Step 1: Document Ingestion

The system can process various document formats:

```python
from document_processor import DocumentProcessor

processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)

# Single file
docs = processor.process_documents("document.pdf")

# Directory (recursive)
docs = processor.process_documents("./documents", is_directory=True)

# Add custom metadata
docs = processor.add_metadata(docs, {"category": "technical", "year": 2024})
```

**Supported Formats:**
- Text files (`.txt`)
- PDF documents (`.pdf`)
- Word documents (`.docx`, `.doc`)
- HTML files (`.html`, `.htm`)
- CSV files (`.csv`)

### Step 2: Vector Database Setup

Store and retrieve documents using embeddings:

```python
from vector_store import VectorStoreManager

# Initialize
vector_store = VectorStoreManager(
    collection_name="my_documents",
    persist_directory="./chroma_db",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Add documents
ids = vector_store.add_documents(documents)

# Search
results = vector_store.similarity_search("your query", k=4)

# Get statistics
stats = vector_store.get_collection_stats()
print(stats)
```

**Recommended Embedding Models:**
- `sentence-transformers/all-MiniLM-L6-v2` - Fast, good quality (default)
- `sentence-transformers/all-mpnet-base-v2` - Better quality, slower
- `BAAI/bge-small-en-v1.5` - Optimized for retrieval

### Step 3: RAG System

Two modes available:

#### Full RAG with LLM

```python
from rag_system import RAGSystem

rag = RAGSystem(
    vector_store_manager=vector_store,
    model_name="google/flan-t5-base",  # or other HuggingFace models
    temperature=0.7,
    max_new_tokens=512
)

response = rag.query("What are the main topics in the documents?")
print(response['answer'])
```

#### Simple RAG (Retrieval Only)

```python
from rag_system import SimpleRAGSystem

simple_rag = SimpleRAGSystem(vector_store)
response = simple_rag.query("What are the main topics?", k=4)

# Returns relevant documents with scores
for doc in response['documents']:
    print(f"Score: {doc['score']}")
    print(f"Content: {doc['content']}")
```

**Recommended LLM Models:**
- `google/flan-t5-base` - Balanced (default)
- `google/flan-t5-large` - Better quality
- `gpt2` - Fast, creative text
- `facebook/opt-350m` - Good alternative

### Step 4: Chat Agent

Interactive conversational interface:

```python
from chat_agent import ChatAgent

chat_agent = ChatAgent(
    rag_system=rag,
    max_history=10
)

# Interactive mode
chat_agent.multi_turn_chat()

# Or programmatic
response = chat_agent.chat("Tell me about AI")
print(response['answer'])

# Save conversation
chat_agent.save_conversation("conversation.json")
chat_agent.export_conversation_markdown("conversation.md")
```

**Chat Commands:**
- Type your question normally to chat
- `history` - View conversation history
- `clear` - Clear conversation history
- `stats` - Show system statistics
- `quit` or `q` - Exit chat

### Step 5: Updating Documents

Keep your knowledge base current:

```python
# Method 1: Add new documents
new_docs = processor.process_documents("new_document.pdf")
vector_store.add_documents(new_docs)

# Method 2: Update existing documents
updated_docs = processor.process_documents("updated_policy.pdf")
vector_store.update_documents(
    updated_docs,
    source_filter={"source": "policy.pdf"}  # Remove old version
)

# Method 3: Using main app
from main import RAGChatApp

app = RAGChatApp()
app.setup(use_simple_rag=True)
app.update_documents(
    "./new_documents",
    source_filter={"category": "policies"},
    is_directory=True
)
```

### Step 6: Fine-Tuning

Improve model performance on your domain:

```python
from fine_tuning import ModelFineTuner, DatasetCreator

# Create training data
questions = ["What is X?", "How does Y work?"]
answers = ["X is...", "Y works by..."]
training_data = DatasetCreator.create_qa_dataset(questions, answers)

# Fine-tune
fine_tuner = ModelFineTuner(base_model_name="gpt2")
fine_tuner.prepare_lora_model(lora_r=8, lora_alpha=32)

dataset = fine_tuner.prepare_training_data(training_data)
model_path = fine_tuner.fine_tune(
    dataset,
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4
)

print(f"Model saved to: {model_path}")

# Use fine-tuned model
fine_tuner.load_fine_tuned_model(model_path)
```

**Fine-Tuning Tips:**
- Use LoRA for efficient training (trains only ~1% of parameters)
- Start with 3-5 epochs, monitor for overfitting
- Batch size depends on your GPU memory (4-8 typical)
- Learning rate: 2e-4 to 5e-4 for LoRA

## 🔧 Configuration

Customize system behavior via `config.py`:

```python
from config import SystemConfig

config = SystemConfig()

# Document processing
config.document.chunk_size = 1000
config.document.chunk_overlap = 200

# Vector store
config.vector_store.collection_name = "my_collection"
config.vector_store.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# Model
config.model.model_name = "google/flan-t5-base"
config.model.temperature = 0.7
config.model.device = "cpu"  # or "cuda"

# RAG
config.rag.retrieval_k = 4
config.rag.return_sources = True

# Chat
config.chat.max_history = 10

# Fine-tuning
config.fine_tuning.lora_r = 8
config.fine_tuning.num_epochs = 3
```

## 📖 Examples

Run the example scripts:

```bash
# Run specific example
python examples.py 1  # Basic setup
python examples.py 2  # Multiple documents
python examples.py 3  # Updating documents
python examples.py 4  # Custom configuration
python examples.py 5  # Fine-tuning setup
python examples.py 6  # Chat history

# Run all examples
python examples.py all
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG Chat Agent                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐     ┌─────────────┐│
│  │   Document   │      │    Vector    │     │     RAG     ││
│  │  Processor   │─────▶│    Store     │────▶│   System    ││
│  │              │      │  (ChromaDB)  │     │             ││
│  └──────────────┘      └──────────────┘     └─────────────┘│
│         │                      │                     │       │
│         │                      │                     │       │
│         ▼                      ▼                     ▼       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Chat Agent Interface                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Fine-Tuning Module (Optional)              │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
AI_RAG_Model/
├── config.py              # Configuration management
├── document_processor.py  # Document loading and chunking
├── vector_store.py        # Vector database management
├── rag_system.py          # RAG implementation
├── chat_agent.py          # Conversational interface
├── fine_tuning.py         # Model fine-tuning
├── main.py                # Main application
├── examples.py            # Usage examples
└── README.md              # This file

Generated directories:
├── chroma_db/            # Vector database storage
├── data/                 # Document storage
├── logs/                 # Application logs
└── fine_tuned_models/    # Saved fine-tuned models
```

## 🎯 Use Cases

1. **Internal Knowledge Base**: Company documentation, policies, procedures
2. **Customer Support**: Product manuals, FAQs, troubleshooting guides
3. **Research Assistant**: Academic papers, research notes
4. **Code Documentation**: API docs, code comments, tutorials
5. **Legal/Compliance**: Contracts, regulations, compliance documents

## 🔍 Troubleshooting

### Issue: Out of Memory

**Solution:** Use Simple RAG mode or lighter models

```python
app.setup(use_simple_rag=True)
# Or use a smaller model
config.model.model_name = "gpt2"
```

### Issue: Slow Performance

**Solutions:**
- Use CPU-optimized embedding model
- Reduce chunk size
- Lower retrieval k value
- Use GPU if available

```python
config.vector_store.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
config.document.chunk_size = 500
config.rag.retrieval_k = 2
```

### Issue: Poor Quality Answers

**Solutions:**
- Increase retrieval k
- Use better embedding model
- Fine-tune the model on your domain
- Improve document preprocessing

```python
config.rag.retrieval_k = 6
config.vector_store.embedding_model = "sentence-transformers/all-mpnet-base-v2"
```

## 🛠️ Advanced Features

### Custom Prompt Templates

```python
custom_prompt = """Based on the following context, answer the question.
If unsure, say "I don't know."

Context: {context}

Question: {question}

Answer:"""

answer = rag_system.query_with_custom_prompt(
    "What is...?",
    custom_prompt,
    k=4
)
```

### Filtered Retrieval

```python
# Search within specific metadata
results = vector_store.similarity_search(
    "query",
    k=4,
    filter_dict={"category": "technical", "year": 2024}
)
```

### Batch Processing

```python
queries = ["Question 1?", "Question 2?", "Question 3?"]
responses = [app.query(q) for q in queries]
```

## 📊 Performance Tips

1. **Embedding Model Selection**: Balance between speed and quality
2. **Chunk Size**: 500-1000 tokens is usually optimal
3. **Chunk Overlap**: 10-20% of chunk size
4. **Retrieval K**: Start with 4, adjust based on results
5. **GPU Usage**: Significantly faster for LLM inference

## 🤝 Contributing

This is a modular system designed for extension. Key extension points:

- Add new document loaders in `document_processor.py`
- Implement alternative vector stores in `vector_store.py`
- Add new LLM backends in `rag_system.py`
- Extend chat features in `chat_agent.py`

## 📄 License

This project uses open-source packages. Please refer to individual package licenses.

## 🙏 Acknowledgments

Built with:
- LangChain - RAG framework
- ChromaDB - Vector database
- HuggingFace - Models and embeddings
- Sentence Transformers - Embeddings
- PEFT - Parameter-efficient fine-tuning

## 📞 Support

For issues or questions:
1. Check the examples in `examples.py`
2. Review the troubleshooting section
3. Check individual module documentation

---

**Happy Building! 🚀**
