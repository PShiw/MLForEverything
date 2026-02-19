"""
Configuration Module
Central configuration for the RAG system
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DocumentConfig:
    """Configuration for document processing"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    supported_formats: List[str] = field(
        default_factory=lambda: ['.txt', '.pdf', '.docx', '.doc', '.html', '.htm', '.csv']
    )


@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    collection_name: str = "rag_documents"
    persist_directory: str = "./chroma_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Alternative embedding models:
    # - "sentence-transformers/all-mpnet-base-v2" (better quality, slower)
    # - "sentence-transformers/paraphrase-MiniLM-L6-v2" (fast, good quality)
    # - "BAAI/bge-small-en-v1.5" (good for retrieval)


@dataclass
class ModelConfig:
    """Configuration for language models"""
    # Default model - lightweight and fast
    model_name: str = "google/flan-t5-base"
    
    # Alternative models:
    # - "google/flan-t5-large" (better quality, larger)
    # - "gpt2" (good for general text)
    # - "facebook/opt-350m" (fast, reasonable quality)
    # - "EleutherAI/gpt-neo-125M" (creative text)
    
    temperature: float = 0.7
    max_new_tokens: int = 512
    device: str = "cpu"  # or "cuda" for GPU


@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    retrieval_k: int = 4  # Number of documents to retrieve
    return_sources: bool = True
    use_simple_rag: bool = False  # Set to True to use SimpleRAGSystem


@dataclass
class ChatConfig:
    """Configuration for chat agent"""
    max_history: int = 10
    system_prompt: Optional[str] = None


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning"""
    output_dir: str = "./fine_tuned_models"
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 4


@dataclass
class SystemConfig:
    """Main system configuration"""
    document: DocumentConfig = field(default_factory=DocumentConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    fine_tuning: FineTuningConfig = field(default_factory=FineTuningConfig)
    
    # Paths
    data_directory: str = "./data"
    logs_directory: str = "./logs"
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.data_directory, exist_ok=True)
        os.makedirs(self.logs_directory, exist_ok=True)
        os.makedirs(self.vector_store.persist_directory, exist_ok=True)
        os.makedirs(self.fine_tuning.output_dir, exist_ok=True)


# Default configuration instance
default_config = SystemConfig()


if __name__ == "__main__":
    config = SystemConfig()
    print("Configuration initialized with following settings:")
    print(f"Vector Store: {config.vector_store.collection_name}")
    print(f"Model: {config.model.model_name}")
    print(f"Embedding Model: {config.vector_store.embedding_model}")
    print(f"Device: {config.model.device}")
