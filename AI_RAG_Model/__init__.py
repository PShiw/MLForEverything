"""
RAG Chat Agent System
A complete Retrieval-Augmented Generation system for document-based Q&A
"""

__version__ = "1.0.0"
__author__ = "MLForEverything Project"

# Core modules
from .document_processor import DocumentProcessor
from .vector_store import VectorStoreManager
from .rag_system import RAGSystem, SimpleRAGSystem
from .chat_agent import ChatAgent, SimpleChatAgent
from .fine_tuning import ModelFineTuner, DatasetCreator
from .config import SystemConfig
from .main import RAGChatApp

__all__ = [
    'DocumentProcessor',
    'VectorStoreManager',
    'RAGSystem',
    'SimpleRAGSystem',
    'ChatAgent',
    'SimpleChatAgent',
    'ModelFineTuner',
    'DatasetCreator',
    'SystemConfig',
    'RAGChatApp',
]
