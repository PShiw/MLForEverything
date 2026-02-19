"""
Main Application Module
Complete RAG Chat Agent System
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from config import SystemConfig
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_system import RAGSystem, SimpleRAGSystem
from chat_agent import ChatAgent, SimpleChatAgent
from fine_tuning import ModelFineTuner, DatasetCreator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGChatApp:
    """
    Complete RAG Chat Application
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize the RAG Chat Application
        
        Args:
            config: System configuration (uses default if None)
        """
        self.config = config or SystemConfig()
        
        # Initialize components
        self.document_processor = None
        self.vector_store = None
        self.rag_system = None
        self.chat_agent = None
        self.fine_tuner = None
        
        logger.info("RAG Chat Application initialized")
    
    def setup(self, use_simple_rag: bool = False):
        """
        Setup all components
        
        Args:
            use_simple_rag: Whether to use SimpleRAGSystem (no LLM required)
        """
        logger.info("Setting up RAG Chat Application...")
        
        # Initialize document processor
        self.document_processor = DocumentProcessor(
            chunk_size=self.config.document.chunk_size,
            chunk_overlap=self.config.document.chunk_overlap
        )
        logger.info("✓ Document processor initialized")
        
        # Initialize vector store
        self.vector_store = VectorStoreManager(
            collection_name=self.config.vector_store.collection_name,
            persist_directory=self.config.vector_store.persist_directory,
            embedding_model=self.config.vector_store.embedding_model
        )
        logger.info("✓ Vector store initialized")
        
        # Initialize RAG system
        if use_simple_rag:
            self.rag_system = SimpleRAGSystem(self.vector_store)
            logger.info("✓ Simple RAG system initialized (no LLM)")
            
            # Initialize simple chat agent
            self.chat_agent = SimpleChatAgent(
                self.rag_system,
                max_history=self.config.chat.max_history
            )
        else:
            try:
                self.rag_system = RAGSystem(
                    vector_store_manager=self.vector_store,
                    model_name=self.config.model.model_name,
                    temperature=self.config.model.temperature,
                    max_new_tokens=self.config.model.max_new_tokens,
                    device=self.config.model.device
                )
                logger.info("✓ RAG system with LLM initialized")
                
                # Initialize chat agent
                self.chat_agent = ChatAgent(
                    self.rag_system,
                    system_prompt=self.config.chat.system_prompt,
                    max_history=self.config.chat.max_history
                )
            except Exception as e:
                logger.warning(f"Failed to initialize full RAG system: {e}")
                logger.info("Falling back to Simple RAG system")
                self.rag_system = SimpleRAGSystem(self.vector_store)
                self.chat_agent = SimpleChatAgent(
                    self.rag_system,
                    max_history=self.config.chat.max_history
                )
        
        logger.info("✓ Chat agent initialized")
        logger.info("Setup complete!")
    
    def ingest_documents(
        self,
        source_path: str,
        is_directory: bool = False,
        metadata: Optional[dict] = None
    ):
        """
        Ingest documents into the system
        
        Args:
            source_path: Path to document or directory
            is_directory: Whether source is a directory
            metadata: Optional metadata to add to documents
        """
        logger.info(f"Ingesting documents from: {source_path}")
        
        # Process documents
        documents = self.document_processor.process_documents(
            source_path,
            is_directory=is_directory
        )
        
        if not documents:
            logger.warning("No documents were processed")
            return
        
        # Add metadata if provided
        if metadata:
            documents = self.document_processor.add_metadata(documents, metadata)
        
        # Add to vector store
        ids = self.vector_store.add_documents(documents)
        
        logger.info(f"Successfully ingested {len(documents)} document chunks")
        return ids
    
    def update_documents(
        self,
        source_path: str,
        source_filter: Optional[dict] = None,
        is_directory: bool = False
    ):
        """
        Update documents in the system
        
        Args:
            source_path: Path to new documents
            source_filter: Filter to identify old documents to replace
            is_directory: Whether source is a directory
        """
        logger.info(f"Updating documents from: {source_path}")
        
        # Process new documents
        documents = self.document_processor.process_documents(
            source_path,
            is_directory=is_directory
        )
        
        if not documents:
            logger.warning("No documents were processed")
            return
        
        # Update vector store
        self.vector_store.update_documents(documents, source_filter)
        
        logger.info(f"Successfully updated with {len(documents)} document chunks")
    
    def start_chat(self):
        """Start interactive chat session"""
        if self.chat_agent is None:
            logger.error("Chat agent not initialized. Call setup() first.")
            return
        
        self.chat_agent.multi_turn_chat()
    
    def query(self, question: str, k: int = 4) -> dict:
        """
        Single query to the system
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            Response dictionary
        """
        if self.chat_agent is None:
            logger.error("Chat agent not initialized. Call setup() first.")
            return {}
        
        if isinstance(self.chat_agent, SimpleChatAgent):
            return self.chat_agent.chat(question, k=k)
        else:
            return self.chat_agent.chat(question, k_documents=k)
    
    def get_stats(self) -> dict:
        """Get system statistics"""
        stats = {}
        
        if self.vector_store:
            stats.update(self.vector_store.get_collection_stats())
        
        if self.chat_agent:
            stats['conversation_turns'] = len(self.chat_agent.conversation_history)
        
        return stats
    
    def setup_fine_tuning(self, base_model: Optional[str] = None):
        """
        Setup fine-tuning module
        
        Args:
            base_model: Base model name (uses config default if None)
        """
        model_name = base_model or self.config.model.model_name
        
        self.fine_tuner = ModelFineTuner(
            base_model_name=model_name,
            output_dir=self.config.fine_tuning.output_dir,
            device=self.config.model.device
        )
        
        logger.info("Fine-tuning module initialized")
    
    def fine_tune_model(self, training_data: list, **kwargs):
        """
        Fine-tune a model
        
        Args:
            training_data: List of training examples
            **kwargs: Additional training arguments
        """
        if self.fine_tuner is None:
            self.setup_fine_tuning()
        
        # Prepare model
        self.fine_tuner.prepare_lora_model(
            lora_r=self.config.fine_tuning.lora_r,
            lora_alpha=self.config.fine_tuning.lora_alpha,
            lora_dropout=self.config.fine_tuning.lora_dropout
        )
        
        # Prepare dataset
        dataset = self.fine_tuner.prepare_training_data(training_data)
        
        # Train
        model_path = self.fine_tuner.fine_tune(
            dataset,
            num_epochs=kwargs.get('num_epochs', self.config.fine_tuning.num_epochs),
            batch_size=kwargs.get('batch_size', self.config.fine_tuning.batch_size),
            learning_rate=kwargs.get('learning_rate', self.config.fine_tuning.learning_rate),
            gradient_accumulation_steps=kwargs.get(
                'gradient_accumulation_steps',
                self.config.fine_tuning.gradient_accumulation_steps
            )
        )
        
        logger.info(f"Model fine-tuned and saved to: {model_path}")
        return model_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="RAG Chat Agent System")
    parser.add_argument(
        '--mode',
        choices=['chat', 'ingest', 'update', 'stats'],
        default='chat',
        help='Operation mode'
    )
    parser.add_argument(
        '--documents',
        type=str,
        help='Path to documents or directory'
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Use simple RAG system without LLM'
    )
    parser.add_argument(
        '--collection',
        type=str,
        default='rag_documents',
        help='Vector store collection name'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = SystemConfig()
    config.vector_store.collection_name = args.collection
    
    # Initialize app
    app = RAGChatApp(config)
    
    if args.mode == 'ingest':
        if not args.documents:
            print("Error: --documents required for ingest mode")
            return
        
        app.setup(use_simple_rag=True)  # Don't need LLM for ingestion
        
        doc_path = Path(args.documents)
        app.ingest_documents(
            str(doc_path),
            is_directory=doc_path.is_dir()
        )
        
        print(f"\n✓ Documents ingested successfully!")
        print(f"Stats: {app.get_stats()}")
    
    elif args.mode == 'update':
        if not args.documents:
            print("Error: --documents required for update mode")
            return
        
        app.setup(use_simple_rag=True)
        
        doc_path = Path(args.documents)
        app.update_documents(
            str(doc_path),
            is_directory=doc_path.is_dir()
        )
        
        print(f"\n✓ Documents updated successfully!")
        print(f"Stats: {app.get_stats()}")
    
    elif args.mode == 'stats':
        app.setup(use_simple_rag=True)
        stats = app.get_stats()
        
        print("\n" + "="*80)
        print("SYSTEM STATISTICS")
        print("="*80)
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("="*80)
    
    elif args.mode == 'chat':
        app.setup(use_simple_rag=args.simple)
        app.start_chat()


if __name__ == "__main__":
    main()
