"""
Vector Store Module
Manages vector database for document embeddings using ChromaDB
"""

import os
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

import chromadb
from chromadb.config import Settings

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage vector database for RAG system"""
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize vector store manager
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: HuggingFace model for embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embeddings
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize vector store
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize or load existing vector store"""
        try:
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
            logger.info(f"Vector store initialized with collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> List[str]:
        """
        Add documents to vector store
        
        Args:
            documents: List of documents to add
            batch_size: Batch size for adding documents
            
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents to add")
            return []
        
        try:
            # Add timestamp to metadata
            for doc in documents:
                doc.metadata['added_at'] = datetime.now().isoformat()
            
            # Add documents in batches
            ids = []
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_ids = self.vectorstore.add_documents(batch)
                ids.extend(batch_ids)
                logger.info(f"Added batch {i//batch_size + 1}: {len(batch)} documents")
            
            logger.info(f"Successfully added {len(documents)} documents to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def update_documents(
        self,
        documents: List[Document],
        source_filter: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update documents in vector store
        If source_filter is provided, removes old documents matching filter first
        
        Args:
            documents: New documents to add
            source_filter: Filter to identify documents to replace
        """
        try:
            # Remove old documents if filter provided
            if source_filter:
                logger.info(f"Removing old documents with filter: {source_filter}")
                self.delete_documents(source_filter)
            
            # Add new documents
            self.add_documents(documents)
            logger.info("Documents updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating documents: {str(e)}")
            raise
    
    def delete_documents(self, filter_dict: Dict[str, Any]) -> None:
        """
        Delete documents matching filter criteria
        
        Args:
            filter_dict: Filter criteria for deletion
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            
            # Get IDs matching filter
            results = collection.get(where=filter_dict)
            
            if results['ids']:
                collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} documents")
            else:
                logger.info("No documents found matching filter")
                
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of similar documents
        """
        try:
            if filter_dict:
                results = self.vectorstore.similarity_search(
                    query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search(query, k=k)
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents with relevance scores
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of (document, score) tuples
        """
        try:
            if filter_dict:
                results = self.vectorstore.similarity_search_with_score(
                    query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search with score: {str(e)}")
            raise
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Get a retriever for the vector store
        
        Args:
            search_kwargs: Arguments for search (e.g., {'k': 4})
            
        Returns:
            Retriever object
        """
        if search_kwargs is None:
            search_kwargs = {'k': 4}
        
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'persist_directory': self.persist_directory,
                'embedding_model': self.embedding_model_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    def reset_collection(self) -> None:
        """Delete and recreate the collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            self._initialize_vectorstore()
            logger.info("Collection reset successfully")
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    vector_store = VectorStoreManager(
        collection_name="test_collection",
        persist_directory="./test_chroma_db"
    )
    
    # Print stats
    stats = vector_store.get_collection_stats()
    print(f"Collection stats: {stats}")
