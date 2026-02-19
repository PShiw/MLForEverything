"""
Document Processor Module
Handles ingestion and processing of various document formats
"""

import os
from typing import List, Dict, Any
from pathlib import Path
import logging

# Document loaders
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    CSVLoader,
)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process and chunk documents for RAG system"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a single document based on file extension
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of Document objects
        """
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif file_extension in ['.html', '.htm']:
                loader = UnstructuredHTMLLoader(file_path)
            elif file_extension == '.csv':
                loader = CSVLoader(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return []
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} document(s) from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return []
    
    def load_directory(self, directory_path: str, recursive: bool = True) -> List[Document]:
        """
        Load all supported documents from a directory
        
        Args:
            directory_path: Path to directory
            recursive: Whether to search subdirectories
            
        Returns:
            List of all loaded documents
        """
        documents = []
        supported_extensions = {'.txt', '.pdf', '.docx', '.doc', '.html', '.htm', '.csv'}
        
        path = Path(directory_path)
        
        if recursive:
            file_iterator = path.rglob('*')
        else:
            file_iterator = path.glob('*')
        
        for file_path in file_iterator:
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                docs = self.load_document(str(file_path))
                documents.extend(docs)
        
        logger.info(f"Loaded total of {len(documents)} document(s) from {directory_path}")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        chunked_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunked_docs)} chunks")
        return chunked_docs
    
    def process_documents(
        self,
        source: str,
        is_directory: bool = False,
        recursive: bool = True
    ) -> List[Document]:
        """
        Complete pipeline: load and chunk documents
        
        Args:
            source: File path or directory path
            is_directory: Whether source is a directory
            recursive: If directory, search recursively
            
        Returns:
            Processed and chunked documents
        """
        if is_directory:
            documents = self.load_directory(source, recursive)
        else:
            documents = self.load_document(source)
        
        if not documents:
            logger.warning("No documents were loaded")
            return []
        
        chunked_documents = self.chunk_documents(documents)
        return chunked_documents
    
    def add_metadata(self, documents: List[Document], metadata: Dict[str, Any]) -> List[Document]:
        """
        Add custom metadata to documents
        
        Args:
            documents: List of documents
            metadata: Dictionary of metadata to add
            
        Returns:
            Documents with updated metadata
        """
        for doc in documents:
            doc.metadata.update(metadata)
        return documents


if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    
    # Process a single file
    # docs = processor.process_documents("path/to/document.pdf")
    
    # Process a directory
    # docs = processor.process_documents("path/to/documents", is_directory=True)
    
    print("Document processor initialized successfully!")
