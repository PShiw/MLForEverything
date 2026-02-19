"""
RAG System Module
Implements Retrieval Augmented Generation pipeline
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Make complex LangChain components optional
try:
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain_community.llms import HuggingFacePipeline
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    FULL_RAG_AVAILABLE = True
except ImportError as e:
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning(f"Full RAG dependencies not available: {e}")
    FULL_RAG_AVAILABLE = False
    RetrievalQA = None
    PromptTemplate = None
    HuggingFacePipeline = None

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain.docstore.document import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG (Retrieval Augmented Generation) System"""
    
    def __init__(
        self,
        vector_store_manager,
        model_name: str = "google/flan-t5-base",
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        device: str = "cpu"
    ):
        """
        Initialize RAG system
        
        Args:
            vector_store_manager: VectorStoreManager instance
            model_name: HuggingFace model name
            temperature: Generation temperature
            max_new_tokens: Maximum tokens to generate
            device: Device to run model on ('cpu' or 'cuda')
        """
        if not FULL_RAG_AVAILABLE:
            raise ImportError(
                "Full RAG dependencies not available. "
                "Please use SimpleRAGSystem instead, or install missing dependencies."
            )
        
        self.vector_store = vector_store_manager
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        # Import torch here since it's part of FULL_RAG_AVAILABLE check
        import torch
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Initialize LLM
        self.llm = None
        self.qa_chain = None
        self._initialize_llm()
        self._build_qa_chain()
    
    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            logger.info(f"Loading model: {self.model_name} on {self.device}")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            
            if self.device == "cpu":
                model = model.to(self.device)
            
            # Create text generation pipeline
            text_gen_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.15,
            )
            
            # Wrap in LangChain
            self.llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.info("Falling back to simplified model initialization")
            # Fallback to a simpler initialization
            raise
    
    def _build_qa_chain(self):
        """Build the QA chain"""
        try:
            # Custom prompt template
            prompt_template = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Always provide detailed and well-formatted answers based on the context.

Context:
{context}

Question: {question}

Answer: """

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Get retriever from vector store
            retriever = self.vector_store.get_retriever(
                search_kwargs={'k': 4}
            )
            
            # Build QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": PROMPT,
                }
            )
            
            logger.info("QA chain built successfully")
            
        except Exception as e:
            logger.error(f"Error building QA chain: {str(e)}")
            raise
    
    def query(
        self,
        question: str,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User's question
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and optionally source documents
        """
        try:
            logger.info(f"Processing query: {question[:50]}...")
            
            result = self.qa_chain({"query": question})
            
            response = {
                "question": question,
                "answer": result["result"],
                "timestamp": datetime.now().isoformat()
            }
            
            if return_sources and "source_documents" in result:
                sources = []
                for doc in result["source_documents"]:
                    sources.append({
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    })
                response["sources"] = sources
            
            logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "question": question,
                "answer": f"Error processing query: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "error": True
            }
    
    def query_with_custom_prompt(
        self,
        question: str,
        custom_prompt: str,
        k: int = 4
    ) -> str:
        """
        Query with a custom prompt template
        
        Args:
            question: User's question
            custom_prompt: Custom prompt template (must include {context} and {question})
            k: Number of documents to retrieve
            
        Returns:
            Generated answer
        """
        try:
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(question, k=k)
            
            # Concatenate context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Format prompt
            formatted_prompt = custom_prompt.format(
                context=context,
                question=question
            )
            
            # Generate answer
            answer = self.llm(formatted_prompt)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error with custom prompt query: {str(e)}")
            return f"Error: {str(e)}"
    
    def retrieve_documents(
        self,
        query: str,
        k: int = 4,
        with_scores: bool = False
    ) -> List[Document] | List[tuple[Document, float]]:
        """
        Retrieve relevant documents without generation
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            with_scores: Whether to return relevance scores
            
        Returns:
            List of documents or (document, score) tuples
        """
        if with_scores:
            return self.vector_store.similarity_search_with_score(query, k=k)
        else:
            return self.vector_store.similarity_search(query, k=k)
    
    def format_response(self, response: Dict[str, Any]) -> str:
        """
        Format response for display
        
        Args:
            response: Response dictionary from query
            
        Returns:
            Formatted string
        """
        formatted = f"""
{'='*80}
QUESTION: {response['question']}
{'='*80}

ANSWER:
{response['answer']}

"""
        if 'sources' in response:
            formatted += f"\nSOURCES ({len(response['sources'])} documents):\n"
            for i, source in enumerate(response['sources'], 1):
                formatted += f"\n{i}. {source['content']}"
                if 'source' in source['metadata']:
                    formatted += f"\n   Source: {source['metadata']['source']}"
                formatted += "\n"
        
        formatted += f"\nTimestamp: {response['timestamp']}\n"
        formatted += "="*80
        
        return formatted
    
    def update_retriever_k(self, k: int):
        """
        Update the number of documents to retrieve
        
        Args:
            k: New number of documents to retrieve
        """
        retriever = self.vector_store.get_retriever(search_kwargs={'k': k})
        self.qa_chain.retriever = retriever
        logger.info(f"Updated retriever to fetch {k} documents")


class SimpleRAGSystem:
    """Simplified RAG system that doesn't require a full LLM"""
    
    def __init__(self, vector_store_manager):
        """
        Initialize simple RAG system
        
        Args:
            vector_store_manager: VectorStoreManager instance
        """
        self.vector_store = vector_store_manager
    
    def query(self, question: str, k: int = 4) -> Dict[str, Any]:
        """
        Simple query that returns relevant documents
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with relevant documents
        """
        docs = self.vector_store.similarity_search_with_score(question, k=k)
        
        return {
            "question": question,
            "documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in docs
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    def format_response(self, response: Dict[str, Any]) -> str:
        """Format simple response"""
        formatted = f"""
{'='*80}
QUESTION: {response['question']}
{'='*80}

RELEVANT DOCUMENTS:
"""
        for i, doc in enumerate(response['documents'], 1):
            formatted += f"\n{i}. [Score: {doc['score']:.4f}]\n"
            formatted += f"{doc['content'][:300]}...\n"
            if 'source' in doc['metadata']:
                formatted += f"Source: {doc['metadata']['source']}\n"
        
        formatted += f"\nTimestamp: {response['timestamp']}\n"
        formatted += "="*80
        
        return formatted


if __name__ == "__main__":
    print("RAG System module ready!")
