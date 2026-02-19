"""
Chat Agent Module
Provides conversational interface for the RAG system
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatAgent:
    """
    Conversational agent with RAG capabilities and chat history
    """
    
    def __init__(
        self,
        rag_system,
        system_prompt: Optional[str] = None,
        max_history: int = 10
    ):
        """
        Initialize chat agent
        
        Args:
            rag_system: RAGSystem or SimpleRAGSystem instance
            system_prompt: Custom system prompt for the agent
            max_history: Maximum number of conversation turns to keep
        """
        self.rag_system = rag_system
        self.max_history = max_history
        self.conversation_history: List[Dict[str, Any]] = []
        
        self.system_prompt = system_prompt or """
You are a helpful AI assistant that answers questions based on the provided documents.
Always be accurate, concise, and cite your sources when possible.
If you don't know something based on the available documents, say so honestly.
"""
        
        logger.info("Chat agent initialized")
    
    def chat(
        self,
        user_message: str,
        include_sources: bool = True,
        k_documents: int = 4
    ) -> Dict[str, Any]:
        """
        Process a user message and generate a response
        
        Args:
            user_message: User's input message
            include_sources: Whether to include source documents
            k_documents: Number of documents to retrieve
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Processing user message: {user_message[:50]}...")
        
        # Get response from RAG system
        response = self.rag_system.query(
            user_message,
            return_sources=include_sources
        )
        
        # Add to conversation history
        conversation_turn = {
            "user": user_message,
            "assistant": response.get("answer", ""),
            "timestamp": datetime.now().isoformat(),
            "sources": response.get("sources", []) if include_sources else []
        }
        
        self.conversation_history.append(conversation_turn)
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        return response
    
    def multi_turn_chat(self):
        """
        Start an interactive multi-turn conversation
        """
        print("\n" + "="*80)
        print("Chat Agent Started - Type 'quit' to exit, 'history' to view conversation")
        print("="*80 + "\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if user_input.lower() == 'history':
                    self.display_history()
                    continue
                
                if user_input.lower() == 'clear':
                    self.clear_history()
                    print("Conversation history cleared.")
                    continue
                
                if user_input.lower() == 'stats':
                    self.display_stats()
                    continue
                
                # Process the message
                response = self.chat(user_input)
                
                # Display response
                print(f"\nAssistant: {response['answer']}")
                
                if 'sources' in response and response['sources']:
                    print(f"\n[Used {len(response['sources'])} source document(s)]")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in chat: {str(e)}")
                print(f"\nError: {str(e)}")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history
        
        Returns:
            List of conversation turns
        """
        return self.conversation_history
    
    def display_history(self):
        """Display conversation history"""
        if not self.conversation_history:
            print("\nNo conversation history yet.")
            return
        
        print("\n" + "="*80)
        print("CONVERSATION HISTORY")
        print("="*80)
        
        for i, turn in enumerate(self.conversation_history, 1):
            print(f"\n--- Turn {i} ---")
            print(f"User: {turn['user']}")
            print(f"Assistant: {turn['assistant']}")
            print(f"Time: {turn['timestamp']}")
        
        print("\n" + "="*80)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def display_stats(self):
        """Display statistics about the conversation"""
        stats = {
            "total_turns": len(self.conversation_history),
            "max_history": self.max_history,
        }
        
        # Get vector store stats
        if hasattr(self.rag_system, 'vector_store'):
            vs_stats = self.rag_system.vector_store.get_collection_stats()
            stats.update(vs_stats)
        
        print("\n" + "="*80)
        print("SYSTEM STATISTICS")
        print("="*80)
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("="*80)
    
    def save_conversation(self, filepath: str):
        """
        Save conversation history to a file
        
        Args:
            filepath: Path to save the conversation
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
            logger.info(f"Conversation saved to {filepath}")
            print(f"Conversation saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")
            print(f"Error saving conversation: {str(e)}")
    
    def load_conversation(self, filepath: str):
        """
        Load conversation history from a file
        
        Args:
            filepath: Path to load the conversation from
        """
        try:
            with open(filepath, 'r') as f:
                self.conversation_history = json.load(f)
            logger.info(f"Conversation loaded from {filepath}")
            print(f"Conversation loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading conversation: {str(e)}")
            print(f"Error loading conversation: {str(e)}")
    
    def export_conversation_markdown(self, filepath: str):
        """
        Export conversation to markdown format
        
        Args:
            filepath: Path to save the markdown file
        """
        try:
            with open(filepath, 'w') as f:
                f.write("# Conversation History\n\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                f.write("---\n\n")
                
                for i, turn in enumerate(self.conversation_history, 1):
                    f.write(f"## Turn {i}\n\n")
                    f.write(f"**User:** {turn['user']}\n\n")
                    f.write(f"**Assistant:** {turn['assistant']}\n\n")
                    
                    if turn.get('sources'):
                        f.write("**Sources:**\n\n")
                        for j, source in enumerate(turn['sources'], 1):
                            f.write(f"{j}. {source['content']}\n")
                        f.write("\n")
                    
                    f.write(f"*Time: {turn['timestamp']}*\n\n")
                    f.write("---\n\n")
            
            logger.info(f"Conversation exported to {filepath}")
            print(f"Conversation exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting conversation: {str(e)}")
            print(f"Error exporting conversation: {str(e)}")


class SimpleChatAgent:
    """
    Simplified chat agent that works with SimpleRAGSystem
    """
    
    def __init__(self, simple_rag_system, max_history: int = 10):
        """
        Initialize simple chat agent
        
        Args:
            simple_rag_system: SimpleRAGSystem instance
            max_history: Maximum conversation turns to keep
        """
        self.rag_system = simple_rag_system
        self.max_history = max_history
        self.conversation_history = []
    
    def chat(self, user_message: str, k: int = 4) -> Dict[str, Any]:
        """
        Simple chat that retrieves relevant documents
        
        Args:
            user_message: User's question
            k: Number of documents to retrieve
            
        Returns:
            Response dictionary
        """
        response = self.rag_system.query(user_message, k=k)
        
        self.conversation_history.append({
            "user": user_message,
            "documents": response['documents'],
            "timestamp": response['timestamp']
        })
        
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        return response
    
    def multi_turn_chat(self):
        """Interactive chat session"""
        print("\n" + "="*80)
        print("Simple Chat Agent - Type 'quit' to exit")
        print("="*80 + "\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                response = self.chat(user_input)
                
                print("\nRelevant Documents:")
                for i, doc in enumerate(response['documents'], 1):
                    print(f"\n{i}. [Score: {doc['score']:.4f}]")
                    print(f"{doc['content'][:200]}...")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")


if __name__ == "__main__":
    print("Chat Agent module ready!")
