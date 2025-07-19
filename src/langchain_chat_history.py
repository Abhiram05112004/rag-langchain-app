"""
Simple LangChain Chat History - Lightweight & Efficient
Minimal LangChain memory implementation for conversational context
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

logger = logging.getLogger(__name__)

class SimpleLangChainHistory:
    """
    Lightweight LangChain chat history for conversational RAG
    Simple, efficient, and minimal - just what you need
    """
    
    def __init__(self, max_history: int = 10):
        """Initialize with LangChain memory components"""
        self.max_history = max_history
        
        # Use LangChain's in-memory chat history
        self.chat_memory = InMemoryChatMessageHistory()
        
        # Window memory keeps only recent exchanges
        self.memory = ConversationBufferWindowMemory(
            chat_memory=self.chat_memory,
            k=max_history,
            return_messages=True,
            memory_key="history"
        )
        
        self.sources_tracking = {}  # Track sources per exchange
        logger.info(f"LangChain chat history initialized (max: {max_history})")
    
    def add_exchange(self, question: str, answer: str, sources: List[str] = None):
        """Add Q&A exchange - LangChain handles the memory management"""
        # Store in LangChain memory
        self.memory.save_context(
            {"input": question},
            {"output": answer}
        )
        
        # Track sources separately (simple approach)
        exchange_id = len(self.chat_memory.messages) // 2
        if sources:
            self.sources_tracking[exchange_id] = sources
        
        logger.debug(f"Added exchange {exchange_id}")
    
    def get_conversation_context(self, include_last_n: int = 3) -> str:
        """Get conversation context - LangChain formats it nicely"""
        # Load memory variables
        memory_vars = self.memory.load_memory_variables({})
        messages = memory_vars.get("history", [])
        
        if not messages:
            return ""
        
        # Take recent exchanges
        recent = messages[-(include_last_n * 2):] if include_last_n else messages
        
        if len(recent) < 2:
            return ""
        
        context = ["Previous conversation:"]
        
        # Process in pairs (Human, AI)
        for i in range(0, len(recent), 2):
            if i + 1 < len(recent):
                q = recent[i].content
                a = recent[i + 1].content
                
                # Keep answers short for context
                a_short = a[:150] + "..." if len(a) > 150 else a
                context.append(f"Q: {q}")
                context.append(f"A: {a_short}")
        
        return "\n".join(context) + "\n---"
    
    def get_recent_questions(self, limit: int = 3) -> List[str]:
        """Get recent questions"""
        memory_vars = self.memory.load_memory_variables({})
        messages = memory_vars.get("history", [])
        
        questions = [msg.content for msg in messages if isinstance(msg, HumanMessage)]
        return questions[-limit:] if limit else questions
    
    def is_follow_up_question(self, question: str) -> bool:
        """Simple follow-up detection with LangChain context"""
        if not self.chat_memory.messages:
            return False
        
        # Quick heuristics
        q_lower = question.lower()
        follow_ups = ["it", "that", "this", "they", "them", "tell me more", "what about"]
        
        return any(indicator in q_lower for indicator in follow_ups) or len(question.split()) <= 5
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get history as dict list for UI"""
        memory_vars = self.memory.load_memory_variables({})
        messages = memory_vars.get("history", [])
        
        history = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                exchange_id = i // 2
                history.append({
                    "question": messages[i].content,
                    "answer": messages[i + 1].content,
                    "timestamp": datetime.now().isoformat(),
                    "sources": self.sources_tracking.get(exchange_id, [])
                })
        
        return history
    
    def clear_history(self):
        """Clear everything"""
        self.memory.clear()
        self.sources_tracking.clear()
        logger.info("Chat history cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Simple stats"""
        total = len(self.chat_memory.messages) // 2
        return {
            "total_exchanges": total,
            "max_history": self.max_history,
            "has_context": total > 0,
            "memory_type": "LangChain Window"
        }
