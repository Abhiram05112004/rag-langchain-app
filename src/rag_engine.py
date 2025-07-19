from pdf_extractor import PdfExtractor
from web_extractor import WebExtractor
from groq_llm import GroqLLM
from langchain_chat_history import SimpleLangChainHistory
from vector_store import VectorStore
import logging
import re

logger = logging.getLogger(__name__)

class RagEngine:
    """
    Retrieval-Augmented Generation (RAG) engine for orchestrating document ingestion, retrieval, and LLM-based answering.
    Enhanced with simple conversational context awareness.
    """
    def __init__(self, db_path="faiss_index", enable_chat_history=True, max_history=10):
        self.vector_store = VectorStore(db_path)
        self.llm = GroqLLM()
        
        # Simple chat history for conversational context
        self.enable_chat_history = enable_chat_history
        self.chat_history = SimpleLangChainHistory(max_history=max_history) if enable_chat_history else None
        
        if enable_chat_history:
            logger.info(f"LangChain chat history enabled with max {max_history} exchanges")
        
        # Enhanced prompt template with conversation context
        self.prompt_template = """
You are a knowledgeable assistant with access to document context and conversation history. Answer questions based on the provided context and maintain conversational flow.

{conversation_context}

Current Question: {question}

Document Context:
{context}

CONVERSATION GUIDELINES:
- If this seems like a follow-up question, reference the previous conversation naturally
- Use phrases like "As we discussed..." or "Building on your previous question..." when relevant
- Connect the current question to the ongoing conversation flow
- If no document context matches, acknowledge this clearly but maintain conversational tone

FORMATTING INSTRUCTIONS:
- Give a direct, conversational answer that flows naturally from any previous discussion
- Be precise and to the point like ChatGPT
- Use simple bullet points only if specifically requested or when listing items
- Provide the most relevant information that directly answers the question
- Use formal conversational, natural language

FORMATTING RULES:
- You MUST output HTML format, NOT markdown
- For bold text: Use <strong>word</strong> - NEVER use **word** or *word*
- For italic text: Use <em>word</em> - NEVER use *word* or _word_
- FORBIDDEN: Do not use asterisks (*) or underscores (_) for formatting
- Example correct format: "As we discussed earlier, Shiva is depicted with <strong>three eyes</strong>"

Answer:"""

    def ingest_web(self, url):
        """
        Ingests and indexes content from a web URL.
        Returns a list of document chunks added to the vector store.
        """
        extractor = WebExtractor(url)
        docs = extractor.load_documents()
        # Set the source metadata for web documents
        for doc in docs:
            doc.metadata.setdefault('source', url)
        chunks = extractor.split_documents(docs)
        self.vector_store.add_documents(chunks)
        return chunks

    def ingest_pdf(self, file_path):
        """
        Ingests and indexes content from a PDF file.
        Returns a list of document chunks added to the vector store.
        """
        extractor = PdfExtractor(file_path)
        docs = extractor.load_documents()
        for doc in docs:
            doc.metadata.setdefault('source', file_path)
        chunks = extractor.split_documents(docs)
        self.vector_store.add_documents(chunks)
        return chunks

    def query(self, question, k=5):
        """
        Answers a question using RAG with simple conversational context awareness.
        """
        # Handle simple greetings first, regardless of document status
        if self._is_simple_greeting(question):
            greeting_response = "Hello! 👋 I'm your document assistant. I can help you find information from your uploaded documents. What would you like to know?"
            return greeting_response
        
        # Check if documents are available for actual questions
        if self.vector_store.is_empty():
            no_docs_response = "I don't have access to any documents yet. Please upload some PDFs or add web content first, and I'll be happy to help answer your questions!"
            return no_docs_response
        
        try:
            # Search for relevant documents
            relevant_docs = self.vector_store.search(question, k=k)
            logger.info(f"Search for '{question}' returned {len(relevant_docs)} documents")
            
            if not relevant_docs:
                no_context_response = "I couldn't find specific information related to your question in the uploaded documents. Could you try rephrasing your question or asking about a different topic?"
                return no_context_response
            
            # Log document content for debugging
            for i, doc in enumerate(relevant_docs[:3]):
                logger.info(f"Document {i+1} preview: {doc.page_content[:100]}...")
            
            # Prepare context from top relevant documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])
            
            # Get conversation context if chat history is enabled
            conversation_context = ""
            is_follow_up = False
            
            if self.enable_chat_history and self.chat_history:
                conversation_context = self.chat_history.get_conversation_context(include_last_n=3)
                is_follow_up = self.chat_history.is_follow_up_question(question)
                
                if is_follow_up:
                    recent_questions = self.chat_history.get_recent_questions(limit=2)
                    if recent_questions:
                        conversation_context += f"\nNote: This appears to be a follow-up question to: {recent_questions[-1]}"
            
            # Format the prompt with conversation context
            formatted_prompt = self.prompt_template.format(
                conversation_context=conversation_context,
                context=context, 
                question=question
            )
            
            # Generate answer using GroqLLM
            answer = self.llm.generate(formatted_prompt)
            
            # Clean up the answer and ensure HTML formatting
            answer = answer.strip()
            
            # Convert any markdown that slipped through to HTML (backup safety)
            answer = self._convert_markdown_to_html(answer)
            
            # Add sources in a clean format
            sources = self._extract_sources(relevant_docs[:3])
            if sources:
                answer += f"\n---<em>Based on: {', '.join(sources)}</em>"
            
            # Add exchange to chat history
            if self.enable_chat_history and self.chat_history:
                self.chat_history.add_exchange(question, answer, sources)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in query processing: {str(e)}", exc_info=True)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Question was: {question}")
            
            # Provide more specific error messages based on the error type
            if "API" in str(e) or "requests" in str(e).lower():
                error_response = "I'm having trouble connecting to the AI service. Please check your internet connection and API key, then try again."
            elif "embedding" in str(e).lower() or "model" in str(e).lower():
                error_response = "I'm having trouble processing your question with the embedding model. Please try again in a moment."
            elif "vector" in str(e).lower() or "faiss" in str(e).lower():
                error_response = "I'm having trouble searching through the documents. Please try rephrasing your question."
            else:
                error_response = f"I encountered an issue while processing your question: {str(e)}. Please try again or rephrase your question."
            
            return error_response

    @staticmethod
    def _extract_sources(docs_or_context):
        """
        Extracts unique source names from a list of Document objects or context.
        """
        sources = set()
        for doc in docs_or_context:
            if hasattr(doc, "metadata") and "source" in doc.metadata:
                source = doc.metadata["source"]
                if '/' in source or '\\' in source:
                    source = source.split('/')[-1].split('\\')[-1]
                sources.add(source)
        return list(sources)

    def _is_simple_greeting(self, question):
        """Check if the question is a simple greeting or non-specific query."""
        question_lower = question.lower().strip()
        greetings = {
            'hi', 'hello','hlo', 'hey', 'hiya', 'good morning', 'good afternoon', 
            'good evening', 'how are you', 'whats up', "what's up", 'sup',
            'greetings', 'howdy', 'yo'
        }
        return question_lower in greetings or len(question.strip()) < 3

    def _convert_markdown_to_html(self, text):
        """
        Convert markdown formatting to HTML as a backup safety measure.
        This ensures proper HTML formatting even if the model outputs markdown.
        """
        # Convert **text** to <strong>text</strong>
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        
        # Convert *text* to <em>text</em> (simpler pattern to avoid regex issues)
        text = re.sub(r'(?<!\*)\*([^*\n]+?)\*(?!\*)', r'<em>\1</em>', text)
        
        # Convert _text_ to <em>text</em>
        text = re.sub(r'\b_([^_\n]+?)_\b', r'<em>\1</em>', text)
        
        return text

    # Simple Chat History Methods
    
    def get_chat_history(self):
        """Get chat history for display"""
        if not self.enable_chat_history or not self.chat_history:
            return []
        return self.chat_history.get_history()
    
    def clear_chat_history(self):
        """Clear chat history"""
        if self.enable_chat_history and self.chat_history:
            self.chat_history.clear_history()
            logger.info("Chat history cleared")
    
    def get_chat_stats(self):
        """Get simple chat statistics"""
        if not self.enable_chat_history or not self.chat_history:
            return {"total_exchanges": 0, "has_context": False}
        return self.chat_history.get_stats()