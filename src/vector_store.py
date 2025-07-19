from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import logging
import os

logger = logging.getLogger(__name__)

# Global model cache to prevent reloading
_model_cache = {}

class CustomEmbeddings:
    """Custom embeddings wrapper using sentence-transformers with caching."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Use cached model if available
        if model_name not in _model_cache:
            logger.info(f"Loading embedding model: {model_name}")
            _model_cache[model_name] = SentenceTransformer(model_name)
        else:
            logger.info(f"Using cached embedding model: {model_name}")
        
        self.model = _model_cache[model_name]
        self.model_name = model_name
        
    def embed_documents(self, texts):
        """Embed a list of documents."""
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    
    def embed_query(self, text):
        """Embed a single query."""
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()
    
    def __call__(self, text):
        """Make the object callable for FAISS compatibility."""
        return self.embed_query(text)

class VectorStore:
    """Optimized FAISS-based vector store for better performance."""
    
    # Class-level cache for embeddings to prevent reinitialization
    _embeddings_instance = None
    
    def __init__(self, db_path: str = "faiss_index"):
        # Use class-level cached embeddings instance
        if VectorStore._embeddings_instance is None:
            logger.info("Creating new embeddings instance for VectorStore")
            VectorStore._embeddings_instance = CustomEmbeddings("all-MiniLM-L6-v2")
        else:
            logger.info("Reusing cached embeddings instance for VectorStore")
            
        self.embeddings = VectorStore._embeddings_instance
        
        # Adjust db_path based on current working directory
        # If we're in src/, look for faiss_index in parent directory
        if os.path.basename(os.getcwd()) == 'src':
            self.db_path = os.path.join("..", db_path)
        else:
            self.db_path = db_path
            
        self.vector_store = None
        self._load_or_create()

    def _load_or_create(self):
        """Load existing FAISS index or create new one."""
        try:
            # Check for the FAISS index files directly in the db_path directory
            # FAISS.load_local expects the directory path, and internally adds "index" to find index.faiss and index.pkl
            faiss_file = os.path.join(self.db_path, "index.faiss")
            pkl_file = os.path.join(self.db_path, "index.pkl")
            
            if os.path.exists(faiss_file) and os.path.exists(pkl_file):
                # Pass just the db_path - FAISS will look for index.faiss and index.pkl in that directory
                self.vector_store = FAISS.load_local(
                    self.db_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded existing FAISS index from {self.db_path}")
                return
            
            # No existing index found
            self.vector_store = None
            logger.info("FAISS index will be created when first documents are added")
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {str(e)}")
            self.vector_store = None

    def add_documents(self, documents, metadatas=None):
        """Add documents to the vector store with optimized batching."""
        if not documents:
            return
        
        try:
            if self.vector_store is None:
                # Create new FAISS index from first batch of documents
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                logger.info(f"Created new FAISS index with {len(documents)} documents")
            else:
                # Add to existing index
                new_vector_store = FAISS.from_documents(documents, self.embeddings)
                self.vector_store.merge_from(new_vector_store)
                logger.debug(f"Added {len(documents)} documents to existing FAISS index")  # Changed to debug
            
            self.save()
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")

    def search(self, query, k=5):
        """Optimized similarity search with score threshold."""
        if self.vector_store is None:
            return []
        
        try:
            # Use similarity_search_with_score for better ranking
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k*2)  # Get more, then filter
            
            # Log scores for debugging
            logger.info(f"Search scores: {[(score, doc.page_content[:50]) for doc, score in docs_with_scores[:3]]}")
            
            # Filter by relevance score (lower is better for FAISS)
            # Use a very lenient threshold for testing - accept almost anything
            filtered_docs = [doc for doc, score in docs_with_scores if score < 10.0]
            
            # If no docs pass the filter, return the best ones anyway
            if not filtered_docs and docs_with_scores:
                logger.info("No docs passed score filter, returning best matches anyway")
                filtered_docs = [doc for doc, score in docs_with_scores[:k]]
            
            logger.info(f"Returning {len(filtered_docs)} documents after filtering")
            return filtered_docs[:k]  # Return top k after filtering
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []

    def search_with_scores(self, query, k=5):
        """Search with similarity scores for debugging/tuning."""
        if self.vector_store is None:
            return []
        
        try:
            return self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error(f"Error in search with scores: {str(e)}")
            return []

    def get_retriever(self, search_kwargs=None):
        """Get optimized retriever with better search parameters."""
        if self.vector_store is None:
            return None
        
        if search_kwargs is None:
            search_kwargs = {
                "k": 5,
                "fetch_k": 10,  # Fetch more candidates for better ranking
            }
        
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

    def is_empty(self):
        """Check if the vector store is empty."""
        return self.vector_store is None

    def list_sources(self):
        """Return list of unique sources from metadata without triggering embedding."""
        if self.vector_store is None:
            return []
        
        try:
            # Access the FAISS index directly to get all document metadata
            # This avoids triggering the embedding model
            if hasattr(self.vector_store, 'docstore') and hasattr(self.vector_store.docstore, '_dict'):
                # Get all documents from the docstore
                all_docs = list(self.vector_store.docstore._dict.values())
                sources = {doc.metadata.get("source") for doc in all_docs if doc.metadata.get("source")}
                return list(sources)
            else:
                # Fallback: return empty list to avoid embedding calls
                logger.warning("Cannot access docstore directly, returning empty sources list")
                return []
        except Exception as e:
            logger.error(f"Error listing sources: {str(e)}")
            return []

    def save(self):
        """Save FAISS index to disk."""
        if self.vector_store is not None:
            try:
                # Ensure directory exists
                os.makedirs(self.db_path, exist_ok=True)
                
                # Save to the db_path directory - FAISS will create index.faiss and index.pkl
                self.vector_store.save_local(self.db_path)
                logger.info(f"Saved FAISS index to {self.db_path}")
            except Exception as e:
                logger.error(f"Error saving FAISS index: {str(e)}")

    def clear_all(self):
        """Clear all documents and remove index files."""
        self.vector_store = None
        
        # Remove FAISS files from the db_path directory
        try:
            faiss_file = os.path.join(self.db_path, "index.faiss")
            pkl_file = os.path.join(self.db_path, "index.pkl")
            
            for file_path in [faiss_file, pkl_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed {file_path}")
                        
        except Exception as e:
            logger.error(f"Error clearing FAISS files: {str(e)}")

    def remove_by_source(self, source_path):
        """Remove documents by source (requires rebuilding index)."""
        if self.vector_store is None:
            return 0
        
        try:
            # Get all documents using a generic search
            all_docs = self.vector_store.similarity_search("document", k=10000)
            
            # Filter out documents from the specified source
            remaining_docs = [doc for doc in all_docs if doc.metadata.get("source") != source_path]
            
            if len(remaining_docs) == len(all_docs):
                return 0  # No documents removed
            
            # Rebuild index with remaining documents
            if remaining_docs:
                self.vector_store = FAISS.from_documents(remaining_docs, self.embeddings)
                self.save()
            else:
                self.clear_all()
            
            removed_count = len(all_docs) - len(remaining_docs)
            logger.info(f"Removed {removed_count} documents from source: {source_path}")
            return removed_count
            
        except Exception as e:
            logger.error(f"Error removing documents by source: {str(e)}")
            return 0
