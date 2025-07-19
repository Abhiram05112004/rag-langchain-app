import os
import secrets
import logging

# Set USER_AGENT to avoid warnings during web scraping
os.environ.setdefault('USER_AGENT', 'RAG-LangChain-App/1.0 (Document Processing Bot)')

from flask import Flask, request, render_template, redirect, url_for, session, flash, get_flashed_messages, jsonify
from rag_engine import RagEngine

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Use environment variable or generate a random secret key
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(16))

logger.info("Creating RAG engine instance...")
# Use absolute path for FAISS index to avoid working directory issues
faiss_index_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "faiss_index")
logger.info(f"Using FAISS index path: {faiss_index_path}")

# Create RAG engine with simple chat history
rag_engine = RagEngine(
    db_path=faiss_index_path,
    enable_chat_history=True,
    max_history=10  # Keep last 10 exchanges for context
)

logger.info("RAG engine created successfully")

# Store sources in memory for demo (use DB for production)
sources = []

@app.before_request
def setup_session():
    """Initialize session data"""
    # Legacy chat_history for backward compatibility
    if 'chat_history' not in session:
        session['chat_history'] = []

def sync_sources_with_vector_store():
    """Rebuild the sources list from the vector store to reflect current state."""
    global sources
    
    try:
        # Get all unique sources from the vector store
        source_paths = rag_engine.vector_store.list_sources()
        logger.info(f"Found {len(source_paths)} sources in vector store")
        
        # Build sources as list of dicts with name and short
        sources = []
        for src in source_paths:
            # If it's a URL, use the first 50 chars as short; if file, use filename
            if src.startswith('http://') or src.startswith('https://'):
                short = src[:50] + ('...' if len(src) > 50 else '')
                name = src
            else:
                short = src
                name = os.path.basename(src)
            sources.append({'name': name, 'short': short})
        
        logger.info(f"Sources synced successfully: {len(sources)} sources ready for display")
        
    except Exception as e:
        logger.error(f"Error syncing sources: {e}")
        # Initialize empty sources to prevent errors
        sources = []

@app.route('/remove_source', methods=['POST'])
def remove_source():
    """Remove a source from the list and vector store"""
    name = request.form.get('name')
    short = request.form.get('short')
    
    if not name or not short:
        logger.warning("Attempted to remove source with missing parameters")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return {"success": False, "message": "Missing parameters"}, 400
        return redirect(url_for('home'))
    
    global sources
    sources = [s for s in sources if not (s['name'] == name and s['short'] == short)]
    
    # Remove from vector DB as well
    removed_count = 0
    if name.endswith('.pdf'):
        # Remove all chunks from this PDF file
        removed_count = rag_engine.vector_store.remove_by_source(short)
        logger.info(f"Removed PDF source: {name} ({removed_count} chunks)")
        
        # Also delete the actual PDF file from disk
        try:
            if os.path.exists(short):
                os.remove(short)
                logger.info(f"Deleted PDF file: {short}")
        except Exception as e:
            logger.error(f"Failed to delete PDF file {short}: {e}")
            
    else:
        # Remove all chunks from this URL
        removed_count = rag_engine.vector_store.remove_by_source(name)
        logger.info(f"Removed URL source: {name} ({removed_count} chunks)")
        
        # Debug: Check what sources remain after removal
        remaining_sources = rag_engine.vector_store.list_sources()
        logger.info(f"Remaining sources after removal: {remaining_sources}")
        
    # Sync sources after removal
    sync_sources_with_vector_store()
    
    # Handle AJAX requests differently
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return {"success": True}, 200
    
    # Regular form submission (fallback)
    return redirect(url_for('home'))

@app.route('/', methods=['GET', 'POST'])
def home():
    """Home page route that handles all primary app functionality"""
    answer = None
    error = None
    
    if request.method == 'POST':
        # Handle URL ingestion
        if 'url' in request.form and request.form['url']:
            url = request.form['url'].strip()
            if not url.startswith(('http://', 'https://')):
                error = "URL must start with http:// or https://"
            else:
                # Check if URL already exists in vector store
                existing_sources = rag_engine.vector_store.list_sources()
                if url in existing_sources:
                    flash(f"URL already exists in the knowledge base: {url}", "warning")
                    logger.warning(f"Attempted to add duplicate URL: {url}")
                    return redirect(url_for('home'))
                else:
                    try:
                        chunks = rag_engine.ingest_web(url)
                        sync_sources_with_vector_store()  # Sync after adding content
                        flash(f"Web content ingested! {len(chunks)} chunks added.", "success")
                        logger.info(f"Successfully ingested URL: {url}")
                        return redirect(url_for('home'))  # Redirect to prevent resubmission
                    except Exception as e:
                        error = f"Error ingesting URL: {str(e)}"
                        logger.error(f"Error ingesting URL {url}: {str(e)}", exc_info=True)
        
        # Handle PDF ingestion
        elif 'pdf' in request.files and request.files['pdf'].filename:
            pdf_file = request.files['pdf']
            if not pdf_file.filename.lower().endswith('.pdf'):
                error = "File must be a PDF"
            else:
                # Use absolute path for uploads directory
                uploads_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
                file_path = os.path.join(uploads_dir, pdf_file.filename)
                
                # Check if PDF already exists in vector store
                existing_sources = rag_engine.vector_store.list_sources()
                if file_path in existing_sources:
                    flash(f"PDF already exists in the knowledge base: {pdf_file.filename}", "warning")
                    logger.warning(f"Attempted to add duplicate PDF: {pdf_file.filename}")
                    return redirect(url_for('home'))
                else:
                    try:
                        os.makedirs(uploads_dir, exist_ok=True)
                        pdf_file.save(file_path)
                        
                        chunks = rag_engine.ingest_pdf(file_path)
                        sync_sources_with_vector_store()  # Sync after adding content
                        flash(f"PDF ingested! {len(chunks)} chunks added.", "success")
                        logger.info(f"Successfully ingested PDF: {pdf_file.filename}")
                        return redirect(url_for('home'))  # Redirect to prevent resubmission
                    except Exception as e:
                        error = f"Error ingesting PDF: {str(e)}"
                        logger.error(f"Error ingesting PDF {pdf_file.filename}: {str(e)}", exc_info=True)
        
        # Handle questions
        elif 'question' in request.form and request.form['question'].strip():
            question = request.form['question'].strip()
            try:
                logger.info(f"Processing question: {question}")
                answer = rag_engine.query(question)
                logger.info(f"Successfully processed question, got answer of length: {len(answer)}")
                
                # Maintain backward compatibility with session chat_history
                if 'chat_history' not in session:
                    session['chat_history'] = []
                session['chat_history'].append({'question': question, 'answer': answer})
                session.modified = True
                
            except Exception as e:
                error_msg = f"Error processing question: {str(e)}"
                logger.error(error_msg, exc_info=True)
                logger.error(f"Question that caused error: {question}")
                logger.error(f"Error type: {type(e).__name__}")
                error = error_msg
                
    # Always refresh sources from vector store to ensure current state
    sync_sources_with_vector_store()
        
    # Get chat history from RAG engine
    try:
        chat_history = rag_engine.get_chat_history()
        # Convert to format expected by template
        formatted_history = []
        for exchange in chat_history:
            formatted_history.append({
                'question': exchange['question'], 
                'answer': exchange['answer']
            })
    except:
        # Fallback to session-based chat history
        formatted_history = session.get('chat_history', [])
    
    # Get any flash messages to display
    messages = get_flashed_messages(with_categories=True)
    
    return render_template(
        'index.html', 
        answer=answer, 
        error=error, 
        sources=sources, 
        chat_history=formatted_history,
        messages=messages
    )

@app.route('/api/question', methods=['POST'])
def api_question():
    """AJAX API endpoint for submitting questions"""
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return {"success": False, "message": "Question is required"}, 400
        
    try:
        logger.info(f"API processing question: {question}")
        answer = rag_engine.query(question)
        logger.info(f"API successfully processed question, got answer of length: {len(answer)}")
        
        # Maintain backward compatibility with session chat history
        if 'chat_history' not in session:
            session['chat_history'] = []
        session['chat_history'].append({'question': question, 'answer': answer})
        session.modified = True
        
        return {
            "success": True,
            "answer": answer,
            "question": question
        }
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        logger.error(error_msg, exc_info=True)
        logger.error(f"API Question that caused error: {question}")
        logger.error(f"API Error type: {type(e).__name__}")
        return {"success": False, "message": error_msg}, 500

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    """AJAX endpoint to clear chat history for the session."""
    # Clear RAG engine chat history
    rag_engine.clear_chat_history()
    
    # Also clear session-based history for backward compatibility
    session['chat_history'] = []
    session.modified = True
    
    return {"success": True}

# Simple Chat History API Endpoints

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Get chat history"""
    try:
        history = rag_engine.get_chat_history()
        return {"success": True, "history": history}
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return {"success": False, "message": str(e)}, 500

@app.route('/api/chat/stats', methods=['GET'])
def get_chat_stats():
    """Get chat statistics"""
    try:
        stats = rag_engine.get_chat_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        logger.error(f"Error getting chat stats: {e}")
        return {"success": False, "message": str(e)}, 500

if __name__ == '__main__':
    logger.info("Starting RAG LangChain Web App with Simple Chat History")
    # Use debug=False to prevent reloading on every request/refresh
    # Set host='0.0.0.0' to allow external connections if needed
    app.run(debug=False, host='127.0.0.1', port=5000)
