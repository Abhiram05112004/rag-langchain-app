# RAG LangChain App with Groq API & Conversational Intelligence

A sophisticated RAG (Retrieval-Augmented Generation) web application built with Flask, LangChain, FAISS vector store, and Groq API with Qwen models. The application features conversational AI capabilities, allowing users to ingest PDFs and web URLs, manage sources, and engage in natural conversations with their documents through a modern ChatGPT-style interface.

> **Note**: This application has been updated to use Groq API with Qwen models and includes advanced conversational context awareness for natural dialogue flow.

## ✨ Key Features

### **Document Management**
- **PDF & Web URL Ingestion**: Upload PDF files or provide web URLs to extract and index content
- **Smart Source Management**: Add and remove sources with automatic vector store updates
- **Duplicate Prevention**: Automatic detection and prevention of duplicate document uploads
- **Source Attribution**: Answers include references to the source documents used

### **Conversational AI**
- **Context-Aware Conversations**: Understands follow-up questions and maintains conversation context
- **Natural Dialogue Flow**: Handles pronouns ("he", "she", "it") and contextual references
- **Smart Question Enhancement**: Automatically enhances questions with relevant conversation context
- **Memory Management**: Maintains optimal chat history (last 10 exchanges) for performance

### **Modern Interface**
- **ChatGPT-Style UI**: Dark-themed, professional chat interface with typing indicators
- **Real-time AJAX**: Smooth chat experience without page reloads
- **Responsive Design**: Built with Bootstrap for all device sizes
- **HTML Formatting**: Clean, formatted responses with proper styling
- **Session Persistence**: Chat history maintained during browser sessions
- **Clear Chat Feature**: Red clear button in top-right corner to start fresh conversations

### **Advanced RAG**
- **Groq API Integration**: Uses high-performance Qwen models for superior response quality
- **FAISS Vector Store**: Fast similarity search with sentence transformers
- **Optimized Retrieval**: Enhanced document search using conversational context
- **Error Recovery**: Intelligent error handling with user-friendly messages

## 🏗️ Project Structure

```
rag-langchain-app/
├── .env                   # API keys (not committed)
├── .env.example          # Template for environment variables
├── .gitignore           # Git ignore rules
├── README.md            # Project documentation
├── requirements.txt     # Clean dependencies
├── faiss_index/        # Vector database storage (created at runtime)
├── uploads/            # PDF file storage (created at runtime)
└── src/
    ├── app.py                    # Flask web application with session management
    ├── rag_engine.py            # Main RAG orchestration with conversational AI
    ├── vector_store.py          # FAISS vector store management with caching
    ├── groq_llm.py              # Groq API integration with Qwen models
    ├── langchain_chat_history.py # LangChain-based chat memory management
    ├── pdf_extractor.py         # PDF document processing
    ├── web_extractor.py         # Web URL content extraction
    └── templates/
        └── index.html           # Modern ChatGPT-style web interface
```

## 🚀 Conversational AI Capabilities

### **Smart Context Understanding**
- **Follow-up Questions**: "Who is Rama?" → "Where was he born?" (understands "he" refers to Rama)
- **Topic Tracking**: Maintains conversation topics across multiple exchanges
- **Pronoun Resolution**: Handles "he", "she", "it", "this", "that", "they" references
- **Contextual Enhancement**: Automatically improves question clarity using conversation history

### **Natural Dialogue Patterns**
- **Multi-turn Conversations**: Seamless dialogue flow like ChatGPT
- **Memory Management**: Optimized to remember relevant context without performance impact
- **Question Enhancement**: Enriches queries with conversation context for better retrieval
- **Topic Continuity**: Maintains discussion threads across multiple questions

### **Example Conversation Flow**
```
User: "Tell me about Hindu deities"
Bot: [Comprehensive answer about Hindu deities]

User: "What about Shiva specifically?"
Bot: [Detailed information about Shiva, understanding the context]

User: "What are his main symbols?"
Bot: [Lists Shiva's symbols, knowing "his" refers to Shiva]

User: "Tell me more about the third eye"
Bot: [Explains Shiva's third eye, maintaining conversation context]
```

## 📋 Prerequisites

- Python 3.8+
- Groq API key (get one from [Groq Console](https://console.groq.com/))
- Internet connection for embeddings and API calls

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/rag-langchain-app.git
   cd rag-langchain-app
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On Windows
   python -m venv venv
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   .\venv\Scripts\Activate.ps1

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your Groq API key:**
   ```bash
   # Copy the environment template
   # On Windows:
   copy .env.example .env
   # On macOS/Linux:
   cp .env.example .env
   
   # Edit .env and add your actual Groq API key:
   # GROQ_API_KEY=your_actual_groq_api_key_here
   ```

## 🚀 Running the Application

1. **Start the Flask web server:**
   ```bash
   cd src
   python app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://127.0.0.1:5000/
   ```

## 💡 Using the Application

### **Document Management**
1. **Add Sources**:
   - **Web URL**: Enter a web URL in the URL field and click "Add Source"
   - **PDF Upload**: Choose a PDF file and click "Add Source"
   - **Duplicate Prevention**: System automatically prevents duplicate uploads

2. **Remove Sources**:
   - Click the trash icon (🗑️) next to any source to remove it
   - Sources are removed from both the UI and vector database

### **Conversational Chat**
1. **Start a Conversation**:
   - Type your question in the chat box and press Enter or click "Ask"
   - Get comprehensive answers with source attribution

2. **Follow-up Questions**:
   - Ask follow-up questions naturally: "What about...", "Tell me more", "How about..."
   - Use pronouns: "Where was he born?", "What are its main features?"
   - The AI maintains context across your entire conversation

3. **Chat Management**:
   - **Clear Chat**: Click the red "Clear Chat" button in the top-right corner to start a new conversation
   - **Session Persistence**: Chat history is maintained during your browser session
   - **Memory Optimization**: System automatically manages memory for optimal performance

## 🛡️ Technical Features

### **Performance Optimizations**
- **Global Model Caching**: Embeddings models cached to prevent reloading
- **Session Management**: Efficient chat history with automatic memory cleanup
- **AJAX Interface**: Real-time responses without page reloads
- **Absolute Path Handling**: Robust file system operations across different environments

### **Error Handling & Recovery**
- **Graceful Degradation**: User-friendly error messages for different failure types
- **API Resilience**: Intelligent retry logic and connection handling
- **Input Validation**: Comprehensive validation for uploads and URLs
- **Logging**: Detailed logging for debugging and monitoring

### **Security & Reliability**
- **Environment Variables**: Secure API key management
- **Input Sanitization**: Safe handling of user inputs and uploads
- **Session Security**: Secure session management with Flask
- **File Validation**: PDF and URL validation before processing

## 🔧 Technologies Used

- **Backend**: Flask (Python web framework)
- **AI/ML**: 
  - LangChain (RAG framework and document processing)
  - Groq API (Cloud-based LLM with Qwen models)
  - Sentence Transformers (Fast embeddings - all-MiniLM-L6-v2)
  - FAISS (Vector database for similarity search)
- **Frontend**: 
  - Bootstrap 5 (Responsive UI framework)
  - FontAwesome (Icons)
  - Vanilla JavaScript (AJAX and DOM manipulation)
- **Document Processing**: 
  - PyPDF (PDF processing)
  - BeautifulSoup4 (Web scraping)
- **Storage**: FAISS (Local vector database)

## 🎯 Use Cases

- **Research Assistance**: Upload research papers and ask complex questions
- **Document Analysis**: Analyze business documents, reports, and manuals
- **Knowledge Base**: Create a conversational interface for your document collections
- **Educational Tool**: Interactive learning with textbooks and academic materials
- **Content Discovery**: Explore large document sets through natural conversation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- **LangChain** for the powerful RAG framework
- **Groq** for providing fast cloud-based LLM API
- **Sentence Transformers** for efficient embeddings
- **FAISS** for high-performance vector similarity search
- **Bootstrap** for the modern UI framework