from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class WebExtractor:
    """Handles extraction of text from web URLs and splitting into chunks."""
    
    def __init__(self, url):
        """Initialize with the URL to extract content from.
        
        Args:
            url: Web URL to extract content from
        """
        self.url = url
        self.loader = WebBaseLoader(url)

    def load_documents(self):
        """Load the web page and convert to LangChain documents.
        
        Returns:
            List of Document objects
        """
        return self.loader.load()

    def split_documents(self, docs, chunk_size=1089, chunk_overlap=108):
        """Split documents into chunks for better processing.
        
        Args:
            docs: List of Document objects
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of split Document objects
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(docs)