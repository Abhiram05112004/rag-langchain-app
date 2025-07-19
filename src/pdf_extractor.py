from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PdfExtractor:
    """Handles extraction of text from PDF documents and splitting into chunks."""
    
    def __init__(self, file_path):
        """Initialize with the path to the PDF file.
        
        Args:
            file_path: Path to the PDF file
        """
        self.file_path = file_path
        self.loader = PyPDFLoader(file_path)

    def load_documents(self):
        """Load the PDF and convert to LangChain documents.
        
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
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(docs)