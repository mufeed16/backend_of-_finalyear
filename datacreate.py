from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_vector_db(filepath):
    try:
        logger.info(f"Processing PDF file: {filepath}")

        # Check if the file exists
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return False
        
        # Create the vectorstoredb_faiss directory if it doesn't exist
        db_path = Path("vectorstoredb_faiss")
        db_path.mkdir(exist_ok=True)
        
        # Load and process the single PDF file
        try:
            loader = PyPDFLoader(filepath)
            documents = loader.load()
            logger.info(f"Number of pages loaded: {len(documents)}")
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            return False
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create and save vector store
        db = FAISS.from_documents(documents, embeddings)
        db.save_local("vectorstoredb_faiss")
        logger.info("Vector database created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error creating vector database: {e}")
        return False

if __name__ == "__main__":
    create_vector_db("path/to/your/pdf/file.pdf")
