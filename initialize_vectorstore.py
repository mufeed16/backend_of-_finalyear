import os
from pathlib import Path
import logging
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_vector_store():
    try:
        # Setup paths
        base_dir = Path(__file__).parent
        data_dir = base_dir / "uploads"
        db_faiss_path = base_dir / "vectorstoredb_faiss"

        # Create directories if they don't exist
        data_dir.mkdir(exist_ok=True)
        db_faiss_path.parent.mkdir(exist_ok=True)

        logger.info(f"Loading documents from {data_dir}")
        loader = DirectoryLoader(str(data_dir), glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        if not documents:
            logger.warning("No documents found to process")
            return False

        logger.info(f"Loaded {len(documents)} documents")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Add the model name
            model_kwargs={'device': 'cpu'}
        )

        logger.info("Creating vector store...")
        db = FAISS.from_documents(documents, embeddings)
        
        logger.info(f"Saving vector store to {db_faiss_path}")
        db.save_local(str(db_faiss_path))
        
        logger.info("Vector store created successfully")
        return True

    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise

if __name__ == "__main__":
    create_vector_store()
