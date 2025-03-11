import os
import logging
from pathlib import Path
from typing import Iterator
import threading
from queue import Queue

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from config import DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Update paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "backend" /"ai-model"/ "llama-2-7b-chat.ggmlv3.q8_0.bin"
DB_FAISS_PATH = Path(__file__).parent / "vectorstoredb_faiss"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def load_llm():
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
            
        logger.info(f"Loading model from: {MODEL_PATH}")
        llm = CTransformers(
            model=str(MODEL_PATH),
            model_type="llama",
            max_new_tokens=2048,
            temperature=0.3,
            context_length=4096,
            n_ctx=4096, # Explicitly set n_ctx
            gpu_layers=0  # Force CPU-only mode
        )
        logger.info("Model loaded successfully")
        return llm
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

from langchain_core.callbacks import BaseCallbackHandler, CallbackManager

class QueueCallback(BaseCallbackHandler):
    def __init__(self, queue: Queue):
        super().__init__()
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.queue.put({'type': 'token', 'token': token})

    def on_llm_end(self, response, **kwargs) -> None:
        self.queue.put(None)

def qa_bot(streaming=False, queue=None):
    try:
        logger.info("Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        if not DB_FAISS_PATH.exists():
            raise FileNotFoundError(f"Vector store not found at {DB_FAISS_PATH}")
        
        logger.info(f"Loading vector store from {DB_FAISS_PATH}")
        # db = FAISS.load_local(str(DB_FAISS_PATH), embeddings)
        db = FAISS.load_local(str(DB_FAISS_PATH), embeddings, allow_dangerous_deserialization=True)


        logger.info("Vector store loaded successfully")
        
        callbacks = []
        if streaming and queue:
            callbacks.append(QueueCallback(queue))
        
        llm = load_llm()
        qa_prompt = set_custom_prompt()
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=db.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': qa_prompt},
            callback_manager=CallbackManager(callbacks)
        )
        return qa
    except Exception as e:
        logger.error(f"Error in qa_bot: {str(e)}")
        raise

def stream_response(query: str) -> Iterator[dict]:
    queue = Queue()
    qa = qa_bot(streaming=True, queue=queue)
    
    def process_query():
        response = qa({'query': query})
        queue.put({'type': 'sources', 'data': response.get('source_documents', [])})
    
    thread = threading.Thread(target=process_query)
    thread.start()
    
    current_text = []
    while True:
        token = queue.get()
        if token is None:
            break
        elif isinstance(token, dict) and token.get('type') == 'sources':
            yield {
                'type': 'sources',
                'sources': [str(doc) for doc in token['data']]
            }
            # Removed 'break' to allow finishing streaming tokens
        else:
            current_text.append(token)
            yield {
                'type': 'token',
                'text': ''.join(current_text),
                'token': token
            }

def final_result(query: str, stream: bool = False):
    if stream:
        return stream_response(query)
    else:
        qa = qa_bot()
        response = qa({'query': query})
        return response
