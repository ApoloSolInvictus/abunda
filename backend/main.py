import logging
import sys
import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient

# --- CONFIGURATION ---
# Qdrant must be running in Docker: docker run -p 6333:6333 qdrant/qdrant
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "abunda_knowledge"

# Ollama Configuration
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3"

# Logs
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("ABUNDA_BRAIN")

class AbundaBrain:
    def __init__(self):
        logger.info(f"⚡ Initializing ABUNDA Brain with {MODEL_NAME}...")
        self.active = False
        self.index = None
        
        try:
            # 1. Setup LLM (Llama 3 Local)
            Settings.llm = Ollama(
                model=MODEL_NAME, 
                base_url=OLLAMA_URL, 
                request_timeout=360.0 
            )
            
            # 2. Setup Embeddings
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            )
            
            # 3. Connection Check - Qdrant
            try:
                self.client = QdrantClient(url=QDRANT_URL)
                # Test connection by listing collections
                self.client.get_collections()
                logger.info("✅ Qdrant Connection: ACTIVE")
            except Exception as e:
                logger.error(f"❌ Qdrant Connection FAILED: {e}")
                logger.error("👉 Please ensure Docker is running: 'docker run -p 6333:6333 qdrant/qdrant'")
                raise e

            # 4. Storage Context
            self.vector_store = QdrantVectorStore(
                client=self.client, 
                collection_name=COLLECTION_NAME
            )
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # 5. Load Index
            try:
                self.index = VectorStoreIndex.from_vector_store(
                    self.vector_store,
                )
                logger.info("✅ Vector Index Loaded.")
            except Exception:
                logger.info("⚠️ Index empty. Waiting for documents.")
                
            self.active = True
            logger.info("🚀 BRAIN ONLINE.")

        except Exception as e:
            logger.error(f"❌ CRITICAL BRAIN FAILURE: {e}")
            self.active = False

    def ingest_document(self, file_path: str):
        if not self.active: return False
        logger.info(f"📥 Ingesting: {file_path}")
        
        try:
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
            
            if self.index is None:
                self.index = VectorStoreIndex.from_documents(
                    documents, 
                    storage_context=self.storage_context
                )
            else:
                for doc in documents:
                    self.index.insert(doc)
            
            logger.info(f"✅ Indexed {len(documents)} fragments.")
            return True
        except Exception as e:
            logger.error(f"❌ Ingestion Error: {e}")
            return False

    def query(self, question: str):
        if not self.active: 
            return "System Error: Brain is offline. Check server logs."
        
        logger.info(f"🧠 Thinking: {question}")
        
        if self.index is None:
            return "Knowledge Base is empty. Please upload a document first."
            
        try:
            query_engine = self.index.as_query_engine(
                similarity_top_k=3, 
            )
            response = query_engine.query(question)
            return str(response)
        except Exception as e:
            logger.error(f"Query Error: {e}")
            return "I encountered an error processing your request. Please check if Ollama is running."

brain = AbundaBrain()
