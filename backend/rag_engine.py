import loggingimport loggingimport logging
import sys
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

# --- CONFIGURATION & LOGGING ---
# Set logging to display in the terminal
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("ABUNDA_BRAIN")

# Network Configuration (Explicit IPs for stability)
# Ensure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant
QDRANT_URL = "http://127.0.0.1:6333" 
COLLECTION_NAME = "abunda_knowledge"

# Ensure Ollama is running: ollama serve
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL_NAME = "llama3"

class AbundaBrain:
    def __init__(self):
        logger.info(f"⚡ Initializing ABUNDA Brain with {MODEL_NAME}...")
        self.active = False
        self.index = None
        
        try:
            # 1. Setup LLM (Llama 3 Local via Ollama)
            Settings.llm = Ollama(
                model=MODEL_NAME, 
                base_url=OLLAMA_URL, 
                request_timeout=360.0 # Extended timeout for local CPU processing
            )
            
            # 2. Setup Embeddings (Local & Fast Model)
            # Using BAAI/bge-small-en-v1.5 standard for efficiency
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            )
            
            # 3. Connect to Vector Memory (Qdrant)
            try:
                self.client = QdrantClient(url=QDRANT_URL)
                # Simple connection test
                self.client.get_collections()
                logger.info("✅ Connection to Qdrant: SUCCESS")
            except Exception as e:
                logger.error(f"❌ Failed to connect to Qdrant at {QDRANT_URL}. Ensure Docker container is running.")
                # We return here to allow the API to start in 'Offline Mode' instead of crashing
                return

            # 4. Storage Context Setup
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
                logger.info("✅ Vector Memory Loaded.")
            except Exception:
                logger.info("⚠️ Memory is empty. System ready to learn (Upload a document).")
                self.index = None
                
            self.active = True
            logger.info("🚀 BRAIN ONLINE.")

        except Exception as e:
            logger.error(f"❌ Critical Brain Failure: {e}")
            self.active = False

    def ingest_document(self, file_path: str):
        """Reads a document, chunks it, and saves vectors to Qdrant."""
        if not self.active: 
            logger.error("Attempted ingestion with inactive brain.")
            return False
            
        logger.info(f"📥 Learning document: {file_path}")
        
        try:
            # LlamaIndex automatically detects file type (PDF, TXT, DOCX)
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
            
            if self.index is None:
                # Create new index if none exists
                self.index = VectorStoreIndex.from_documents(
                    documents, 
                    storage_context=self.storage_context
                )
            else:
                # Insert into existing index
                for doc in documents:
                    self.index.insert(doc)
            
            logger.info(f"✅ Document processed ({len(documents)} fragments).")
            return True
        except Exception as e:
            logger.error(f"❌ Ingestion Error: {e}")
            return False

    def query(self, question: str):
        """Queries the vector database using Llama 3."""
        if not self.active: 
            return "Error: Brain is disconnected. Please check Docker (Qdrant) and Ollama."
        
        logger.info(f"🧠 Analyzing: {question}")
        
        if self.index is None:
            return "I have no knowledge stored yet. Please upload a document to the Knowledge Base first."
            
        try:
            # Configure chat engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=3, # Retrieve top 3 most relevant context chunks
            )
            response = query_engine.query(question)
            return str(response)
        except Exception as e:
            logger.error(f"Query Error: {e}")
            return f"An error occurred while generating the response: {str(e)}"

# Global Instance
brain = AbundaBrain()
