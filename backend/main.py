import logging
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

# Configuración de Logs
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("ABUNDA_BRAIN")

# --- AJUSTE DE RED ---
# Usamos 127.0.0.1 para forzar IPv4 y evitar ambigüedad con 'localhost'
QDRANT_URL = "http://127.0.0.1:6333" 
COLLECTION_NAME = "abunda_knowledge"

# Ollama en puerto 11434
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL_NAME = "llama3"

class AbundaBrain:
    def __init__(self):
        logger.info(f"⚡ Inicializando ABUNDA Brain con {MODEL_NAME} en {OLLAMA_URL}...")
        self.active = False
        self.index = None
        
        try:
            # 1. Configurar LLM
            Settings.llm = Ollama(
                model=MODEL_NAME, 
                base_url=OLLAMA_URL, 
                request_timeout=360.0 
            )
            
            # 2. Configurar Embeddings
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            )
            
            # 3. Conexión a Qdrant
            try:
                self.client = QdrantClient(url=QDRANT_URL)
                # Test de conexión
                self.client.get_collections()
                logger.info(f"✅ Conexión a Memoria Qdrant ({QDRANT_URL}): EXITOSA")
            except Exception as e:
                logger.error(f"❌ Fallo al conectar Qdrant en {QDRANT_URL}: {e}")
                raise e

            # 4. Storage Context
            self.vector_store = QdrantVectorStore(
                client=self.client, 
                collection_name=COLLECTION_NAME
            )
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # 5. Cargar Índice
            try:
                self.index = VectorStoreIndex.from_vector_store(
                    self.vector_store,
                )
                logger.info("✅ Índice Vectorial Cargado.")
            except Exception:
                logger.info("⚠️ Índice nuevo. Esperando ingesta de documentos.")
                
            self.active = True
            logger.info("🚀 SISTEMA CEREBRAL TOTALMENTE OPERATIVO.")

        except Exception as e:
            logger.error(f"❌ Error Crítico de Inicio: {e}")
            self.active = False

    def ingest_document(self, file_path: str):
        if not self.active: return False
        logger.info(f"📥 Ingestando archivo: {file_path}")
        
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
            
            logger.info(f"✅ Documento aprendido ({len(documents)} fragmentos).")
            return True
        except Exception as e:
            logger.error(f"❌ Error de Ingesta: {e}")
            return False

    def query(self, question: str):
        if not self.active: 
            return "Error: El cerebro está desconectado. Revisa la terminal."
        
        logger.info(f"🧠 Procesando pregunta: {question}")
        
        if self.index is None:
            return "No tengo información en mi memoria. Por favor sube un documento primero."
            
        try:
            query_engine = self.index.as_query_engine(
                similarity_top_k=3, 
            )
            response = query_engine.query(question)
            return str(response)
        except Exception as e:
            logger.error(f"Error en consulta: {e}")
            return "Ocurrió un error al pensar la respuesta."

brain = AbundaBrain()
