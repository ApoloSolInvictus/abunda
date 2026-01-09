import loggingimport logging
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

# --- AJUSTE DE RED (IPs Explícitas) ---
# Usamos 127.0.0.1 para máxima compatibilidad local
QDRANT_URL = "http://127.0.0.1:6333" 
COLLECTION_NAME = "abunda_knowledge"

# Ollama en puerto 11434
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL_NAME = "llama3"

class AbundaBrain:
    def __init__(self):
        logger.info(f"⚡ Inicializando ABUNDA Brain con {MODEL_NAME}...")
        self.active = False
        self.index = None
        
        try:
            # 1. Configurar LLM (Llama 3)
            Settings.llm = Ollama(
                model=MODEL_NAME, 
                base_url=OLLAMA_URL, 
                request_timeout=300.0 
            )
            
            # 2. Configurar Embeddings
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            )
            
            # 3. Conexión a Qdrant
            try:
                self.client = QdrantClient(url=QDRANT_URL)
                # Test de conexión simple
                self.client.get_collections()
                logger.info("✅ Conexión a Qdrant: EXITOSA")
            except Exception as e:
                logger.error(f"❌ Fallo al conectar Qdrant en {QDRANT_URL}. Asegúrate de que Docker esté corriendo.")
                # No lanzamos error fatal para que la API pueda iniciar en modo 'Offline'
                return

            # 4. Contexto de Almacenamiento
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
                logger.info("✅ Memoria Vectorial Cargada.")
            except Exception:
                logger.info("⚠️ Memoria vacía. El sistema está listo para aprender (Sube un documento).")
                self.index = None
                
            self.active = True
            logger.info("🚀 CEREBRO EN LÍNEA.")

        except Exception as e:
            logger.error(f"❌ Error General del Cerebro: {e}")
            self.active = False

    def ingest_document(self, file_path: str):
        if not self.active: 
            logger.error("Intento de ingesta con cerebro inactivo.")
            return False
            
        logger.info(f"📥 Aprendiendo documento: {file_path}")
        
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
            
            logger.info(f"✅ Documento procesado ({len(documents)} páginas/fragmentos).")
            return True
        except Exception as e:
            logger.error(f"❌ Error de Ingesta: {e}")
            return False

    def query(self, question: str):
        if not self.active: 
            return "Error: El cerebro está desconectado. Verifica que Docker (Qdrant) y Ollama estén corriendo."
        
        logger.info(f"🧠 Analizando: {question}")
        
        if self.index is None:
            return "No tengo conocimiento almacenado aún. Por favor sube un documento PDF o TXT primero."
            
        try:
            # Configurar motor de chat
            query_engine = self.index.as_query_engine(
                similarity_top_k=3, # Usar las 3 mejores referencias
            )
            response = query_engine.query(question)
            return str(response)
        except Exception as e:
            logger.error(f"Error en consulta: {e}")
            return f"Ocurrió un error al generar la respuesta: {str(e)}"

# Instancia Global
brain = AbundaBrain()
