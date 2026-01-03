import logging
import sys
import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    Document
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient

# --- CONFIGURACIÓN ---
# Asegúrate de tener Qdrant corriendo en Docker: docker run -p 6333:6333 qdrant/qdrant
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "abunda_knowledge"

# Configuración de Ollama (Llama 3)
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3"

# Configuración de Logs
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("ABUNDA_BRAIN")

class AbundaBrain:
    def __init__(self):
        logger.info(f"⚡ Inicializando ABUNDA Brain con {MODEL_NAME}...")
        self.active = False
        
        try:
            # 1. Configurar LLM (Llama 3 Local)
            Settings.llm = Ollama(
                model=MODEL_NAME, 
                base_url=OLLAMA_URL, 
                request_timeout=360.0 
            )
            
            # 2. Configurar Embeddings (Modelo Local Rápido)
            # Usamos un modelo estándar de HuggingFace que no requiere API Key
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            )
            
            # 3. Conexión a Memoria Vectorial (Qdrant)
            self.client = QdrantClient(url=QDRANT_URL)
            self.vector_store = QdrantVectorStore(
                client=self.client, 
                collection_name=COLLECTION_NAME
            )
            
            # 4. Contexto de Almacenamiento
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # Intentar cargar índice existente
            try:
                self.index = VectorStoreIndex.from_vector_store(
                    self.vector_store,
                )
                logger.info("✅ Memoria vectorial cargada.")
            except Exception:
                logger.info("⚠️ Índice vacío. Se creará al subir el primer documento.")
                self.index = None
                
            self.active = True
            logger.info("🚀 SISTEMA CEREBRAL EN LÍNEA")

        except Exception as e:
            logger.error(f"❌ Error crítico iniciando el cerebro: {e}")
            self.active = False

    def ingest_document(self, file_path: str):
        """Lee, fragmenta y vectoriza un documento."""
        if not self.active: return False
        logger.info(f"📥 Ingestando: {file_path}")
        
        try:
            # LlamaIndex detecta el tipo de archivo automáticamente
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
            
            if self.index is None:
                # Crear nuevo índice si es el primer documento
                self.index = VectorStoreIndex.from_documents(
                    documents, 
                    storage_context=self.storage_context
                )
            else:
                # Insertar en índice existente
                for doc in documents:
                    self.index.insert(doc)
            
            logger.info(f"✅ {len(documents)} fragmentos indexados en Qdrant.")
            return True
        except Exception as e:
            logger.error(f"❌ Error de ingestión: {e}")
            return False

    def query(self, question: str):
        """Consulta inteligente al cerebro."""
        if not self.active: return "Error: Cerebro desconectado. Verifica Ollama y Qdrant."
        
        logger.info(f"🧠 Pensando: {question}")
        
        if self.index is None:
            return "La base de conocimiento está vacía. Por favor sube un documento PDF o TXT primero."
            
        # Motor de Chat con Contexto
        # similarity_top_k=3 significa que busca los 3 párrafos más relevantes del PDF
        query_engine = self.index.as_query_engine(
            similarity_top_k=3, 
        )
        
        response = query_engine.query(question)
        return str(response)

# Instancia Global
brain = AbundaBrain()
