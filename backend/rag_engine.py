# abunda/core/rag_engine.py
# Implementación de Épica 1: Core RAG

import os
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    ServiceContext, 
    StorageContext
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient

# Configuración
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "abunda_knowledge_base"
OLLAMA_BASE_URL = "http://localhost:11434" # Puerto por defecto de Ollama
MODEL_NAME = "llama3" # Asegúrate de tener llama3 instalado: `ollama pull llama3`

class AbundaBrain:
    def __init__(self):
        print("⚡ Inicializando ABUNDA Brain (Llama 3 + Qdrant)...")
        
        # 1. Conexión a Vector Store (Memoria a Largo Plazo)
        self.client = QdrantClient(url=QDRANT_URL)
        self.vector_store = QdrantVectorStore(
            client=self.client, 
            collection_name=COLLECTION_NAME
        )
        
        # 2. Configurar Modelo de Embeddings (Local - Gratis)
        # Usamos un modelo ligero y eficiente de HuggingFace
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5" 
        )
        
        # 3. Configurar LLM (Llama 3 Local)
        self.llm = Ollama(
            model=MODEL_NAME, 
            base_url=OLLAMA_BASE_URL,
            request_timeout=120.0
        )
        
        # 4. Contexto de Servicio
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model,
            chunk_size=512 # Tamaño de fragmento óptimo para RAG
        )
        
        # Inicializar índice (crear o cargar)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        try:
            # Intentar cargar índice existente
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store, 
                service_context=self.service_context
            )
            print("✅ Memoria cargada exitosamente.")
        except Exception:
            # Si es nuevo, iniciar vacío
            self.index = []
            print("⚠️ Memoria vacía. Esperando documentos.")

    def ingest_document(self, file_path: str):
        """
        Ingesta de documentos (PDF, TXT, etc.) a la memoria vectorial.
        """
        print(f"📥 Procesando documento: {file_path}")
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        
        if not self.index:
            self.index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=self.storage_context,
                service_context=self.service_context
            )
        else:
            for doc in documents:
                self.index.insert(doc)
        
        print(f"✅ Documento vectorizado y almacenado en Qdrant.")
        return True

    def query(self, question: str):
        """
        Consulta al sistema RAG.
        """
        print(f"🧠 Pensando: {question}")
        if not self.index:
            return "El sistema está vacío. Por favor carga documentos primero."
            
        query_engine = self.index.as_query_engine(
            similarity_top_k=3, # Traer los 3 fragmentos más relevantes
            streaming=False
        )
        response = query_engine.query(question)
        return str(response)

# Instancia Global
brain = AbundaBrain()