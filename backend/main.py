# main.py
# Servidor API para Abunda OS con CORS totalmente abierto para desarrollo local.

import os
import shutil
import uvicorn
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configurar logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ABUNDA_API")

# Intentar importar el cerebro (si falla, usará modo simulación para que el server arranque igual)
try:
    from rag_engine import brain
    BRAIN_ACTIVE = True
except ImportError as e:
    logger.warning(f"⚠️ No se pudo cargar rag_engine: {e}. Iniciando en modo SIMULACIÓN.")
    BRAIN_ACTIVE = False

app = FastAPI(title="ABUNDA API v2.6")

# --- CONFIGURACIÓN CORS (PERMISIVA) ---
# Esto permite que cualquier origen (incluso archivos locales) haga peticiones.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def health_check():
    """Verifica si el servidor está vivo."""
    return {
        "status": "online", 
        "brain_active": BRAIN_ACTIVE,
        "model": "Llama 3" if BRAIN_ACTIVE else "Simulated"
    }

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat con la IA."""
    logger.info(f"📨 Chat request: {request.message}")
    
    if BRAIN_ACTIVE:
        try:
            response = brain.query(request.message)
            return {"response": response, "sources": ["Knowledge Base"]}
        except Exception as e:
            logger.error(f"❌ Error en cerebro: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Respuesta simulada si no hay cerebro conectado
        return {
            "response": f"[SIMULACIÓN] Backend conectado pero Llama 3 no disponible. Recibí: '{request.message}'",
            "sources": ["System"]
        }

@app.post("/api/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    """Subida de documentos."""
    logger.info(f"📂 Recibiendo archivo: {file.filename}")
    
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        if BRAIN_ACTIVE:
            success = brain.ingest_document(file_path)
            if success:
                return {"status": "success", "filename": file.filename}
            else:
                raise HTTPException(status_code=500, detail="Fallo al indexar en Qdrant")
        else:
            return {"status": "success", "filename": file.filename, "note": "Modo simulación (archivo guardado)"}
            
    except Exception as e:
        logger.error(f"❌ Error upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 ABUNDA Server v2.6 Iniciando...")
    # Escucha en 0.0.0.0 para aceptar conexiones de red local si es necesario
    uvicorn.run(app, host="0.0.0.0", port=8000)
