import os
import shutil
import uvicorn
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configuración de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ABUNDA_API")

# Intentar conectar con el cerebro (rag_engine.py)
try:
    from rag_engine import brain
    BRAIN_ACTIVE = True
    logger.info("✅ Cerebro Llama 3 detectado y vinculado.")
except ImportError as e:
    logger.warning(f"⚠️ No se pudo cargar rag_engine: {e}. Iniciando en modo SIMULACIÓN.")
    BRAIN_ACTIVE = False

app = FastAPI(title="ABUNDA API v4.0")

# --- CONFIGURACIÓN CORS BLINDADA ---
# Permitimos * (todos) para que Ngrok no bloquee la conexión entrante
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
    """Ping para verificar si el sistema está vivo."""
    return {
        "status": "online", 
        "brain": "Llama 3" if BRAIN_ACTIVE else "Simulated",
        "port": 8000
    }

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Procesa mensajes de chat."""
    logger.info(f"📨 Chat recibido: {request.message}")
    
    if BRAIN_ACTIVE:
        try:
            # Enviar al cerebro
            response = brain.query(request.message)
            return {"response": response, "sources": ["Base de Conocimiento"]}
        except Exception as e:
            logger.error(f"❌ Error en cerebro: {e}")
            raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
    else:
        return {
            "response": f"[SIMULACIÓN] Backend conectado. Llama 3 no respondió, pero la API sí. Mensaje: '{request.message}'",
            "sources": ["System Check"]
        }

@app.post("/api/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    """Procesa subida de documentos."""
    logger.info(f"📂 Recibiendo archivo: {file.filename}")
    
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        if BRAIN_ACTIVE:
            success = brain.ingest_document(file_path)
            if success:
                return {"status": "success", "filename": file.filename, "message": "Indexado en Qdrant"}
            else:
                raise HTTPException(status_code=500, detail="Fallo al indexar")
        else:
            return {"status": "success", "filename": file.filename, "note": "Modo simulación (archivo guardado)"}
            
    except Exception as e:
        logger.error(f"❌ Error upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 ABUNDA Server Iniciando en PUERTO 8000...")
    print("👉 Asegúrate que tu Ngrok apunte a 8000: ngrok http 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
