import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# --- CONFIGURACIÓN DEL SOBERANO ---
# En Heroku, usted configurará la variable de entorno GEMINI_API_KEY
# No escriba la llave directamente aquí por seguridad.
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    print("⚠️ ADVERTENCIA: No se encontró GEMINI_API_KEY. El sistema no funcionará hasta configurarlo.")

# Configurar Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Configuración del Modelo (Usamos Flash por velocidad y economía)
# System Instructions: Definimos la personalidad de Abunda aquí.
generation_config = {
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="Eres Abunda AI, el Sistema Operativo de Conocimiento Empresarial creado por Apollo Sol Invictus. Tu misión es ayudar a los empleados respondiendo preguntas basándote en la información provista. Eres profesional, preciso y útil. Si no sabes la respuesta, dilo honestamente."
)

app = FastAPI()

# Permitir que el Frontend (GitHub Pages) hable con el Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción, cambie "*" por su dominio de GitHub/ProfCR
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELOS DE DATOS ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "Abunda AI Brain Online", "model": "Gemini 1.5 Flash"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        if not GOOGLE_API_KEY:
            raise HTTPException(status_code=500, detail="API Key no configurada en el servidor.")

        # Iniciar sesión de chat con historial
        chat_history = []
        for msg in request.history:
            role = "user" if msg.role == "user" else "model"
            chat_history.append({"role": role, "parts": [msg.content]})

        chat = model.start_chat(history=chat_history)
        
        # Enviar mensaje a Gemini
        response = chat.send_message(request.message)
        
        return {
            "response": response.text,
            "sources": [] # En V2 implementaremos RAG real con documentos
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint Placeholder para subir archivos (Se conectará a la API de Archivos de Gemini en el futuro)
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename, "status": "Simulated Upload - Ready for Gemini File API integration"}

# Para correr localmente: uvicorn main:app --reload