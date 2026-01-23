import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# --- CONFIGURACIÓN ---
# Intentamos leer la llave de Heroku
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    print("⚠️ ADVERTENCIA: No se encontró GEMINI_API_KEY.")

# Configurar Gemini
try:
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Error configurando API Key: {e}")

# --- SELECCIÓN DINÁMICA DE MODELO ---
# Esta función busca el mejor modelo disponible para evitar errores 404
def get_best_model_name():
    if not GOOGLE_API_KEY:
        return "models/gemini-1.5-flash" 

    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        print(f"📋 Modelos encontrados: {available_models}")

        priorities = [
            'models/gemini-1.5-pro',
            'models/gemini-1.5-flash',
            'models/gemini-pro'
        ]

        for priority in priorities:
            match = next((m for m in available_models if priority in m), None)
            if match:
                return match
        
        if available_models:
            return available_models[0]
            
    except Exception as e:
        print(f"⚠️ Error listando modelos: {e}")
    
    return "models/gemini-1.5-flash"

MODEL_NAME = get_best_model_name()

generation_config = {
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = None

def initialize_model():
    global model
    global MODEL_NAME
    try:
        if not model:
            MODEL_NAME = get_best_model_name()
            
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config,
            system_instruction="Eres Abunda AI, el Sistema Operativo de Conocimiento Empresarial creado por Apollo Sol Invictus. Tu misión es ayudar a los empleados respondiendo preguntas basándote en la información provista. Eres profesional, preciso y útil."
        )
        print(f"🚀 SISTEMA LISTO CON MODELO: {MODEL_NAME}")
    except Exception as e:
        print(f"⚠️ Error inicializando modelo: {e}")

if GOOGLE_API_KEY:
    initialize_model()

app = FastAPI()

# --- CORS PERMISIVO ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permitir acceso desde cualquier sitio
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

@app.get("/")
def read_root():
    return {"status": "Abunda AI Brain Online", "model": MODEL_NAME}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    global model
    try:
        if not GOOGLE_API_KEY:
            raise HTTPException(status_code=500, detail="API Key no configurada en Heroku.")
        
        if not model:
            initialize_model()
            if not model:
                 raise HTTPException(status_code=500, detail="Fallo crítico al iniciar el modelo Gemini.")

        # Preparar historial para Python SDK
        chat_history = []
        for msg in request.history:
            role = "user" if msg.role == "user" else "model"
            if msg.content:
                chat_history.append({"role": role, "parts": [msg.content]})

        chat = model.start_chat(history=chat_history)
        response = chat.send_message(request.message)
        
        return {
            "response": response.text,
            "sources": [],
            "used_model": MODEL_NAME
        }

    except Exception as e:
        print(f"ERROR EN CHAT: {e}")
        # Intentar reiniciar modelo si hay error de sesión
        try:
            initialize_model()
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")