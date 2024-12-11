from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from generate import router
from client import Client

app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)


asr_model = 'streaming_audio'  # ['sensetime', 'whisper_large-v2', 'streaming_audio']
tts_model = 'meloTTS'  # ['sensetime', 'meloTTS', 'f5-tts']
tp = 1
app.client = Client(asr_model, tts_model, tp)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add routes
app.include_router(router)

@app.get("/")
def read_root():
    return {"Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
