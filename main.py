from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.model import predict_indobert, predict_lstm, predict_gru
from fastapi.staticfiles import StaticFiles


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/predict_all")
async def predict_all(input: dict):
    text = input["text"]

    return {
        "lstm": predict_lstm(text),
        "gru": predict_gru(text),
        "indobert": predict_indobert(text)
    }