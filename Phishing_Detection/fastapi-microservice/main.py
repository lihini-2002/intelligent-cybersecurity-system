from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import torch
from contextlib import asynccontextmanager
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification

# --------- Configurable Paths ---------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"
SMS_MODEL_PATH = MODEL_DIR / "sms-bert-model"
URL_MODEL_PATH = MODEL_DIR / "url-bert-model/phishing_model_v1_after_phase1"

# --------- Request Schemas ---------
class SMSRequest(BaseModel):
    message: str

class URLRequest(BaseModel):
    url: str

# --------- Load Models and Tokenizers ---------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models at startup
    global sms_model, sms_tokenizer, url_model, url_tokenizer

    try:
        # Load SMS model (still BERT)
        sms_tokenizer = BertTokenizer.from_pretrained(str(SMS_MODEL_PATH))
        sms_model = BertForSequenceClassification.from_pretrained(str(SMS_MODEL_PATH))
        sms_model.eval()

        # Load URL model (now DistilBERT)
        url_tokenizer = DistilBertTokenizer.from_pretrained(str(URL_MODEL_PATH))
        url_model = DistilBertForSequenceClassification.from_pretrained(str(URL_MODEL_PATH))
        url_model.eval()

        print(" Models successfully loaded at startup")
    except Exception as e:
        print(" Error loading models:", e)
        raise e

    yield  # <-- this tells FastAPI: "I'm done with startup setup"

    # You could do cleanup code here if needed on shutdown


app = FastAPI(
    title="Phishing Detection API",
    lifespan=lifespan
)

# --------- Inference Routes ---------
@app.post("/predict/sms")
def predict_sms(request: SMSRequest):
    print(f"Incoming SMS: {request.message}")
    try:
        if len(request.message) > 500:
            raise HTTPException(status_code=400, detail="Message too long (max 500 characters)")

        inputs = sms_tokenizer(request.message, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = sms_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        predicted_class = torch.argmax(probs).item()
        confidence = probs[predicted_class].item()

        label = "phishing" if predicted_class == 1 else "legitimate"

        return {
            "class": predicted_class,
            "label": label,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SMS prediction failed: {str(e)}")


@app.post("/predict/url")
def predict_url(request: URLRequest):
    print(f"Incoming URL: {request.url}")
    try:
        if len(request.url) > 2048:
            raise HTTPException(status_code=400, detail="URL too long (max 2048 characters)")

        inputs = url_tokenizer(request.url, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = url_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        predicted_class = torch.argmax(probs).item()
        confidence = probs[predicted_class].item()

        label = "phishing" if predicted_class == 1 else "legitimate"

        return {
            "class": predicted_class,
            "label": label,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL prediction failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok"}
