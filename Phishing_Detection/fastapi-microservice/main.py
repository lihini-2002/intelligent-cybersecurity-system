from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import torch
import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from contextlib import asynccontextmanager
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification

# --------- Configurable Paths ---------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
SMS_MODEL_PATH = MODEL_DIR / "sms-bert-model"
URL_MODEL_PATH = MODEL_DIR / "url-bert-model/phishing_model_v1_after_phase1"

def download_model_from_s3(bucket_name, s3_prefix, local_dir):
    """
    Downloads an entire folder (prefix) from an S3 bucket to a local directory.
    """
    s3 = boto3.client('s3')

    try:
        # List all objects in the S3 folder (prefix)
        objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix)

        if 'Contents' not in objects:
            print(f"No objects found at s3://{bucket_name}/{s3_prefix}")
            return False

        for obj in objects['Contents']:
            s3_key = obj['Key']
            file_name = s3_key.replace(s3_prefix, "").lstrip("/")

            if not file_name:  # It's a folder
                continue

            local_path = os.path.join(local_dir, file_name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket_name, s3_key, local_path)
            print(f"Downloaded {s3_key} to {local_path}")

        return True

    except (NoCredentialsError, ClientError) as e:
        print(f"Failed to download from S3: {e}")
        return False


# --------- Request Schemas ---------
class SMSRequest(BaseModel):
    message: str

class URLRequest(BaseModel):
    url: str

# --------- Load Models and Tokenizers ---------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global sms_model, sms_tokenizer, url_model, url_tokenizer

    bucket_name = "phishing-models-lihini-2002"
    
    sms_s3_prefix = "sms-bert-model"
    url_s3_prefix = "url-bert-model/phishing_model_v1_after_phase1"

    sms_local_path = BASE_DIR / "models/sms-bert-model"
    url_local_path = BASE_DIR / "models/url-bert-model/phishing_model_v1_after_phase1"

    # Download SMS model
    print("Downloading SMS model...")
    download_model_from_s3(bucket_name, sms_s3_prefix, sms_local_path)

    print("Downloading URL model...")
    download_model_from_s3(bucket_name, url_s3_prefix, url_local_path)

    print("Loading tokenizers and models into memory...")
    try:
        sms_tokenizer = BertTokenizer.from_pretrained(str(sms_local_path))
        sms_model = BertForSequenceClassification.from_pretrained(str(sms_local_path))
        sms_model.eval()

        url_tokenizer = DistilBertTokenizer.from_pretrained(str(url_local_path))
        url_model = DistilBertForSequenceClassification.from_pretrained(str(url_local_path))
        url_model.eval()

        print("Models successfully loaded from S3 at startup")

    except Exception as e:
        print(" Error loading models:", e)
        raise e

    yield  # Startup done

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
