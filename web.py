from schemas import PredictRequest, PredictResponse
from functools import lru_cache
from fastapi import Request, Depends
from typing import Any
import requests
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_path: str = '/app/model'


@lru_cache(maxsize=None)
def get_model():
    settings = Settings()
    fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(settings.model_path)
    tokenizer = AutoTokenizer.from_pretrained(settings.model_path)
    clf = pipeline("text-classification", fine_tuned_model, tokenizer=tokenizer)
    return clf


@app.post("/predict")
async def request_processing(
    body: PredictRequest,
    model: Any = Depends(get_model)
) -> PredictResponse:
    answer = model(body.description)
    return PredictResponse(prediction=answer)



