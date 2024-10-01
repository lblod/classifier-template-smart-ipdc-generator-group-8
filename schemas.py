from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class PredictRequest(BaseModel):
    description: str


class Prediction(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    prediction: List[Prediction] = []