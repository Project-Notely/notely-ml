from pydantic import BaseModel
from typing import Optional


class OCRResponse(BaseModel):
    success: bool
    text: str
    confidence: float
    error: Optional[str] = None


class TextBox(BaseModel):
    text: str
    confidence: float
    bbox: list[int]  # [x, y, width, height]
