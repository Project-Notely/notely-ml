from typing import Optional

from pydantic import BaseModel


class OCRResponse(BaseModel):
    success: bool
    text: str
    confidence: float
    error: Optional[str] = None


class TextBox(BaseModel):
    text: str
    confidence: float
    bbox: list[int]  # [x, y, width, height]
