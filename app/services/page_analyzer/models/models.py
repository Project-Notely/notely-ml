from dataclasses import dataclass
from typing import Optional


@dataclass
class TextBox:
    """Represents a detected text box"""

    text: str
    confidence: float
    bbox: list[int]  # [x, y, width, height]


@dataclass
class OCRResult:
    """OCR processing result"""

    full_text: str
    average_confidence: float
    total_words: int
    text_boxes: list[TextBox]


@dataclass
class ProcessingResult:
    """Complete processing result"""

    success: bool
    result: Optional[OCRResult] = None
    error: Optional[str] = None
