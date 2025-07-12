from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class ProcessingResult:
    """Standard result format for all processors"""

    success: bool
    result: Any
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class TextBox:
    """Represents a detected text box with location"""

    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    word_level: bool = False


@dataclass
class OCRResult:
    """Result for OCR processing"""

    full_text: str
    text_boxes: list[TextBox]
    average_confidence: float
    total_words: int
    languages_used: list[str]
    processing_config: str
