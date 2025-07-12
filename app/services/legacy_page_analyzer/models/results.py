from dataclasses import dataclass
from typing import Any


@dataclass
class ProcessingResult:
    """Standard result format for all processors."""

    success: bool
    result: Any
    confidence: float | None = None
    metadata: dict[str, Any] | None = None
    error_message: str | None = None


@dataclass
class TextBox:
    """Represents a detected text box with location."""

    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x, y, width, height)
    word_level: bool = False


@dataclass
class OCRResult:
    """Result for OCR processing."""

    full_text: str
    text_boxes: list[TextBox]
    average_confidence: float
    total_words: int
    languages_used: list[str]
    processing_config: str
