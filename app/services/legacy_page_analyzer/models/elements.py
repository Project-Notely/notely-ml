from dataclasses import dataclass
from typing import Any


@dataclass
class DocumentElement:
    """Represents a document element from unstructured analysis."""

    element_type: str
    text: str
    coordinates: dict[str, float] | None
    page_number: int | None
    confidence: float | None
    metadata: dict[str, Any]


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
