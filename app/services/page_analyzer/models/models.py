from dataclasses import dataclass


@dataclass
class WordMatch:
    """Represents a matched word with its bounding box."""

    word: str
    bbox: list[int]
    confidence: float
    gemini_word: str
    similarity: float


@dataclass
class TextBox:
    """Represents a detected text box."""

    text: str
    confidence: float
    bbox: list[int]  # [x, y, width, height]


@dataclass
class OCRResult:
    """OCR processing result."""

    full_text: str
    average_confidence: float
    total_words: int
    text_boxes: list[TextBox]


@dataclass
class ProcessingResult:
    """Complete processing result."""

    success: bool
    result: OCRResult | None = None
    error: str | None = None
