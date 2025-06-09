from dataclasses import dataclass


@dataclass
class TrOCRConfig:
    """Configuration for TrOCR OCR processing"""

    model_name: str
    processor_name: str
    description: str
    best_for: str
