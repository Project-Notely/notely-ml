from .recognition import perform_ocr, convert_to_recognition_results
from .indexing import index_text, search_text
from .main import process_handwritten_text, run_htr_demo

__all__ = [
    "perform_ocr",
    "convert_to_recognition_results",
    "index_text",
    "search_text",
    "process_handwritten_text",
    "run_htr_demo",
]
