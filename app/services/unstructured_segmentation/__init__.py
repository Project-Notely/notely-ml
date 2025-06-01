"""
Unstructured-based document segmentation service
"""

from .main import (
    segment_document_unstructured,
    process_pdf_with_highlights,
    UnstructuredSegmentationService,
    DocumentElement,
)

__all__ = [
    "segment_document_unstructured",
    "process_pdf_with_highlights", 
    "UnstructuredSegmentationService",
    "DocumentElement",
] 