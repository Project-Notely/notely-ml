from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.page_analyzer.models.models import DocumentElement, ProcessingResult


class OCRProcessor(ABC):
    """Abstract base class for OCR processors"""

    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """
        Initialize the OCR processor

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def process_text_region(
        self,
        image_region: np.ndarray,
        preprocessing_options: Optional[Dict[str, Any]] = None,
    ) -> ProcessingResult:
        """Process a text region of an image

        Args:
            image_region (np.ndarray): The image region to process
            preprocessing_options (Optional[Dict[str, Any]], optional): Optional preprocessing options

        Returns:
            ProcessingResult: recognized text and confidence
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages

        Returns:
            list[str]: List of supported languages
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources"""
        pass


class ImageClassifier(ABC):
    """Abstract interface for image classifiers"""

    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """Initialize the image classifier

        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def classify_image(
        self, image: Union[Image.Image, np.ndarray], top_k: int = 1
    ) -> ProcessingResult:
        """Classify an image

        Args:
            image (Union[Image.Image, np.ndarray]): The image to classify
            top_k (int, optional): Number of top predictions to return. Defaults to 1.

        Returns:
            ProcessingResult: classification result
        """
        pass

    @abstractmethod
    def get_class_names(self) -> list[str]:
        """Get list of supported class names"""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources"""
        pass


class TableProcessor(ABC):
    """Abstract interface for table processing"""

    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """Initialize the table processor"""
        pass

    @abstractmethod
    def process_table(
        self, image_region: np.ndarray, extract_structure: bool = True
    ) -> ProcessingResult:
        """Process a table region

        Args:
            image_region: Image region containing table
            extract_structure: Whether to extract table structure

        Returns:
            ProcessingResult with table data and structure
        """
        pass


class DocumentSegmenter(ABC):
    """Abstract interface for document segmenters"""

    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """Initialize the document segmenter

        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def segment_document(
        self, file_path: Union[str, Path], strategy: str = "hi_res", **kwargs
    ) -> ProcessingResult:
        """Segment a document

        Args:
            file_path (Union[str, Path]): Path to the document file
            strategy (str, optional): Partitioning strategy. Defaults to "hi_res".
            **kwargs: Additional keyword arguments

        Returns:
            ProcessingResult with list of DocumentElement objects
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        """Get list of supported document formats"""
        pass


class RegionExtractor(ABC):
    """Abstract interface for extracting regions from images"""

    @abstractmethod
    def extract_regions(
        self, image: Union[str, Path, Image.Image, np.ndarray], element: DocumentElement
    ) -> ProcessingResult:
        """
        Extract image region for a document element

        Args:
            image: Source image
            element: Document element with coordinates

        Returns:
            ProcessingResult with extracted region as numpy array
        """
        pass


class ProcessorFactory(ABC):
    """Abstract factory for creating processors"""

    @abstractmethod
    def create_ocr_processor(self, processor_type: str, **kwargs) -> OCRProcessor:
        """Create an OCR processor"""
        pass

    @abstractmethod
    def create_image_classifier(
        self, classifier_type: str, **kwargs
    ) -> ImageClassifier:
        """Create an image classifier"""
        pass

    @abstractmethod
    def create_document_segmenter(
        self, segmenter_type: str, **kwargs
    ) -> DocumentSegmenter:
        """Create a document segmenter"""
        pass

    @abstractmethod
    def create_region_extractor(self, extractor_type: str, **kwargs) -> RegionExtractor:
        """Create a region extractor"""
        pass

    @abstractmethod
    def create_table_processor(self, processor_type: str, **kwargs) -> TableProcessor:
        """Create a table processor"""
        pass


class ProcessingPipeline(ABC):
    """Abstract interface for processing pipelines"""

    @abstractmethod
    def add_processor(self, processor_type: str, processor: Any) -> None:
        """Add a processor to the pipeline"""
        pass

    @abstractmethod
    def process_element(
        self, element: DocumentElement, image: Union[str, Path, Image.Image, np.ndarray]
    ) -> ProcessingResult:
        """Process a single document element"""
        pass

    @abstractmethod
    def process_document(
        self, file_path: Union[str, Path], **kwargs
    ) -> ProcessingResult:
        """Process an entire document"""
        pass
