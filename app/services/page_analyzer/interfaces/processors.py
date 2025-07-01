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
