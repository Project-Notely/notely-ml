"""Concrete implementations of the abstract processors."""

from pathlib import Path
from typing import Any

import cv2
import easyocr
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

from ..interfaces.interfaces import (
    DocumentSegmenter,
    ImageClassifier,
    OCRProcessor,
    RegionExtractor,
    TableProcessor,
)
from ..main import UnstructuredSegmentationService
from ..models.elements import DocumentElement, ProcessingResult


class EasyOCRProcessor(OCRProcessor):
    """EasyOCR implementation of OCR processor."""

    def __init__(self):
        self.reader = None
        self.languages = ["en"]

    def initialize(self, languages: list[str] | None = None, **kwargs) -> bool:
        """Initialize EasyOCR."""
        try:
            if languages:
                self.languages = languages
            self.reader = easyocr.Reader(self.languages)
            return True
        except Exception as e:
            print(f"Failed to initialize EasyOCR: {e}")
            return False

    def process_text_region(
        self,
        image_region: np.ndarray,
        preprocessing_options: dict[str, Any] | None = None,
    ) -> ProcessingResult:
        """Process text region with EasyOCR."""
        if self.reader is None:
            return ProcessingResult(
                success=False, result=None, error_message="OCR reader not initialized"
            )

        try:
            # Apply preprocessing if specified
            if preprocessing_options:
                # Simple brightness/contrast adjustment
                if "brightness" in preprocessing_options:
                    image_region = cv2.convertScaleAbs(
                        image_region, alpha=1, beta=preprocessing_options["brightness"]
                    )
                if "contrast" in preprocessing_options:
                    image_region = cv2.convertScaleAbs(
                        image_region, alpha=preprocessing_options["contrast"], beta=0
                    )

            # Perform OCR
            results = self.reader.readtext(image_region)

            if not results:
                return ProcessingResult(success=True, result="", confidence=0.0)

            # Combine results
            text_parts = []
            confidences = []

            for _bbox, text, confidence in results:
                text_parts.append(text)
                confidences.append(confidence)

            combined_text = " ".join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return ProcessingResult(
                success=True,
                result=combined_text,
                confidence=avg_confidence,
                metadata={
                    "num_detections": len(results),
                    "individual_results": results,
                },
            )

        except Exception as e:
            return ProcessingResult(success=False, result=None, error_message=str(e))

    def get_supported_languages(self) -> list[str]:
        """Get supported languages."""
        # Common EasyOCR languages
        return ["en", "ch_sim", "ch_tra", "fr", "de", "ja", "ko", "es"]

    def cleanup(self) -> None:
        """Clean up resources."""
        self.reader = None


class ResNetImageClassifier(ImageClassifier):
    """ResNet50-based image classifier."""

    def __init__(self):
        self.model = None
        self.transform = None
        self.class_names = []

    def initialize(self, **kwargs) -> bool:
        """Initialize ResNet classifier."""
        try:
            # Load pretrained ResNet50
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.model.eval()

            # Define preprocessing
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            # Load basic class names
            self._load_basic_classes()
            return True

        except Exception as e:
            print(f"Failed to initialize ResNet classifier: {e}")
            return False

    def _load_basic_classes(self):
        """Load basic class names."""
        self.class_names = [
            "person",
            "car",
            "dog",
            "cat",
            "bird",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "truck",
            "boat",
            "train",
            "motorcycle",
            "bicycle",
            "airplane",
            "bus",
            "book",
            "clock",
            "scissors",
            "teddy bear",
            "table",
            "chair",
            "sofa",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "refrigerator",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
        ]

    def classify_image(
        self, image: Image.Image | np.ndarray, top_k: int = 1
    ) -> ProcessingResult:
        """Classify image with ResNet."""
        if self.model is None:
            return ProcessingResult(
                success=False, result=None, error_message="Classifier not initialized"
            )

        try:
            # Convert to PIL if numpy array
            if isinstance(image, np.ndarray):
                if image.ndim == 3 and image.shape[2] == 3:
                    # BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Preprocess
            input_tensor = self.transform(image)
            input_batch = input_tensor.unsqueeze(0)

            # Classify
            with torch.no_grad():
                output = self.model(input_batch)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)

                # Get top predictions
                top_probs, top_classes = torch.topk(
                    probabilities, min(top_k, len(probabilities))
                )

                predictions = []
                for i in range(len(top_probs)):
                    class_idx = top_classes[i].item()
                    confidence = top_probs[i].item()

                    if class_idx < len(self.class_names):
                        class_name = self.class_names[class_idx]
                    else:
                        class_name = f"class_{class_idx}"

                    predictions.append(
                        {
                            "class": class_name,
                            "confidence": confidence,
                            "class_id": class_idx,
                        }
                    )

                return ProcessingResult(
                    success=True,
                    result=predictions[0]["class"] if predictions else "unknown",
                    confidence=predictions[0]["confidence"] if predictions else 0.0,
                    metadata={"all_predictions": predictions, "top_k": top_k},
                )

        except Exception as e:
            return ProcessingResult(success=False, result=None, error_message=str(e))

    def get_class_names(self) -> list[str]:
        """Get supported class names."""
        return self.class_names.copy()

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            del self.model
        self.model = None


class UnstructuredDocumentSegmenter(DocumentSegmenter):
    """Unstructured library-based document segmenter."""

    def __init__(self):
        self.service = UnstructuredSegmentationService()

    def initialize(self, **kwargs) -> bool:
        """Initialize the segmenter."""
        # Unstructured service doesn't need explicit initialization
        return True

    def segment_document(
        self, file_path: str | Path, strategy: str = "hi_res", **kwargs
    ) -> ProcessingResult:
        """Segment document using unstructured."""
        try:
            # Check if file exists first
            file_path = Path(file_path)
            if not file_path.exists():
                return ProcessingResult(
                    success=False,
                    result=None,
                    error_message=f"File not found: {file_path}",
                )

            elements = self.service.segment_document(
                file_path=file_path, strategy=strategy, **kwargs
            )

            # Check if segmentation actually succeeded
            # The service returns empty list on error instead of raising exception
            if not elements:
                return ProcessingResult(
                    success=False,
                    result=None,
                    error_message="Document segmentation failed - no elements found",
                )

            return ProcessingResult(
                success=True,
                result=elements,
                metadata={
                    "num_elements": len(elements),
                    "strategy": strategy,
                    "file_path": str(file_path),
                },
            )

        except Exception as e:
            return ProcessingResult(success=False, result=None, error_message=str(e))

    def get_supported_formats(self) -> list[str]:
        """Get supported document formats."""
        return [
            "pdf",
            "png",
            "jpg",
            "jpeg",
            "tiff",
            "bmp",
            "doc",
            "docx",
            "html",
            "txt",
        ]


class DefaultRegionExtractor(RegionExtractor):
    """Default implementation for extracting regions from images."""

    def extract_regions(
        self, image: str | Path | Image.Image | np.ndarray, element: DocumentElement
    ) -> ProcessingResult:
        """Extract region from image based on element coordinates."""
        try:
            # Load image if path provided
            if isinstance(image, str | Path):
                img_array = cv2.imread(str(image))
            elif isinstance(image, Image.Image):
                img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                img_array = image

            if img_array is None:
                return ProcessingResult(
                    success=False, result=None, error_message="Could not load image"
                )

            # Extract coordinates from element
            coords = element.coordinates
            if not coords:
                # Return full image if no coordinates
                return ProcessingResult(
                    success=True,
                    result=img_array,
                    metadata={"region_type": "full_image"},
                )

            # Get bounding box
            x1 = int(coords.get("x1", 0))
            y1 = int(coords.get("y1", 0))
            x2 = int(coords.get("x2", img_array.shape[1]))
            y2 = int(coords.get("y2", img_array.shape[0]))

            # Ensure coordinates are within image bounds
            h, w = img_array.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(x1, min(x2, w))
            y2 = max(y1, min(y2, h))

            # Extract region
            region = img_array[y1:y2, x1:x2]

            return ProcessingResult(
                success=True,
                result=region,
                metadata={
                    "coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "region_shape": region.shape,
                },
            )

        except Exception as e:
            return ProcessingResult(success=False, result=None, error_message=str(e))


class BasicTableProcessor(TableProcessor):
    """Basic table processor (placeholder implementation)."""

    def initialize(self, **kwargs) -> bool:
        """Initialize table processor."""
        return True

    def process_table(
        self, image_region: np.ndarray, extract_structure: bool = True
    ) -> ProcessingResult:
        """Basic table processing (placeholder)."""
        try:
            # This is a placeholder - in practice you'd use a proper table extraction
            # library
            # like PaddleOCR's table recognition or similar

            height, width = image_region.shape[:2]

            result = {
                "table_detected": True,
                "estimated_rows": max(1, height // 30),  # Rough estimate
                "estimated_cols": max(1, width // 100),  # Rough estimate
                "extraction_method": "basic_estimation",
            }

            if extract_structure:
                result["structure"] = {
                    "type": "basic_grid",
                    "confidence": 0.5,  # Low confidence for basic method
                }

            return ProcessingResult(
                success=True,
                result=result,
                confidence=0.5,
                metadata={"method": "basic_placeholder"},
            )

        except Exception as e:
            return ProcessingResult(success=False, result=None, error_message=str(e))
