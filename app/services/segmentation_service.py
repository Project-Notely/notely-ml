"""
Document segmentation service using unstructured.io
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from unstructured.documents.elements import Element
from unstructured.partition.auto import partition
from unstructured.partition.utils.constants import PartitionStrategy

from app.models.segmentation_models import (
    BoundingBox,
    DocumentSegment,
    SegmentationResult,
    SegmentType,
)

logger = logging.getLogger(__name__)


class SegmentationService:
    """
    Service for document segmentation using unstructured.io
    """

    def __init__(self):
        """Initialize the segmentation service"""
        self.supported_formats = {
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".bmp",
            ".pdf",
            ".docx",
            ".doc",
            ".txt",
            ".html",
            ".md",
        }

    async def segment_document(
        self,
        file_path: Union[str, Path],
        strategy: str = "hi_res",
        extract_images: bool = True,
        infer_table_structure: bool = True,
    ) -> SegmentationResult:
        """
        Segment a document into different element types

        Args:
            file_path: Path to the document file
            strategy: Partitioning strategy ('fast', 'hi_res', 'auto')
            extract_images: Whether to extract images
            infer_table_structure: Whether to infer table structure

        Returns:
            SegmentationResult: Segmentation results with bounding boxes
        """
        try:
            file_path = Path(file_path)

            # Validate file format
            if file_path.suffix.lower() not in self.supported_formats:
                return SegmentationResult(
                    success=False,
                    error=f"Unsupported file format: {file_path.suffix}",
                    segments=[],
                )

            # Configure partitioning strategy
            partition_strategy = self._get_partition_strategy(strategy)

            # Partition the document
            elements = partition(
                filename=str(file_path),
                strategy=partition_strategy,
                extract_images=extract_images,
                infer_table_structure=infer_table_structure,
                include_metadata=True,
            )

            # Convert elements to segments
            segments = self._convert_elements_to_segments(elements)

            # Calculate statistics
            stats = self._calculate_statistics(segments)

            return SegmentationResult(
                success=True,
                segments=segments,
                total_segments=len(segments),
                statistics=stats,
                strategy_used=strategy,
            )

        except Exception as e:
            logger.error(f"Document segmentation failed: {str(e)}")
            return SegmentationResult(success=False, error=str(e), segments=[])

    async def segment_image(
        self, image: Union[str, Path, Image.Image, np.ndarray], strategy: str = "hi_res"
    ) -> SegmentationResult:
        """
        Segment an image file

        Args:
            image: Image file path or PIL Image or numpy array
            strategy: Partitioning strategy

        Returns:
            SegmentationResult: Segmentation results
        """
        temp_path = None
        try:
            # Handle different input types
            if isinstance(image, (str, Path)):
                file_path = Path(image)
            elif isinstance(image, Image.Image):
                # Save PIL Image to temporary file
                temp_path = self._save_temp_image(image)
                file_path = Path(temp_path)
            elif isinstance(image, np.ndarray):
                # Convert numpy array to PIL Image and save
                pil_image = Image.fromarray(image)
                temp_path = self._save_temp_image(pil_image)
                file_path = Path(temp_path)
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

            # Use the main segmentation method
            return await self.segment_document(
                file_path=file_path,
                strategy=strategy,
                extract_images=True,
                infer_table_structure=True,
            )

        except Exception as e:
            logger.error(f"Image segmentation failed: {str(e)}")
            return SegmentationResult(success=False, error=str(e), segments=[])
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def _get_partition_strategy(self, strategy: str) -> PartitionStrategy:
        """Get the appropriate partition strategy"""
        strategy_mapping = {
            "fast": PartitionStrategy.FAST,
            "hi_res": PartitionStrategy.HI_RES,
            "auto": PartitionStrategy.AUTO,
        }
        return strategy_mapping.get(strategy, PartitionStrategy.HI_RES)

    def _convert_elements_to_segments(
        self, elements: List[Element]
    ) -> List[DocumentSegment]:
        """Convert unstructured elements to our segment format"""
        segments = []

        for element in elements:
            try:
                # Get element type
                segment_type = self._map_element_type(element.category)

                # Get bounding box if available
                bbox = None
                if hasattr(element, "metadata") and element.metadata:
                    coordinates = getattr(element.metadata, "coordinates", None)
                    if coordinates:
                        bbox = self._extract_bounding_box(coordinates)

                # Create segment
                segment = DocumentSegment(
                    text=str(element).strip(),
                    segment_type=segment_type,
                    bbox=bbox,
                    confidence=self._calculate_confidence(element),
                    metadata=self._extract_metadata(element),
                )

                segments.append(segment)

            except Exception as e:
                logger.warning(f"Failed to convert element: {str(e)}")
                continue

        return segments

    def _map_element_type(self, element_category: str) -> SegmentType:
        """Map unstructured element category to our segment type"""
        mapping = {
            "Title": SegmentType.TITLE,
            "NarrativeText": SegmentType.PARAGRAPH,
            "Table": SegmentType.TABLE,
            "ListItem": SegmentType.LIST_ITEM,
            "Image": SegmentType.IMAGE,
            "Figure": SegmentType.FIGURE,
            "Header": SegmentType.HEADER,
            "Footer": SegmentType.FOOTER,
            "FigureCaption": SegmentType.CAPTION,
            "Address": SegmentType.TEXT,
            "EmailAddress": SegmentType.TEXT,
            "UncategorizedText": SegmentType.TEXT,
        }
        return mapping.get(element_category, SegmentType.TEXT)

    def _extract_bounding_box(self, coordinates) -> Optional[BoundingBox]:
        """Extract bounding box from coordinates"""
        try:
            if hasattr(coordinates, "points"):
                points = coordinates.points
                if len(points) >= 2:
                    # Get min/max coordinates
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]

                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    return BoundingBox(
                        x=int(x_min),
                        y=int(y_min),
                        width=int(x_max - x_min),
                        height=int(y_max - y_min),
                    )
        except Exception as e:
            logger.warning(f"Failed to extract bounding box: {str(e)}")

        return None

    def _calculate_confidence(self, element: Element) -> float:
        """Calculate confidence score for an element"""
        # Basic confidence based on text length and element type
        text_length = len(str(element).strip())

        if text_length == 0:
            return 0.1
        elif text_length < 5:
            return 0.6
        elif text_length < 20:
            return 0.8
        else:
            return 0.95

    def _extract_metadata(self, element: Element) -> Dict[str, Any]:
        """Extract metadata from an element"""
        metadata = {
            "category": element.category,
            "text_length": len(str(element).strip()),
        }

        if hasattr(element, "metadata") and element.metadata:
            # Add relevant metadata fields
            for key in ["page_number", "filename", "file_directory"]:
                if hasattr(element.metadata, key):
                    metadata[key] = getattr(element.metadata, key)

        return metadata

    def _calculate_statistics(self, segments: List[DocumentSegment]) -> Dict[str, Any]:
        """Calculate segmentation statistics"""
        stats = {
            "total_segments": len(segments),
            "segment_types": {},
            "with_bounding_boxes": 0,
            "average_confidence": 0.0,
        }

        if not segments:
            return stats

        # Count segment types
        for segment in segments:
            segment_type = segment.segment_type.value
            stats["segment_types"][segment_type] = (
                stats["segment_types"].get(segment_type, 0) + 1
            )

            if segment.bbox:
                stats["with_bounding_boxes"] += 1

        # Calculate average confidence
        total_confidence = sum(segment.confidence for segment in segments)
        stats["average_confidence"] = total_confidence / len(segments)

        return stats

    def _save_temp_image(self, image: Image.Image) -> str:
        """Save PIL Image to temporary file"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file.name, "PNG")
            return temp_file.name

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return list(self.supported_formats)
