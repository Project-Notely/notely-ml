"""
Main unstructured document segmentation service

This service uses the unstructured library to segment documents into different
types of elements (paragraphs, tables, images, etc.) and creates highlighted
PDF outputs showing the segmentation results.
"""

import io
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import json
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pdf2image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.colors import Color
from reportlab.lib.units import inch

from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf

from ..common.utils import save_output_image


@dataclass
class DocumentElement:
    """Represents a document element from unstructured analysis"""

    element_type: str
    text: str
    coordinates: Optional[Dict[str, float]]
    page_number: Optional[int]
    confidence: Optional[float]
    metadata: Dict[str, Any]


class UnstructuredSegmentationService:
    """Service for document segmentation using unstructured library"""

    # Color mapping for different element types
    ELEMENT_COLORS = {
        "Title": (255, 0, 0, 100),  # Red
        "NarrativeText": (0, 255, 0, 100),  # Green
        "Text": (0, 255, 0, 100),  # Green
        "ListItem": (0, 0, 255, 100),  # Blue
        "Table": (255, 255, 0, 100),  # Yellow
        "Image": (255, 0, 255, 100),  # Magenta
        "Figure": (255, 0, 255, 100),  # Magenta
        "Header": (255, 165, 0, 100),  # Orange
        "Footer": (128, 0, 128, 100),  # Purple
        "PageBreak": (128, 128, 128, 100),  # Gray
        "FigureCaption": (0, 255, 255, 100),  # Cyan
        "Address": (255, 192, 203, 100),  # Pink
        "EmailAddress": (255, 192, 203, 100),  # Pink
    }

    def __init__(self):
        """Initialize the segmentation service"""
        pass

    def segment_document(
        self,
        file_path: Union[str, Path],
        strategy: str = "hi_res",
        extract_images: bool = True,
        include_page_breaks: bool = True,
    ) -> List[DocumentElement]:
        """
        Segment a document using unstructured library

        Args:
            file_path: Path to the document file
            strategy: Partitioning strategy ('fast', 'hi_res', 'auto')
            extract_images: Whether to extract image elements
            include_page_breaks: Whether to include page break elements

        Returns:
            List of DocumentElement objects
        """
        file_path = Path(file_path)

        try:
            if file_path.suffix.lower() == ".pdf":
                # Use PDF-specific partitioning for better results
                elements = partition_pdf(
                    filename=str(file_path),
                    strategy=strategy,
                    extract_images_in_pdf=extract_images,
                    include_page_breaks=include_page_breaks,
                    infer_table_structure=True,
                )
            else:
                # Use auto partitioning for other file types
                elements = partition(
                    filename=str(file_path),
                    strategy=strategy,
                    include_page_breaks=include_page_breaks,
                )

            # Convert to our DocumentElement format
            document_elements = []
            for element in elements:
                doc_element = DocumentElement(
                    element_type=element.category,
                    text=str(element),
                    coordinates=getattr(element, "coordinates", None),
                    page_number=getattr(element.metadata, "page_number", None),
                    confidence=getattr(element, "confidence", None),
                    metadata=(
                        element.metadata.to_dict()
                        if hasattr(element.metadata, "to_dict")
                        else {}
                    ),
                )
                document_elements.append(doc_element)

            return document_elements

        except Exception as e:
            print(f"Error segmenting document: {e}")
            return []

    def convert_pdf_to_images(self, pdf_path: Union[str, Path]) -> List[Image.Image]:
        """Convert PDF pages to PIL Images"""
        try:
            images = pdf2image.convert_from_path(str(pdf_path), dpi=200)
            return images
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            return []

    def draw_highlights_on_image(
        self, image: Image.Image, elements: List[DocumentElement], page_number: int
    ) -> Image.Image:
        """
        Draw highlight boxes on an image for elements on a specific page

        Args:
            image: PIL Image to draw on
            elements: List of document elements
            page_number: Page number to filter elements for

        Returns:
            PIL Image with highlights drawn
        """
        # Create a copy to draw on
        img_with_highlights = image.copy()

        # Create a drawing context
        draw = ImageDraw.Draw(img_with_highlights, "RGBA")

        # Filter elements for this page
        page_elements = [
            elem
            for elem in elements
            if elem.page_number == page_number and elem.coordinates
        ]

        for element in page_elements:
            if not element.coordinates:
                continue

            # Get color for this element type
            color = self.ELEMENT_COLORS.get(element.element_type, (128, 128, 128, 100))

            # Extract coordinates - unstructured may have different coordinate formats
            coords = element.coordinates
            if isinstance(coords, dict):
                # Try different coordinate formats
                if (
                    "x1" in coords
                    and "y1" in coords
                    and "x2" in coords
                    and "y2" in coords
                ):
                    x1, y1, x2, y2 = (
                        coords["x1"],
                        coords["y1"],
                        coords["x2"],
                        coords["y2"],
                    )
                elif "points" in coords:
                    # Handle points format
                    points = coords["points"]
                    if len(points) >= 2:
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        x1, y1 = min(x_coords), min(y_coords)
                        x2, y2 = max(x_coords), max(y_coords)
                    else:
                        continue
                else:
                    continue
            elif isinstance(coords, (list, tuple)) and len(coords) >= 4:
                x1, y1, x2, y2 = coords[:4]
            else:
                continue

            # Draw rectangle
            draw.rectangle(
                [x1, y1, x2, y2], fill=color, outline=color[:3] + (255,), width=2
            )

            # Add label
            try:
                font = ImageFont.truetype("Arial.ttf", 12)
            except:
                font = ImageFont.load_default()

            label = element.element_type
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Draw label background
            label_bg = (x1, y1 - text_height - 4, x1 + text_width + 8, y1)
            draw.rectangle(label_bg, fill=color[:3] + (200,))

            # Draw label text
            draw.text(
                (x1 + 4, y1 - text_height - 2), label, fill=(0, 0, 0, 255), font=font
            )

        return img_with_highlights

    def create_highlighted_pdf(
        self,
        original_pdf_path: Union[str, Path],
        elements: List[DocumentElement],
        output_path: Union[str, Path],
    ) -> bool:
        """
        Create a new PDF with highlights showing document segmentation

        Args:
            original_pdf_path: Path to original PDF
            elements: List of segmented elements
            output_path: Path to save highlighted PDF

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert PDF to images
            images = self.convert_pdf_to_images(original_pdf_path)
            if not images:
                return False

            # Create highlighted images
            highlighted_images = []
            for page_num, image in enumerate(images, 1):
                highlighted_img = self.draw_highlights_on_image(
                    image, elements, page_num
                )
                highlighted_images.append(highlighted_img)

            # Save highlighted images as PDF
            if highlighted_images:
                highlighted_images[0].save(
                    str(output_path),
                    "PDF",
                    resolution=200.0,
                    save_all=True,
                    append_images=(
                        highlighted_images[1:] if len(highlighted_images) > 1 else []
                    ),
                )
                return True

            return False

        except Exception as e:
            print(f"Error creating highlighted PDF: {e}")
            return False

    def export_segmentation_data(
        self,
        elements: List[DocumentElement],
        output_path: Union[str, Path],
        format: str = "json",
    ) -> bool:
        """
        Export segmentation data to file

        Args:
            elements: List of document elements
            output_path: Path to save data
            format: Export format ('json', 'csv')

        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)

            if format.lower() == "json":
                data = [asdict(element) for element in elements]
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            elif format.lower() == "csv":
                import pandas as pd

                df = pd.DataFrame([asdict(element) for element in elements])
                df.to_csv(output_path, index=False)

            else:
                raise ValueError(f"Unsupported format: {format}")

            return True

        except Exception as e:
            print(f"Error exporting segmentation data: {e}")
            return False

    def get_statistics(self, elements: List[DocumentElement]) -> Dict[str, Any]:
        """
        Get statistics about the segmented elements

        Args:
            elements: List of document elements

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_elements": len(elements),
            "element_types": {},
            "pages": set(),
            "total_text_length": 0,
        }

        for element in elements:
            # Count element types
            element_type = element.element_type
            stats["element_types"][element_type] = (
                stats["element_types"].get(element_type, 0) + 1
            )

            # Track pages
            if element.page_number:
                stats["pages"].add(element.page_number)

            # Count text length
            if element.text:
                stats["total_text_length"] += len(element.text)

        stats["total_pages"] = len(stats["pages"])
        stats["pages"] = sorted(list(stats["pages"]))

        return stats


def segment_document_unstructured(
    file_path: Union[str, Path], strategy: str = "hi_res", extract_images: bool = True
) -> Tuple[List[DocumentElement], Dict[str, Any]]:
    """
    Convenience function to segment a document using unstructured

    Args:
        file_path: Path to document file
        strategy: Partitioning strategy
        extract_images: Whether to extract image elements

    Returns:
        Tuple of (elements list, statistics dict)
    """
    service = UnstructuredSegmentationService()
    elements = service.segment_document(file_path, strategy, extract_images)
    stats = service.get_statistics(elements)
    return elements, stats


def process_pdf_with_highlights(
    input_pdf_path: Union[str, Path],
    output_pdf_path: Union[str, Path],
    output_data_path: Optional[Union[str, Path]] = None,
    strategy: str = "hi_res",
) -> bool:
    """
    Process a PDF and create highlighted version showing segmentation

    Args:
        input_pdf_path: Path to input PDF
        output_pdf_path: Path to save highlighted PDF
        output_data_path: Optional path to save segmentation data
        strategy: Partitioning strategy

    Returns:
        True if successful, False otherwise
    """
    service = UnstructuredSegmentationService()

    # Segment the document
    print(f"Segmenting document: {input_pdf_path}")
    elements = service.segment_document(input_pdf_path, strategy=strategy)

    if not elements:
        print("No elements found in document")
        return False

    # Get statistics
    stats = service.get_statistics(elements)
    print(
        f"Found {stats['total_elements']} elements across {stats['total_pages']} pages"
    )
    print("Element types:", stats["element_types"])

    # Create highlighted PDF
    print(f"Creating highlighted PDF: {output_pdf_path}")
    success = service.create_highlighted_pdf(input_pdf_path, elements, output_pdf_path)

    if success:
        print(f"Highlighted PDF saved to: {output_pdf_path}")
    else:
        print("Failed to create highlighted PDF")
        return False

    # Export segmentation data if requested
    if output_data_path:
        print(f"Exporting segmentation data: {output_data_path}")
        data_success = service.export_segmentation_data(elements, output_data_path)
        if data_success:
            print(f"Segmentation data saved to: {output_data_path}")
        else:
            print("Failed to export segmentation data")

    return success


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python main.py <input_pdf> <output_pdf> [output_data.json]")
        sys.exit(1)

    input_pdf = sys.argv[1]
    output_pdf = sys.argv[2]
    output_data = sys.argv[3] if len(sys.argv) > 3 else None

    success = process_pdf_with_highlights(
        input_pdf, output_pdf, output_data, strategy="hi_res"
    )

    if success:
        print("Document processing completed successfully!")
    else:
        print("Document processing failed!")
        sys.exit(1)
