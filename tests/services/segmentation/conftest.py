"""
Pytest configuration and fixtures for segmentation tests
"""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from app.models.segmentation_models import (
    BoundingBox,
    DocumentSegment,
    SegmentationResult,
    SegmentType,
)
from app.services.segmentation_service import SegmentationService


@pytest.fixture
def segmentation_service() -> SegmentationService:
    """
    Create a SegmentationService instance for testing

    Returns:
        SegmentationService: Service instance
    """
    return SegmentationService()


@pytest.fixture
def sample_image() -> Generator[Path, None, None]:
    """
    Create a sample test image

    Yields:
        Path: Path to temporary image file
    """
    # Create a simple test image with text-like patterns
    width, height = 800, 600
    image = Image.new("RGB", (width, height), color="white")

    # Add some colored rectangles to simulate different document elements
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)

    # Title area (top)
    draw.rectangle([50, 50, 750, 100], fill="lightblue", outline="black")
    draw.text((60, 65), "Sample Document Title", fill="black")

    # Paragraph area (middle)
    draw.rectangle([50, 150, 750, 300], fill="lightgray", outline="black")
    draw.text((60, 160), "This is a sample paragraph with multiple lines", fill="black")
    draw.text((60, 180), "of text that would typically be found in a", fill="black")
    draw.text((60, 200), "document that needs to be segmented.", fill="black")

    # Table area (bottom left)
    draw.rectangle([50, 350, 350, 500], fill="lightyellow", outline="black")
    draw.text((60, 360), "Table Header 1 | Header 2", fill="black")
    draw.text((60, 380), "Row 1 Data   | More Data", fill="black")
    draw.text((60, 400), "Row 2 Data   | Even More", fill="black")

    # Image area (bottom right)
    draw.rectangle([400, 350, 700, 500], fill="lightgreen", outline="black")
    draw.text((410, 425), "Image Placeholder", fill="black")

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        image.save(tmp_file.name)
        yield Path(tmp_file.name)

    # Cleanup
    os.unlink(tmp_file.name)


@pytest.fixture
def sample_pdf_content() -> bytes:
    """
    Create sample PDF content for testing

    Returns:
        bytes: PDF content as bytes
    """
    # Create a simple PDF using reportlab
    try:
        import io

        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)

        # Add title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, "Sample PDF Document")

        # Add paragraph
        c.setFont("Helvetica", 12)
        c.drawString(100, 700, "This is a sample paragraph in a PDF document.")
        c.drawString(100, 680, "It contains multiple lines of text that should")
        c.drawString(100, 660, "be properly segmented by the service.")

        # Add table-like structure
        c.drawString(100, 600, "Table Data:")
        c.drawString(100, 580, "Header 1     Header 2     Header 3")
        c.drawString(100, 560, "Data 1       Data 2       Data 3")
        c.drawString(100, 540, "More 1       More 2       More 3")

        c.save()
        buffer.seek(0)
        return buffer.getvalue()

    except ImportError:
        # Fallback: return empty bytes if reportlab not available
        return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"


@pytest.fixture
def sample_text_file() -> Generator[Path, None, None]:
    """
    Create a sample text file for testing

    Yields:
        Path: Path to temporary text file
    """
    content = """Sample Document Title

This is a sample paragraph with multiple lines of text that would typically be found in a document that needs to be segmented.

Another paragraph follows here with different content to test the segmentation capabilities.

• List item 1
• List item 2
• List item 3

Table Header 1 | Table Header 2 | Table Header 3
Data Row 1     | Data Row 2     | Data Row 3
More Data 1    | More Data 2    | More Data 3
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_file:
        tmp_file.write(content)
        tmp_file.flush()
        yield Path(tmp_file.name)

    # Cleanup
    os.unlink(tmp_file.name)


@pytest.fixture
def sample_segments() -> list[DocumentSegment]:
    """
    Create sample document segments for testing

    Returns:
        list[DocumentSegment]: List of sample segments
    """
    return [
        DocumentSegment(
            text="Sample Document Title",
            segment_type=SegmentType.TITLE,
            bbox=BoundingBox(x=50, y=50, width=700, height=50),
            confidence=0.95,
            metadata={"category": "Title", "text_length": 21},
        ),
        DocumentSegment(
            text="This is a sample paragraph with multiple lines of text.",
            segment_type=SegmentType.PARAGRAPH,
            bbox=BoundingBox(x=50, y=150, width=700, height=150),
            confidence=0.90,
            metadata={"category": "NarrativeText", "text_length": 55},
        ),
        DocumentSegment(
            text="Table data with headers and rows",
            segment_type=SegmentType.TABLE,
            bbox=BoundingBox(x=50, y=350, width=300, height=150),
            confidence=0.85,
            metadata={"category": "Table", "text_length": 32},
        ),
        DocumentSegment(
            text="Image placeholder content",
            segment_type=SegmentType.IMAGE,
            bbox=BoundingBox(x=400, y=350, width=300, height=150),
            confidence=0.80,
            metadata={"category": "Image", "text_length": 25},
        ),
    ]


@pytest.fixture
def sample_segmentation_result(sample_segments) -> SegmentationResult:
    """
    Create a sample segmentation result for testing

    Args:
        sample_segments: Sample segments fixture

    Returns:
        SegmentationResult: Sample segmentation result
    """
    stats = {
        "total_segments": len(sample_segments),
        "segment_types": {"title": 1, "paragraph": 1, "table": 1, "image": 1},
        "with_bounding_boxes": 4,
        "average_confidence": 0.875,
    }

    return SegmentationResult(
        success=True,
        segments=sample_segments,
        total_segments=len(sample_segments),
        statistics=stats,
        strategy_used="hi_res",
    )


@pytest.fixture
def mock_unstructured_elements():
    """
    Create mock unstructured elements for testing

    Returns:
        list: List of mock elements
    """

    class MockElement:
        def __init__(
            self, text: str, category: str, metadata: dict[str, Any] | None = None
        ):
            self.text = text
            self.category = category
            self.metadata = metadata or {}

        def __str__(self):
            return self.text

    class MockCoordinates:
        def __init__(self, points):
            self.points = points

    elements = [
        MockElement(
            "Sample Document Title",
            "Title",
            {"coordinates": MockCoordinates([(50, 50), (750, 100)])},
        ),
        MockElement(
            "This is a sample paragraph with multiple lines of text.",
            "NarrativeText",
            {"coordinates": MockCoordinates([(50, 150), (750, 300)])},
        ),
        MockElement(
            "Table data with headers and rows",
            "Table",
            {"coordinates": MockCoordinates([(50, 350), (350, 500)])},
        ),
        MockElement(
            "Image placeholder content",
            "Image",
            {"coordinates": MockCoordinates([(400, 350), (700, 500)])},
        ),
    ]

    return elements


@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """
    Create a temporary directory for testing

    Yields:
        Path: Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


# Performance test fixtures
@pytest.fixture
def large_image() -> Generator[Path, None, None]:
    """
    Create a large test image for performance testing

    Yields:
        Path: Path to large temporary image file
    """
    # Create a large image (2000x1500)
    width, height = 2000, 1500
    image = Image.new("RGB", (width, height), color="white")

    # Add multiple elements across the image
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)

    # Add multiple rectangles to simulate a complex document
    for i in range(10):
        for j in range(8):
            x = 50 + i * 180
            y = 50 + j * 180
            draw.rectangle(
                [x, y, x + 150, y + 150],
                fill=f"rgb({100 + i * 15}, {100 + j * 15}, 200)",
                outline="black",
            )
            draw.text((x + 10, y + 10), f"Element {i},{j}", fill="black")

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        image.save(tmp_file.name)
        yield Path(tmp_file.name)

    # Cleanup
    os.unlink(tmp_file.name)


# Error testing fixtures
@pytest.fixture
def invalid_image_file() -> Generator[Path, None, None]:
    """
    Create an invalid image file for error testing

    Yields:
        Path: Path to invalid image file
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_file.write(b"This is not a valid image file")
        tmp_file.flush()
        yield Path(tmp_file.name)

    # Cleanup
    os.unlink(tmp_file.name)


@pytest.fixture
def empty_file() -> Generator[Path, None, None]:
    """
    Create an empty file for error testing

    Yields:
        Path: Path to empty file
    """
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
        # File is empty
        tmp_file.flush()
        yield Path(tmp_file.name)

    # Cleanup
    os.unlink(tmp_file.name)
