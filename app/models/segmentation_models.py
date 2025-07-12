"""
Models for document segmentation
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SegmentType(str, Enum):
    """Types of document segments"""

    TITLE = "title"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST_ITEM = "list_item"
    IMAGE = "image"
    FIGURE = "figure"
    HEADER = "header"
    FOOTER = "footer"
    CAPTION = "caption"
    TEXT = "text"


class BoundingBox(BaseModel):
    """Bounding box coordinates"""

    x: int = Field(..., description="X coordinate")
    y: int = Field(..., description="Y coordinate")
    width: int = Field(..., description="Width of bounding box")
    height: int = Field(..., description="Height of bounding box")


class DocumentSegment(BaseModel):
    """A single document segment"""

    text: str = Field(..., description="Text content of the segment")
    segment_type: SegmentType = Field(..., description="Type of segment")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box coordinates")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class SegmentationResult(BaseModel):
    """Complete segmentation result"""

    success: bool = Field(..., description="Whether segmentation was successful")
    segments: List[DocumentSegment] = Field(
        ..., description="List of detected segments"
    )
    total_segments: int = Field(default=0, description="Total number of segments")
    statistics: Dict[str, Any] = Field(
        default_factory=dict, description="Segmentation statistics"
    )
    strategy_used: Optional[str] = Field(None, description="Partitioning strategy used")
    error: Optional[str] = Field(None, description="Error message if failed")
