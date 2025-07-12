"""Drawing models for the notely-ml application.

These models represent the structure of drawing data that will be stored and retrieved
from the database, based on the TypeScript interfaces defined in the frontend.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Point(BaseModel):
    """A point in a drawing stroke with coordinates and optional metadata."""

    x: float = Field(..., description="X coordinate of the point")
    y: float = Field(..., description="Y coordinate of the point")
    pressure: float | None = Field(None, description="Pressure applied at this point")
    timestamp: int | None = Field(
        None, description="Unix timestamp when point was created"
    )


class StrokeStyle(BaseModel):
    """Styling information for a drawing stroke."""

    color: str = Field(..., description="Color of the stroke (hex, rgb, etc.)")
    thickness: float = Field(..., gt=0, description="Thickness of the stroke")
    opacity: float = Field(
        ..., ge=0.0, le=1.0, description="Opacity of the stroke (0-1)"
    )
    line_cap: Literal["round", "square", "butt"] = Field(
        ..., description="Line cap style"
    )
    line_join: Literal["round", "bevel", "miter"] = Field(
        ..., description="Line join style"
    )


class Stroke(BaseModel):
    """A single drawing stroke containing points and styling."""

    id: str = Field(..., description="Unique identifier for the stroke")
    points: list[Point] = Field(
        ..., description="List of points that make up the stroke"
    )
    style: StrokeStyle = Field(..., description="Styling information for the stroke")
    timestamp: int = Field(..., description="Unix timestamp when stroke was created")
    completed: bool = Field(
        default=False, description="Whether the stroke is completed"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "stroke_123",
                "points": [
                    {"x": 10.0, "y": 20.0, "pressure": 0.5},
                    {"x": 15.0, "y": 25.0, "pressure": 0.6},
                ],
                "style": {
                    "color": "#000000",
                    "thickness": 2.0,
                    "opacity": 1.0,
                    "line_cap": "round",
                    "line_join": "round",
                },
                "timestamp": 1640995200000,
                "completed": True,
            }
        }
    }


class DrawingDimensions(BaseModel):
    """Dimensions of the drawing canvas."""

    width: float = Field(..., gt=0, description="Width of the drawing canvas")
    height: float = Field(..., gt=0, description="Height of the drawing canvas")


class DrawingMetadata(BaseModel):
    """Metadata about the drawing."""

    created: int = Field(..., description="Unix timestamp when drawing was created")
    modified: int = Field(
        ..., description="Unix timestamp when drawing was last modified"
    )
    version: str = Field(..., description="Version of the drawing format")


class DrawingData(BaseModel):
    """Complete drawing data including strokes, dimensions, and metadata."""

    strokes: list[Stroke] = Field(..., description="List of strokes in the drawing")
    dimensions: DrawingDimensions = Field(..., description="Canvas dimensions")
    metadata: DrawingMetadata = Field(..., description="Drawing metadata")

    model_config = {
        "json_schema_extra": {
            "example": {
                "strokes": [],
                "dimensions": {"width": 800, "height": 600},
                "metadata": {
                    "created": 1640995200000,
                    "modified": 1640995200000,
                    "version": "1.0",
                },
            }
        }
    }


class SaveDrawingRequest(BaseModel):
    """Request model for saving a drawing."""

    drawing: DrawingData = Field(..., description="The drawing data to save")
    user_id: str | None = Field(None, description="ID of the user saving the drawing")
    title: str | None = Field(None, description="Title of the drawing")
    description: str | None = Field(None, description="Description of the drawing")

    model_config = {
        "json_schema_extra": {
            "example": {
                "drawing": {
                    "strokes": [],
                    "dimensions": {"width": 800, "height": 600},
                    "metadata": {
                        "created": 1640995200000,
                        "modified": 1640995200000,
                        "version": "1.0",
                    },
                },
                "user_id": "user_123",
                "title": "My Drawing",
                "description": "A simple drawing",
            }
        }
    }


class SavedDrawing(BaseModel):
    """Model representing a saved drawing with database metadata."""

    id: str = Field(..., description="Unique identifier for the saved drawing")
    drawing: DrawingData = Field(..., description="The drawing data")
    user_id: str | None = Field(
        None, description="ID of the user who saved the drawing"
    )
    title: str | None = Field(None, description="Title of the drawing")
    description: str | None = Field(None, description="Description of the drawing")
    saved_at: datetime = Field(..., description="When the drawing was saved")
    updated_at: datetime = Field(..., description="When the drawing was last updated")

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}
