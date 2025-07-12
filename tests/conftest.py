"""
Shared pytest fixtures for the notely-ml test suite.

This file contains common fixtures that can be used across multiple test modules.
"""

import asyncio
from datetime import datetime

import pytest
from bson import ObjectId

from app.models.drawing_models import (
    DrawingData,
    DrawingDimensions,
    DrawingMetadata,
    Point,
    SaveDrawingRequest,
    Stroke,
    StrokeStyle,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_point():
    """Create a sample Point for testing."""
    return Point(x=10.5, y=20.3, pressure=0.75, timestamp=1640995200000)


@pytest.fixture
def sample_stroke_style():
    """Create a sample StrokeStyle for testing."""
    return StrokeStyle(
        color="#FF0000", thickness=3.0, opacity=0.9, line_cap="round", line_join="round"
    )


@pytest.fixture
def sample_stroke(sample_stroke_style):
    """Create a sample Stroke for testing."""
    return Stroke(
        id="test_stroke_123",
        points=[
            Point(x=0.0, y=0.0, pressure=0.5),
            Point(x=10.0, y=10.0, pressure=0.7),
            Point(x=20.0, y=5.0, pressure=0.6),
        ],
        style=sample_stroke_style,
        timestamp=1640995200000,
        completed=True,
    )


@pytest.fixture
def sample_drawing_metadata():
    """Create sample DrawingMetadata for testing."""
    return DrawingMetadata(created=1640995200000, modified=1640995200000, version="1.0")


@pytest.fixture
def sample_drawing_dimensions():
    """Create sample DrawingDimensions for testing."""
    return DrawingDimensions(width=1920, height=1080)


@pytest.fixture
def empty_drawing_data(sample_drawing_dimensions, sample_drawing_metadata):
    """Create empty DrawingData for testing."""
    return DrawingData(
        strokes=[],
        dimensions=sample_drawing_dimensions,
        metadata=sample_drawing_metadata,
    )


@pytest.fixture
def drawing_data_with_strokes(
    sample_stroke, sample_drawing_dimensions, sample_drawing_metadata
):
    """Create DrawingData with strokes for testing."""
    return DrawingData(
        strokes=[sample_stroke],
        dimensions=sample_drawing_dimensions,
        metadata=sample_drawing_metadata,
    )


@pytest.fixture
def basic_save_request(empty_drawing_data):
    """Create a basic SaveDrawingRequest for testing."""
    return SaveDrawingRequest(
        drawing=empty_drawing_data,
        user_id="test_user_123",
        title="Test Drawing",
        description="A drawing for testing purposes",
    )


@pytest.fixture
def anonymous_save_request(empty_drawing_data):
    """Create an anonymous SaveDrawingRequest (no user_id) for testing."""
    return SaveDrawingRequest(
        drawing=empty_drawing_data,
        user_id=None,
        title="Anonymous Drawing",
        description="An anonymous drawing",
    )


@pytest.fixture
def mock_object_id():
    """Create a mock ObjectId for testing."""
    return ObjectId()


@pytest.fixture
def mock_datetime():
    """Create a mock datetime for testing."""
    return datetime(2024, 1, 1, 12, 0, 0)


@pytest.fixture
def multiple_test_users():
    """Create multiple test user IDs for testing."""
    return ["user_1", "user_2", "user_3", "user_4", "user_5"]


@pytest.fixture
def performance_test_data():
    """Create data for performance testing."""
    # Create multiple strokes for performance testing
    strokes = []
    for i in range(100):  # 100 strokes
        points = []
        for j in range(50):  # 50 points per stroke
            points.append(
                Point(
                    x=float(j * 2),
                    y=float(i * 2 + j),
                    pressure=0.5 + (j % 10) * 0.05,
                    timestamp=1640995200000 + j,
                )
            )

        stroke = Stroke(
            id=f"perf_stroke_{i}",
            points=points,
            style=StrokeStyle(
                color=f"#{'%06x' % (i * 1000)}",
                thickness=1.0 + (i % 5),
                opacity=0.8,
                line_cap="round",
                line_join="round",
            ),
            timestamp=1640995200000 + i * 1000,
            completed=True,
        )
        strokes.append(stroke)

    return DrawingData(
        strokes=strokes,
        dimensions=DrawingDimensions(width=2560, height=1440),
        metadata=DrawingMetadata(
            created=1640995200000, modified=1640995200000, version="1.0"
        ),
    )


@pytest.fixture
def invalid_data_samples():
    """Create various invalid data samples for testing validation."""
    return {
        "invalid_stroke_style": {
            "color": "#FF0000",
            "thickness": -1,  # Invalid: should be > 0
            "opacity": 1.5,  # Invalid: should be <= 1.0
            "line_cap": "invalid",  # Invalid: not in allowed values
            "line_join": "round",
        },
        "invalid_drawing_dimensions": {
            "width": -100,  # Invalid: should be > 0
            "height": 0,  # Invalid: should be > 0
        },
        "invalid_object_id": "not_a_valid_object_id",
        "invalid_user_limits": {
            "limit_too_small": 0,
            "limit_too_large": 101,
            "negative_offset": -1,
        },
    }


# Performance testing fixtures
@pytest.fixture
def benchmark_drawing_data():
    """Create drawing data for benchmarking tests."""
    # Create a large drawing with many strokes for performance testing
    strokes = []
    for i in range(1000):
        points = [Point(x=float(j), y=float(i), pressure=0.5) for j in range(10)]
        stroke = Stroke(
            id=f"benchmark_stroke_{i}",
            points=points,
            style=StrokeStyle(
                color="#000000",
                thickness=2.0,
                opacity=1.0,
                line_cap="round",
                line_join="round",
            ),
            timestamp=1640995200000 + i,
            completed=True,
        )
        strokes.append(stroke)

    return DrawingData(
        strokes=strokes,
        dimensions=DrawingDimensions(width=4096, height=2160),
        metadata=DrawingMetadata(
            created=1640995200000, modified=1640995200000, version="1.0"
        ),
    )
