"""
Comprehensive tests for the DrawingRepository.

This test suite covers all repository operations including CRUD operations,
error handling, edge cases, and model validation.
"""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
from bson import ObjectId
from pymongo.errors import PyMongoError

from app.models.drawing_models import (
    DrawingData,
    DrawingDimensions,
    DrawingMetadata,
    Point,
    SavedDrawing,
    SaveDrawingRequest,
    Stroke,
    StrokeStyle,
)
from app.repositories.drawing_repository import DrawingRepository


class TestDrawingModels:
    """Test cases for drawing data models."""

    def test_point_model_creation(self):
        """Test Point model creation with valid data."""
        point = Point(x=10.5, y=20.3, pressure=0.5, timestamp=1640995200000)

        assert point.x == 10.5
        assert point.y == 20.3
        assert point.pressure == 0.5
        assert point.timestamp == 1640995200000

    def test_point_model_optional_fields(self):
        """Test Point model with optional fields."""
        point = Point(x=10.5, y=20.3)

        assert point.x == 10.5
        assert point.y == 20.3
        assert point.pressure is None
        assert point.timestamp is None

    def test_stroke_style_model_creation(self):
        """Test StrokeStyle model creation with valid data."""
        style = StrokeStyle(
            color="#FF0000",
            thickness=2.5,
            opacity=0.8,
            line_cap="round",
            line_join="round",
        )

        assert style.color == "#FF0000"
        assert style.thickness == 2.5
        assert style.opacity == 0.8
        assert style.line_cap == "round"
        assert style.line_join == "round"

    def test_stroke_style_validation(self):
        """Test StrokeStyle model validation."""
        # Test invalid thickness
        with pytest.raises(ValueError):
            StrokeStyle(
                color="#FF0000",
                thickness=0,  # Should be > 0
                opacity=0.8,
                line_cap="round",
                line_join="round",
            )

        # Test invalid opacity
        with pytest.raises(ValueError):
            StrokeStyle(
                color="#FF0000",
                thickness=2.5,
                opacity=1.5,  # Should be <= 1.0
                line_cap="round",
                line_join="round",
            )

        # Test invalid line_cap
        with pytest.raises(ValueError):
            StrokeStyle(
                color="#FF0000",
                thickness=2.5,
                opacity=0.8,
                line_cap="invalid",  # Should be one of the allowed values
                line_join="round",
            )

    def test_stroke_model_creation(self):
        """Test Stroke model creation with valid data."""
        points = [Point(x=10.0, y=20.0), Point(x=15.0, y=25.0)]
        style = StrokeStyle(
            color="#000000",
            thickness=2.0,
            opacity=1.0,
            line_cap="round",
            line_join="round",
        )

        stroke = Stroke(
            id="stroke_123",
            points=points,
            style=style,
            timestamp=1640995200000,
            completed=True,
        )

        assert stroke.id == "stroke_123"
        assert len(stroke.points) == 2
        assert stroke.style.color == "#000000"
        assert stroke.timestamp == 1640995200000
        assert stroke.completed is True

    def test_drawing_data_model_creation(self):
        """Test DrawingData model creation with valid data."""
        drawing_data = DrawingData(
            strokes=[],
            dimensions=DrawingDimensions(width=800, height=600),
            metadata=DrawingMetadata(
                created=1640995200000, modified=1640995200000, version="1.0"
            ),
        )

        assert len(drawing_data.strokes) == 0
        assert drawing_data.dimensions.width == 800
        assert drawing_data.dimensions.height == 600
        assert drawing_data.metadata.version == "1.0"

    def test_save_drawing_request_model(self):
        """Test SaveDrawingRequest model creation."""
        drawing_data = DrawingData(
            strokes=[],
            dimensions=DrawingDimensions(width=800, height=600),
            metadata=DrawingMetadata(
                created=1640995200000, modified=1640995200000, version="1.0"
            ),
        )

        request = SaveDrawingRequest(
            drawing=drawing_data,
            user_id="user_123",
            title="Test Drawing",
            description="A test drawing",
        )

        assert request.user_id == "user_123"
        assert request.title == "Test Drawing"
        assert request.description == "A test drawing"
        assert request.drawing.dimensions.width == 800


class TestDrawingRepository:
    """Test cases for the DrawingRepository class."""

    @pytest.fixture
    def mock_mongo_client(self):
        """Create a mock MongoDB client."""
        mock_client = Mock()
        mock_database = Mock()
        mock_collection = Mock()

        # Set up the mock chain: client[db_name][collection_name]
        mock_client.__getitem__ = Mock(return_value=mock_database)
        mock_database.__getitem__ = Mock(return_value=mock_collection)

        # Mock collection methods
        mock_collection.create_index = Mock()
        mock_collection.insert_one = Mock()
        mock_collection.find_one = Mock()
        mock_collection.find = Mock()
        mock_collection.find_one_and_update = Mock()
        mock_collection.delete_one = Mock()
        mock_collection.count_documents = Mock()

        return mock_client

    @pytest.fixture
    def repository(self, mock_mongo_client):
        """Create a DrawingRepository instance with mocked MongoDB."""
        return DrawingRepository(mock_mongo_client, "test_db")

    @pytest.fixture
    def sample_drawing_data(self):
        """Create sample drawing data for testing."""
        return DrawingData(
            strokes=[
                Stroke(
                    id="stroke_1",
                    points=[
                        Point(x=10.0, y=20.0, pressure=0.5),
                        Point(x=15.0, y=25.0, pressure=0.6),
                    ],
                    style=StrokeStyle(
                        color="#000000",
                        thickness=2.0,
                        opacity=1.0,
                        line_cap="round",
                        line_join="round",
                    ),
                    timestamp=1640995200000,
                    completed=True,
                )
            ],
            dimensions=DrawingDimensions(width=800, height=600),
            metadata=DrawingMetadata(
                created=1640995200000, modified=1640995200000, version="1.0"
            ),
        )

    @pytest.fixture
    def sample_save_request(self, sample_drawing_data):
        """Create sample save request for testing."""
        return SaveDrawingRequest(
            drawing=sample_drawing_data,
            user_id="user_123",
            title="Test Drawing",
            description="A test drawing",
        )

    def test_repository_initialization(self, mock_mongo_client):
        """Test repository initialization."""
        repo = DrawingRepository(mock_mongo_client, "test_db")

        assert repo.client == mock_mongo_client
        assert repo.database == mock_mongo_client["test_db"]
        assert repo.collection == mock_mongo_client["test_db"]["drawings"]

    def test_index_creation(self, mock_mongo_client):
        """Test that database indexes are created during initialization."""
        repo = DrawingRepository(mock_mongo_client, "test_db")
        collection = repo.collection

        # Verify that create_index was called for expected indexes
        assert collection.create_index.called
        call_args = [call[0][0] for call in collection.create_index.call_args_list]

        assert "user_id" in call_args
        assert "saved_at" in call_args
        assert [("user_id", 1), ("saved_at", -1)] in call_args

    def test_index_creation_failure(self, mock_mongo_client):
        """Test repository initialization continues with index creation failures."""
        mock_mongo_client["test_db"][
            "drawings"
        ].create_index.side_effect = PyMongoError("Index creation failed")

        # Should not raise an exception
        repo = DrawingRepository(mock_mongo_client, "test_db")
        assert repo is not None

    @pytest.mark.asyncio
    async def test_save_drawing_success(self, repository, sample_save_request):
        """Test successful drawing save operation."""
        # Mock successful insert
        mock_result = Mock()
        mock_result.inserted_id = ObjectId()
        repository.collection.insert_one.return_value = mock_result

        # Execute save operation
        result = await repository.save_drawing(sample_save_request)

        # Verify result
        assert isinstance(result, SavedDrawing)
        assert result.id == str(mock_result.inserted_id)
        assert result.user_id == "user_123"
        assert result.title == "Test Drawing"
        assert result.description == "A test drawing"
        assert result.drawing == sample_save_request.drawing
        assert isinstance(result.saved_at, datetime)
        assert isinstance(result.updated_at, datetime)

        # Verify insert_one was called
        repository.collection.insert_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_drawing_failure(self, repository, sample_save_request):
        """Test drawing save operation failure."""
        # Mock failed insert
        mock_result = Mock()
        mock_result.inserted_id = None
        repository.collection.insert_one.return_value = mock_result

        # Execute save operation and expect RuntimeError
        with pytest.raises(RuntimeError, match="Failed to save drawing to database"):
            await repository.save_drawing(sample_save_request)

    @pytest.mark.asyncio
    async def test_save_drawing_exception(self, repository, sample_save_request):
        """Test drawing save operation with database exception."""
        # Mock database exception
        repository.collection.insert_one.side_effect = PyMongoError("Database error")

        # Execute save operation and expect RuntimeError
        with pytest.raises(RuntimeError, match="Failed to save drawing"):
            await repository.save_drawing(sample_save_request)

    @pytest.mark.asyncio
    async def test_get_drawing_by_id_success(self, repository, sample_drawing_data):
        """Test successful drawing retrieval by ID."""
        # Mock successful find
        drawing_id = str(ObjectId())
        mock_document = {
            "_id": ObjectId(drawing_id),
            "drawing": sample_drawing_data.model_dump(),
            "user_id": "user_123",
            "title": "Test Drawing",
            "description": "A test drawing",
            "saved_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
        repository.collection.find_one.return_value = mock_document

        # Execute get operation
        result = await repository.get_drawing_by_id(drawing_id)

        # Verify result
        assert isinstance(result, SavedDrawing)
        assert result.id == drawing_id
        assert result.user_id == "user_123"
        assert result.title == "Test Drawing"

        # Verify find_one was called with correct parameters
        repository.collection.find_one.assert_called_once_with(
            {"_id": ObjectId(drawing_id)}
        )

    @pytest.mark.asyncio
    async def test_get_drawing_by_id_not_found(self, repository):
        """Test drawing retrieval when drawing doesn't exist."""
        # Mock not found
        repository.collection.find_one.return_value = None

        # Execute get operation
        result = await repository.get_drawing_by_id(str(ObjectId()))

        # Verify result is None
        assert result is None

    @pytest.mark.asyncio
    async def test_get_drawing_by_id_invalid_id(self, repository):
        """Test drawing retrieval with invalid ID format."""
        # Execute get operation with invalid ID
        with pytest.raises(ValueError, match="Invalid drawing ID format"):
            await repository.get_drawing_by_id("invalid_id")

    @pytest.mark.asyncio
    async def test_get_drawings_by_user_success(self, repository, sample_drawing_data):
        """Test successful user drawings retrieval."""
        # Mock successful find
        mock_documents = [
            {
                "_id": ObjectId(),
                "drawing": sample_drawing_data.model_dump(),
                "user_id": "user_123",
                "title": f"Drawing {i}",
                "description": f"Description {i}",
                "saved_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
            }
            for i in range(3)
        ]

        mock_cursor = Mock()
        mock_cursor.__iter__ = Mock(return_value=iter(mock_documents))
        repository.collection.find.return_value = mock_cursor

        # Mock cursor methods
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.skip.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor

        # Execute get operation
        result = await repository.get_drawings_by_user("user_123", limit=10, offset=0)

        # Verify result
        assert len(result) == 3
        assert all(isinstance(drawing, SavedDrawing) for drawing in result)
        assert all(drawing.user_id == "user_123" for drawing in result)

        # Verify find was called with correct parameters
        repository.collection.find.assert_called_once_with({"user_id": "user_123"})

    @pytest.mark.asyncio
    async def test_get_drawings_by_user_invalid_limit(self, repository):
        """Test user drawings retrieval with invalid limit."""
        # Test limit too small
        with pytest.raises(ValueError, match="Limit must be between 1 and 100"):
            await repository.get_drawings_by_user("user_123", limit=0)

        # Test limit too large
        with pytest.raises(ValueError, match="Limit must be between 1 and 100"):
            await repository.get_drawings_by_user("user_123", limit=101)

    @pytest.mark.asyncio
    async def test_get_drawings_by_user_invalid_offset(self, repository):
        """Test user drawings retrieval with invalid offset."""
        with pytest.raises(ValueError, match="Offset must be non-negative"):
            await repository.get_drawings_by_user("user_123", offset=-1)

    @pytest.mark.asyncio
    async def test_update_drawing_success(
        self, repository, sample_save_request, sample_drawing_data
    ):
        """Test successful drawing update."""
        # Mock successful update
        drawing_id = str(ObjectId())
        mock_document = {
            "_id": ObjectId(drawing_id),
            "drawing": sample_drawing_data.model_dump(),
            "user_id": "user_123",
            "title": "Updated Drawing",
            "description": "Updated description",
            "saved_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
        repository.collection.find_one_and_update.return_value = mock_document

        # Execute update operation
        result = await repository.update_drawing(drawing_id, sample_save_request)

        # Verify result
        assert isinstance(result, SavedDrawing)
        assert result.id == drawing_id

        # Verify find_one_and_update was called
        repository.collection.find_one_and_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_drawing_not_found(self, repository, sample_save_request):
        """Test drawing update when drawing doesn't exist."""
        # Mock not found
        repository.collection.find_one_and_update.return_value = None

        # Execute update operation
        result = await repository.update_drawing(str(ObjectId()), sample_save_request)

        # Verify result is None
        assert result is None

    @pytest.mark.asyncio
    async def test_update_drawing_invalid_id(self, repository, sample_save_request):
        """Test drawing update with invalid ID format."""
        with pytest.raises(ValueError, match="Invalid drawing ID format"):
            await repository.update_drawing("invalid_id", sample_save_request)

    @pytest.mark.asyncio
    async def test_delete_drawing_success(self, repository):
        """Test successful drawing deletion."""
        # Mock successful delete
        mock_result = Mock()
        mock_result.deleted_count = 1
        repository.collection.delete_one.return_value = mock_result

        # Execute delete operation
        result = await repository.delete_drawing(str(ObjectId()))

        # Verify result
        assert result is True

        # Verify delete_one was called
        repository.collection.delete_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_drawing_not_found(self, repository):
        """Test drawing deletion when drawing doesn't exist."""
        # Mock not found
        mock_result = Mock()
        mock_result.deleted_count = 0
        repository.collection.delete_one.return_value = mock_result

        # Execute delete operation
        result = await repository.delete_drawing(str(ObjectId()))

        # Verify result
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_drawing_invalid_id(self, repository):
        """Test drawing deletion with invalid ID format."""
        with pytest.raises(ValueError, match="Invalid drawing ID format"):
            await repository.delete_drawing("invalid_id")

    @pytest.mark.asyncio
    async def test_count_user_drawings_success(self, repository):
        """Test successful user drawings count."""
        # Mock successful count
        repository.collection.count_documents.return_value = 5

        # Execute count operation
        result = await repository.count_user_drawings("user_123")

        # Verify result
        assert result == 5

        # Verify count_documents was called
        repository.collection.count_documents.assert_called_once_with(
            {"user_id": "user_123"}
        )

    @pytest.mark.asyncio
    async def test_count_user_drawings_exception(self, repository):
        """Test user drawings count with database exception."""
        # Mock database exception
        repository.collection.count_documents.side_effect = PyMongoError(
            "Database error"
        )

        # Execute count operation and expect RuntimeError
        with pytest.raises(RuntimeError, match="Failed to count user drawings"):
            await repository.count_user_drawings("user_123")

    def test_document_to_saved_drawing_conversion(
        self, repository, sample_drawing_data
    ):
        """Test conversion from MongoDB document to SavedDrawing model."""
        # Create mock document
        drawing_id = ObjectId()
        mock_document = {
            "_id": drawing_id,
            "drawing": sample_drawing_data.model_dump(),
            "user_id": "user_123",
            "title": "Test Drawing",
            "description": "A test drawing",
            "saved_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }

        # Execute conversion
        result = repository._document_to_saved_drawing(mock_document)

        # Verify result
        assert isinstance(result, SavedDrawing)
        assert result.id == str(drawing_id)
        assert result.user_id == "user_123"
        assert result.title == "Test Drawing"
        assert result.description == "A test drawing"
        assert isinstance(result.drawing, DrawingData)


@pytest.mark.integration
class TestDrawingRepositoryIntegration:
    """Integration tests for DrawingRepository with actual MongoDB."""

    @pytest.fixture
    def mongo_client(self):
        """Create a real MongoDB client for integration tests."""
        # Note: This would require a real MongoDB instance
        # For now, we'll skip these tests unless MongoDB is available
        pytest.skip("Integration tests require MongoDB instance")

    @pytest.fixture
    def integration_repository(self, mongo_client):
        """Create repository with real MongoDB connection."""
        return DrawingRepository(mongo_client, "test_notely_db")

    @pytest.mark.asyncio
    async def test_full_crud_cycle(self, integration_repository, sample_save_request):
        """Test complete CRUD cycle with real database."""
        # This would test the full cycle: create -> read -> update -> delete
        # Implementation would depend on having a real MongoDB instance
        pass
