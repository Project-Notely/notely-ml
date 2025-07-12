"""Drawing repository for managing drawing data persistence.

This repository provides CRUD operations for drawing data using MongoDB as the backend.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import PyMongoError

if TYPE_CHECKING:
    from pymongo.collection import Collection
    from pymongo.database import Database

from app.models.drawing_models import DrawingData, SavedDrawing, SaveDrawingRequest


class DrawingRepository:
    """Repository for managing drawing data persistence."""

    def __init__(self, mongo_client: MongoClient, database_name: str = "notely_db"):
        """Initialize the drawing repository.

        Args:
            mongo_client: MongoDB client instance
            database_name: Name of the database to use
        """
        self.client = mongo_client
        self.database: Database = self.client[database_name]
        self.collection: Collection = self.database["drawings"]

        # Create indexes for better query performance
        self._create_indexes()

    def _create_indexes(self) -> None:
        """Create database indexes for optimized queries."""
        try:
            # Index on user_id for user-specific queries
            self.collection.create_index("user_id")

            # Index on saved_at for chronological queries
            self.collection.create_index("saved_at")

            # Compound index for user + time-based queries
            self.collection.create_index([("user_id", 1), ("saved_at", -1)])

        except PyMongoError as e:
            # Log the error but don't fail initialization
            print(f"Warning: Failed to create indexes: {e}")

    async def save_drawing(self, request: SaveDrawingRequest) -> SavedDrawing:
        """Save a new drawing to the database.

        Args:
            request: The drawing save request containing drawing data and metadata

        Returns:
            SavedDrawing: The saved drawing with database metadata

        Raises:
            ValueError: If drawing data is invalid
            RuntimeError: If database operation fails
        """
        try:
            current_time = datetime.now(UTC)

            # Prepare document for MongoDB
            document = {
                "_id": ObjectId(),
                "drawing": request.drawing.model_dump(),
                "user_id": request.user_id,
                "title": request.title,
                "description": request.description,
                "saved_at": current_time,
                "updated_at": current_time,
            }

            # Insert the document
            result = self.collection.insert_one(document)

            if not result.inserted_id:
                raise RuntimeError("Failed to save drawing to database")

            # Return the saved drawing
            return SavedDrawing(
                id=str(result.inserted_id),
                drawing=request.drawing,
                user_id=request.user_id,
                title=request.title,
                description=request.description,
                saved_at=current_time,
                updated_at=current_time,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to save drawing: {e!s}") from e

    async def get_drawing_by_id(self, drawing_id: str) -> SavedDrawing | None:
        """Retrieve a drawing by its ID.

        Args:
            drawing_id: The unique identifier of the drawing

        Returns:
            SavedDrawing: The drawing if found, None otherwise

        Raises:
            ValueError: If drawing_id is invalid
            RuntimeError: If database operation fails
        """
        try:
            # Validate ObjectId format
            if not ObjectId.is_valid(drawing_id):
                raise ValueError(f"Invalid drawing ID format: {drawing_id}")

            # Query the database
            document = self.collection.find_one({"_id": ObjectId(drawing_id)})

            if not document:
                return None

            # Convert to SavedDrawing model
            return self._document_to_saved_drawing(document)

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve drawing: {e!s}") from e

    async def get_drawings_by_user(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> list[SavedDrawing]:
        """Retrieve drawings for a specific user.

        Args:
            user_id: The user's identifier
            limit: Maximum number of drawings to return
            offset: Number of drawings to skip

        Returns:
            List[SavedDrawing]: List of user's drawings

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If database operation fails
        """
        try:
            if limit <= 0 or limit > 100:
                raise ValueError("Limit must be between 1 and 100")

            if offset < 0:
                raise ValueError("Offset must be non-negative")

            # Query the database
            cursor = (
                self.collection.find({"user_id": user_id})
                .sort("saved_at", -1)
                .skip(offset)
                .limit(limit)
            )

            # Convert documents to SavedDrawing models
            drawings = []
            for document in cursor:
                drawings.append(self._document_to_saved_drawing(document))

            return drawings

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve user drawings: {e!s}") from e

    async def update_drawing(
        self, drawing_id: str, request: SaveDrawingRequest
    ) -> SavedDrawing | None:
        """Update an existing drawing.

        Args:
            drawing_id: The unique identifier of the drawing to update
            request: The updated drawing data

        Returns:
            SavedDrawing: The updated drawing if found, None otherwise

        Raises:
            ValueError: If drawing_id is invalid
            RuntimeError: If database operation fails
        """
        try:
            # Validate ObjectId format
            if not ObjectId.is_valid(drawing_id):
                raise ValueError(f"Invalid drawing ID format: {drawing_id}")

            # Prepare update document
            update_doc = {
                "$set": {
                    "drawing": request.drawing.model_dump(),
                    "title": request.title,
                    "description": request.description,
                    "updated_at": datetime.now(UTC),
                }
            }

            # Update the document
            result = self.collection.find_one_and_update(
                {"_id": ObjectId(drawing_id)}, update_doc, return_document=True
            )

            if not result:
                return None

            # Return the updated drawing
            return self._document_to_saved_drawing(result)

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to update drawing: {e!s}") from e

    async def delete_drawing(self, drawing_id: str) -> bool:
        """Delete a drawing by its ID.

        Args:
            drawing_id: The unique identifier of the drawing to delete

        Returns:
            bool: True if drawing was deleted, False if not found

        Raises:
            ValueError: If drawing_id is invalid
            RuntimeError: If database operation fails
        """
        try:
            # Validate ObjectId format
            if not ObjectId.is_valid(drawing_id):
                raise ValueError(f"Invalid drawing ID format: {drawing_id}")

            # Delete the document
            result = self.collection.delete_one({"_id": ObjectId(drawing_id)})

            return result.deleted_count > 0

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to delete drawing: {e!s}") from e

    async def count_user_drawings(self, user_id: str) -> int:
        """Count the number of drawings for a specific user.

        Args:
            user_id: The user's identifier

        Returns:
            int: Number of drawings for the user

        Raises:
            RuntimeError: If database operation fails
        """
        try:
            return self.collection.count_documents({"user_id": user_id})
        except Exception as e:
            raise RuntimeError(f"Failed to count user drawings: {e!s}") from e

    def _document_to_saved_drawing(self, document: dict[str, Any]) -> SavedDrawing:
        """Convert a MongoDB document to a SavedDrawing model.

        Args:
            document: The MongoDB document

        Returns:
            SavedDrawing: The converted model
        """
        return SavedDrawing(
            id=str(document["_id"]),
            drawing=DrawingData(**document["drawing"]),
            user_id=document.get("user_id"),
            title=document.get("title"),
            description=document.get("description"),
            saved_at=document["saved_at"],
            updated_at=document["updated_at"],
        )
