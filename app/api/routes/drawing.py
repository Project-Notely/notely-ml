import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()


class DrawingCreate(BaseModel):
    svg: str
    width: float
    height: float


@router.post("/drawings")
async def create_drawing(drawing: DrawingCreate):
    """
    Receives a drawing from the iOS app and saves it as an SVG file.
    """
    logger.info(
        f"Received drawing with width: {drawing.width} and height: {drawing.height}"
    )

    # Define the path to save the SVG file
    # Ensure the data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_path = os.path.join(data_dir, "drawing.svg")

    try:
        with open(file_path, "w") as f:
            f.write(drawing.svg)
        logger.info(f"Drawing saved to {file_path}")
        return {
            "message": "Drawing received and saved successfully.",
            "file_path": file_path,
        }
    except IOError as e:
        logger.error(f"Error saving drawing: {e}")
        raise HTTPException(status_code=500, detail="Failed to save drawing.")
