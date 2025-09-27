import io

from fastapi import APIRouter, File, UploadFile, Form

from app.controllers import segmentation_controller
from app.models.page_segmentation_models import SegmentationResult

router = APIRouter()


@router.post("/segment", response_model=SegmentationResult)
async def segment_document(
    file: UploadFile = File(...),
    query: str = Form(...),
) -> SegmentationResult:
    """
    Handles the API request for document segmentation.

    Args:
        file: The uploaded document image.
        query: The user's natural language query from the form data.

    Returns:
        The segmentation result from the controller.
    """
    return await segmentation_controller.segment_document(
        file,
        query,
    )
