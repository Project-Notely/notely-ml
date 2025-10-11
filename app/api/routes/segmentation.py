import io
from typing import Union

from fastapi import APIRouter, File, UploadFile, Form, Body

from app.controllers import segmentation_controller
from app.controllers.segmentation_controller import SegmentInput
from app.models.page_segmentation_models import SegmentationResult

router = APIRouter()


@router.post("/segment-test", response_model=SegmentationResult)
async def segment_document(
    file: UploadFile = File(...),
    query: str = Form(...),
) -> SegmentationResult:
    """Segment a document from a file upload."""
    return await segmentation_controller.segment_document(
        file,
        query,
    )


@router.post("/segment", response_model=SegmentationResult)
async def segment_drawing(
    request: SegmentInput = Body(...),
) -> SegmentationResult:
    """Segment a document from drawing data (SVG)."""
    return await segmentation_controller.segment_document(
        request,
        request.query,
    )
