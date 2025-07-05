"""
Document segmentation API routes
"""

from fastapi import APIRouter, UploadFile, File, Query
from typing import List

from app.controllers import segmentation_controller
from app.models.page_segmentation_models import SegmentationResult

router = APIRouter()


@router.post("/segment", response_model=SegmentationResult)
async def segment_document(
    file: UploadFile = File(...),
    strategy: str = Query(
        "hi_res", description="Partitioning strategy (fast, hi_res, auto)"
    ),
    extract_images: bool = Query(True, description="Whether to extract images"),
    infer_table_structure: bool = Query(
        True, description="Whether to infer table structure"
    ),
) -> SegmentationResult:
    """
    Segment a document into different element types
    """
    return await segmentation_controller.segment_document(
        file=file,
        strategy=strategy,
        extract_images=extract_images,
        infer_table_structure=infer_table_structure,
    )


@router.get("/supported-formats")
async def get_supported_formats() -> List[str]:
    """
    Get list of supported file formats
    """
    return await segmentation_controller.get_supported_formats()
