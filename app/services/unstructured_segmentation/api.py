"""
FastAPI routes for unstructured document segmentation service
"""

import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .main import UnstructuredSegmentationService, DocumentElement
from .config import SUPPORTED_FORMATS


router = APIRouter(prefix="/unstructured-segmentation", tags=["unstructured-segmentation"])


class SegmentationRequest(BaseModel):
    strategy: str = "hi_res"
    extract_images: bool = True
    include_page_breaks: bool = True


class SegmentationResponse(BaseModel):
    success: bool
    elements: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    message: str


@router.post("/segment", response_model=SegmentationResponse)
async def segment_document(
    file: UploadFile = File(...),
    strategy: str = Form("hi_res"),
    extract_images: bool = Form(True),
    include_page_breaks: bool = Form(True)
):
    """
    Segment a document using unstructured library
    """
    # Validate file format
    file_suffix = Path(file.filename).suffix.lower()
    if file_suffix not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_suffix}. Supported formats: {SUPPORTED_FORMATS}"
        )
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
        temp_path = Path(temp_file.name)
        shutil.copyfileobj(file.file, temp_file)
    
    try:
        # Initialize service and segment document
        service = UnstructuredSegmentationService()
        elements = service.segment_document(
            temp_path, 
            strategy=strategy,
            extract_images=extract_images,
            include_page_breaks=include_page_breaks
        )
        
        # Get statistics
        stats = service.get_statistics(elements)
        
        # Convert elements to dict format
        elements_dict = [
            {
                "element_type": elem.element_type,
                "text": elem.text,
                "coordinates": elem.coordinates,
                "page_number": elem.page_number,
                "confidence": elem.confidence,
                "metadata": elem.metadata
            }
            for elem in elements
        ]
        
        return SegmentationResponse(
            success=True,
            elements=elements_dict,
            statistics=stats,
            message=f"Successfully segmented document with {len(elements)} elements"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()


@router.post("/segment-with-highlights")
async def segment_with_highlights(
    file: UploadFile = File(...),
    strategy: str = Form("hi_res"),
    extract_images: bool = Form(True)
):
    """
    Segment a PDF document and return highlighted version
    """
    # Validate file format (only PDF for highlights)
    file_suffix = Path(file.filename).suffix.lower()
    if file_suffix != '.pdf':
        raise HTTPException(
            status_code=400,
            detail="Highlights are only supported for PDF files"
        )
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as input_temp:
        input_path = Path(input_temp.name)
        shutil.copyfileobj(file.file, input_temp)
    
    output_path = input_path.parent / f"{input_path.stem}_highlighted.pdf"
    
    try:
        # Initialize service
        service = UnstructuredSegmentationService()
        
        # Segment document
        elements = service.segment_document(
            input_path,
            strategy=strategy,
            extract_images=extract_images
        )
        
        if not elements:
            raise HTTPException(status_code=422, detail="No elements found in document")
        
        # Create highlighted PDF
        success = service.create_highlighted_pdf(input_path, elements, output_path)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create highlighted PDF")
        
        # Return the highlighted PDF
        return FileResponse(
            output_path,
            media_type="application/pdf",
            filename=f"{Path(file.filename).stem}_highlighted.pdf"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        for path in [input_path, output_path]:
            if path.exists():
                path.unlink()


@router.get("/element-types")
async def get_supported_element_types():
    """
    Get list of supported element types and their colors
    """
    from .config import ELEMENT_COLORS
    
    return {
        "element_types": list(ELEMENT_COLORS.keys()),
        "color_mapping": {
            element_type: {
                "rgba": color,
                "hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            }
            for element_type, color in ELEMENT_COLORS.items()
        }
    }


@router.get("/supported-formats")
async def get_supported_formats():
    """
    Get list of supported file formats
    """
    return {
        "supported_formats": list(SUPPORTED_FORMATS),
        "description": "List of file formats supported by the unstructured segmentation service"
    } 