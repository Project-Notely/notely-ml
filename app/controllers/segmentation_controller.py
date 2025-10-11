import io
import os
import uuid
from typing import Union

from fastapi import HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from app.models.page_segmentation_models import (
    BoundingBox,
    DocumentSegment,
    SegmentationResult,
    SegmentType,
)
from app.services.page_segmentation.gemini_segmentor import GeminiSegmentor
from app.services.page_segmentation import utils
from app.services.query_parser.query_parser import QueryParser
import logging
from app.core.config import settings


class SegmentInput(BaseModel):
    svg: str
    width: float
    height: float
    query: str


async def segment_document(
    input_data: Union[UploadFile, SegmentInput], query: str
) -> SegmentationResult:
    logging.info(f"Segmenting document with query: {query}")
    try:
        # parse natural language query
        parser = QueryParser()
        processed_query = await parser.execute(user_query=query)

        if settings.DEBUG:
            logging.info(f"Processed query: {processed_query}")
            data_dir = "data"
            os.makedirs(data_dir, exist_ok=True)
            svg_path = os.path.join(data_dir, f"drawing.svg")
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(input_data.svg)
            logging.info(f"Saved SVG to: {svg_path}")

        # convert SVG to PIL image
        image = utils.svg_to_pil_image(
            svg_content=input_data.svg, width=input_data.width, height=input_data.height
        )

        if settings.DEBUG:
            logging.info(f"Converted image to: {image.size}, mode: {image.mode}")
            converted_image_path = os.path.join(data_dir, f"converted.png")
            image.save(converted_image_path)
            logging.info(f"Saved converted image to: {converted_image_path}")
            logging.info(f"Image size: {image.size}, mode: {image.mode}")

        # segment the document using the processed query
        # todo remove ========================================================
        image = Image.open("data/notes.png")
        # todo remove ========================================================
        segmentor = GeminiSegmentor()
        bbox_data: list[dict] = segmentor.execute(
            image=image, query="red text"
        )

        # transform bbox_data into DocumentSegment objects
        segments = []
        for item in bbox_data:
            if "pixel_coords" not in item:
                continue

            coords = item["pixel_coords"]
            segment = DocumentSegment(
                text=item.get("label", "N/A"),
                segment_type=SegmentType.TEXT,  # default type
                bbox=BoundingBox(**coords),
            )
            segments.append(segment)

        logging.info(f"Segmented {len(segments)} segments")
        return SegmentationResult(
            success=True,
            segments=segments,
            total_segments=len(segments),
            strategy_used="gemini-2.5-flash",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process request: {e}"
        ) from e
