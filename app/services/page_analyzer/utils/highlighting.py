from typing import Tuple, Optional
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from app.services.page_analyzer.models.models import TextBox, OCRResult
from app.services.page_analyzer.utils.search import search_text_in_image


def get_highlight_color(confidence: float) -> Tuple[int, int, int]:
    """Get highlight color based on confidence level (BGR format)"""
    if confidence >= 80:
        return (0, 255, 0)  # Green - high confidence
    elif confidence >= 60:
        return (0, 255, 255)  # Yellow - medium confidence
    elif confidence >= 40:
        return (0, 165, 255)  # Orange - low confidence
    else:
        return (0, 0, 255)  # Red - very low confidence


def create_highlighted_image(
    image: np.ndarray,
    text_boxes: list[TextBox],
    highlight_color: Tuple[int, int, int] = (0, 255, 0),
    highlight_opacity: float = 0.3,
    border_colour: Tuple[int, int, int] = (0, 255, 0),
    border_thickness: int = 2,
    show_text: bool = True,
    show_confidence: bool = True,
) -> np.ndarray:
    """Create highlighted image with detected text boxes

    Args:
        image: input image (BGR format)
        text_boxes: list of detected text boxes
        highlight_colour: RGB colour for highlighting
        highlight_opacity: opacity of the highlight overlay
        border_colour: RGB colour for borders
        border_thickness: int = 2,
        show_text: whether to display text labels
        show_confidence; whether to show confidence scores

    Returns:
        Highlighted image
    """
    if len(image.shape) == 3:
        result = image.copy()
    else:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # create overlay for transparency
    overlay = result.copy()

    for text_box in text_boxes:
        x, y, w, h = text_box.bbox

        # draw filled rectangle on overlay
        cv2.rectangle(overlay, (x, y), (x + w, y + h), highlight_color, -1)

        # draw border
        cv2.rectangle(result, (x, y), (x + w, y + h), border_colour, border_thickness)

        # add text label if requested
        if show_text:
            label = text_box.text
            if show_confidence:
                label += f" ({text_box.confidence:.1f}%)"

            # calculate text position
            font_scale = 0.5
            font_thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            # position text above the box if possible, otherwise inside
            text_x = x
            text_y = y - 5 if y > text_height + 10 else y + h - 5

            # draw text background
            cv2.rectangle(
                result,
                (text_x - 2, text_y - text_height - 2),
                (text_x + text_width + 2, text_y + baseline + 2),
                (0, 0, 0),
                -1,
            )

            # draw text
            cv2.putText(
                result,
                label,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
            )

    # blend overlay with result
    result = cv2.addWeighted(
        result, 1 - highlight_opacity, overlay, highlight_opacity, 0
    )
    
    return result


def save_highlighted_image(
    image: np.ndarray,
    output_path: str,
    quality: int = 100,
) -> bool:
    """Save highlighted image to file

    Args:
        image: input image (BGR format)
        output_path: path to save the image
        quality: quality of the image (0-100)

    Returns:
        True if successful, False otherwise
    """
    try:
        # ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # save image
        if output_path.lower().endswith(".jpg") or output_path.lower().endswith(
            ".jpeg"
        ):
            cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(output_path, image)

        return True
    except Exception as e:
        print(f"Error saving highlighted image: {e}")
        return False


def create_text_summary(ocr_result: OCRResult) -> str:
    """Create a text summary of the OCR results

    Args:
        ocr_result: OCR processing result

    Returns:
        Formatted sumary string
    """
    summary = []
    summary.append("=== OCR RESULTS SUMMARY ===")
    summary.append(f"Full Text: {ocr_result.full_text}")
    summary.append(f"Total Words: {ocr_result.total_words}")
    summary.append(f"Average Confidence: {ocr_result.average_confidence:.1f}%")
    summary.append("\n=== WORD DETAILS ===")

    for i, text_box in enumerate(ocr_result.text_boxes, 1):
        x, y, w, h = text_box.bbox
        summary.append(
            f"{i:2d}. '{text_box.text}' | "
            f"Confidence: {text_box.confidence:.1f}% | "
            f"Position: ({x}, {y}) | "
            f"Size: {w}x{h}"
        )

    return "\n".join(summary)
