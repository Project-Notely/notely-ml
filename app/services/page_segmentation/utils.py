import io
import json
import os

import cairosvg
from PIL import Image, ImageDraw, ImageFont

from app.core.config import settings


def resize_image(image: Image.Image, max_size: int = 640) -> Image.Image:
    """Resize image while maintaining aspect ratio."""
    width, height = image.size
    scale = min(max_size / width, max_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def svg_to_pil_image(svg_content: str, width: float, height: float) -> Image.Image:
    """Convert SVG content to PIL Image with white background.

    Args:
        svg_content: The SVG content as a string.
        width: The width of the SVG canvas.
        height: The height of the SVG canvas.

    Returns:
        PIL Image object created from the SVG with white background.

    Raises:
        ValueError: If the SVG content is invalid or conversion fails.
    """
    try:
        # Convert SVG to PNG bytes using cairosvg
        png_bytes = cairosvg.svg2png(
            bytestring=svg_content.encode("utf-8"),
            output_width=int(width),
            output_height=int(height),
        )

        # Convert PNG bytes to PIL Image
        image = Image.open(io.BytesIO(png_bytes))

        # Create a white background image
        white_background = Image.new("RGB", image.size, (255, 255, 255))

        # If the image has an alpha channel, composite it onto the white background
        if image.mode in ("RGBA", "LA"):
            white_background.paste(
                image, mask=image.split()[-1]
            )  # Use alpha channel as mask
            return white_background
        else:
            # If no alpha channel, just paste the image directly
            white_background.paste(image)
            return white_background

    except Exception as e:
        raise ValueError(f"Failed to convert SVG to PIL Image: {str(e)}") from e


def parse_gemini_response(response_text: str) -> list[dict]:
    """Parse Gemini response, extracting JSON from markdown if needed."""
    try:
        # Handle JSON in markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                json_str = response_text[start:end].strip()
            else:
                # Fallback if closing ``` not found
                json_str = response_text[start:].strip()
        else:
            json_str = response_text.strip()

        # Parse JSON
        parsed_data = json.loads(json_str)

        # Ensure it's a list
        if isinstance(parsed_data, dict):
            # Sometimes Gemini returns a single object instead of a list
            parsed_data = [parsed_data]

        return parsed_data

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        fallback = json_str if "json_str" in locals() else response_text
        print(f"Attempting to parse: {fallback}")

        # Try to extract any JSON-like structures
        import re

        json_pattern = r"\[.*?\]"
        matches = re.findall(json_pattern, response_text, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return []


def build_detection_prompt(search_term: str) -> str:
    return (
        f"# Detect {search_term} in this screenshot with high accuracy.\n\n"
        "## Instructions:\n"
        f"1. Return up to 10 detections of {search_term} "
        "with highest confidence scores\n"
        "2. If no detections are found, return an empty list []\n"
        "## Output format:\n"
        "- Return a JSON list where each entry contains:\n"
        "- 'box_2d': bounding box coordinates [ymin, xmin, ymax, xmax] "
        "normalized to 0-1000\n"
        f"## Example output for successful detection of {search_term}:\n"
        "[\n"
        "  {\n"
        '    "box_2d": [100, 200, 300, 600],\n'
        "  },\n"
        "  {\n"
        '    "box_2d": [50, 400, 150, 800],\n'
        "  }\n"
        "]\n\n"
        "## Example output for no detection:\n"
        "[]"
    )


def draw_debug_bounding_boxes(
    image: Image.Image, bbox_data: list[dict], output_path: str = "outputs/test.png"
) -> None:
    """Draw bounding boxes on the image for debugging purposes.

    Args:
        image: The original image to draw on.
        bbox_data: List of bounding box data with pixel_coords already calculated.
        output_path: Path to save the debug image.
    """

    # draw bounding boxes on original image
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    for i, item in enumerate(bbox_data):
        if "pixel_coords" not in item:
            continue

        coords = item["pixel_coords"]
        x1, y1 = coords["x"], coords["y"]
        x2, y2 = x1 + coords["width"], y1 + coords["height"]

        # draw bounding box
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

        # add label
        label = item.get("label", f"Box {i + 1}")
        try:
            font = ImageFont.load_default()
            # draw background for text visibility
            text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
            draw.rectangle(text_bbox, fill="red")
            draw.text((x1, y1 - 25), label, fill="white", font=font)
        except Exception:
            draw.text((x1, y1 - 20), label, fill="red")

        print(f"Detected: {label} at ({x1}, {y1}, {x2}, {y2})")

    # save image with bounding boxes
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    draw_image.save(output_path)
    print(f"Saved image with bounding boxes to: {output_path}")
    print(f"Found {len(bbox_data)} objects total")
