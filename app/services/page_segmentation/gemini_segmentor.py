import os

from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

from app.core.config import settings

from . import utils


class GeminiSegmentor:
    def __init__(self):
        self.gemini = genai.Client(api_key=settings.GEMINI_API_KEY)

    def execute(self, image: Image.Image) -> list[dict]:
        original_width, original_height = image.size

        resized_image = utils.resize_image(image=image, max_size=640)

        prompt = utils.build_detection_prompt("red text")

        try:
            response = self.gemini.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, resized_image],
                config=types.GenerateContentConfig(
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )

            # Parse response
            bbox_data = utils.parse_gemini_response(response.text)

            # Draw bounding boxes on original image
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)

            for i, item in enumerate(bbox_data):
                if "box_2d" in item:
                    # Gemini returns [ymin, xmin, ymax, xmax] normalized to 0-1000
                    coords = item["box_2d"]
                    if len(coords) == 4:
                        ymin, xmin, ymax, xmax = coords

                        # Convert from normalized coordinates (0-1000) to pixel coordinates
                        x1 = int((xmin / 1000) * original_width)
                        y1 = int((ymin / 1000) * original_height)
                        x2 = int((xmax / 1000) * original_width)
                        y2 = int((ymax / 1000) * original_height)

                        # Draw bounding box
                        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

                        # Add label
                        label = item.get("label", f"Box {i + 1}")
                        try:
                            font = ImageFont.load_default()
                            # Draw background for text visibility
                            text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
                            draw.rectangle(text_bbox, fill="red")
                            draw.text((x1, y1 - 25), label, fill="white", font=font)
                        except:
                            draw.text((x1, y1 - 20), label, fill="red")

                        print(f"Detected: {label} at ({x1}, {y1}, {x2}, {y2})")

            # Save image with bounding boxes
            output_path = "outputs/test.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            draw_image.save(output_path)
            print(f"Saved image with bounding boxes to: {output_path}")
            print(f"Found {len(bbox_data)} objects total")

            return bbox_data

        except Exception as e:
            print(f"Error during processing: {e}")
            print(
                f"Raw response: {response.text if 'response' in locals() else 'No response'}"
            )
            raise e
