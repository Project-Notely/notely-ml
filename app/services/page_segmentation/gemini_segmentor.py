from google import genai
from google.genai import types
from PIL import Image

from app.core.config import settings

from . import utils


class GeminiSegmentor:
    def __init__(self):
        self.gemini = genai.Client(api_key=settings.GEMINI_API_KEY)

    def execute(self, image: Image.Image, query: str) -> list[dict]:
        original_width, original_height = image.size

        resized_image = utils.resize_image(image=image, max_size=640)

        prompt = utils.build_detection_prompt(query)

        try:
            response = self.gemini.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, resized_image],
                config=types.GenerateContentConfig(
                    temperature=0,
                ),
            )

            # parse response
            bbox_data = utils.parse_gemini_response(response.text)

            # convert normalized coordinates to pixel coordinates
            for item in bbox_data:
                if "box_2d" not in item:
                    continue

                coords = item["box_2d"]
                if len(coords) != 4:
                    continue

                ymin, xmin, ymax, xmax = coords

                # convert from normalized coordinates (0-1000) to pixel coordinates
                x1 = int((xmin / 1000) * original_width)
                y1 = int((ymin / 1000) * original_height)
                x2 = int((xmax / 1000) * original_width)
                y2 = int((ymax / 1000) * original_height)

                # store pixel coordinates in the item for later use
                item["pixel_coords"] = {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                }

            # draw debug bounding boxes if DEBUG is enabled
            utils.draw_debug_bounding_boxes(image, bbox_data)

            return bbox_data

        except Exception as e:
            print(f"Error during processing: {e}")
            raw_response = response.text if "response" in locals() else "No response"
            print(f"Raw response: {raw_response}")
            raise e
