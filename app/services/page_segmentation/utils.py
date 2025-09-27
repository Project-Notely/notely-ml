import json

from PIL import Image
from google.genai import types


def resize_image(image: Image.Image, max_size: int = 640) -> Image.Image:
    """Resize image while maintaining aspect ratio."""
    width, height = image.size
    scale = min(max_size / width, max_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


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
        print(
            f"Attempting to parse: {json_str if 'json_str' in locals() else response_text}"
        )

        # Try to extract any JSON-like structures
        import re

        json_pattern = r"\[.*?\]"
        matches = re.findall(json_pattern, response_text, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)
            except:
                continue

        return []


def build_detection_prompt(search_term: str) -> str:
    return (
        f"# Detect {search_term} in this screenshot with high accuracy.\n\n"
        "## Instructions:\n"
        f"1. Return up to 10 detections of {search_term} with highest confidence scores\n"
        "2. If no detections are found, return an empty list []\n"
        "## Output format:\n"
        "- Return a JSON list where each entry contains:\n"
        "- 'box_2d': bounding box coordinates [ymin, xmin, ymax, xmax] normalized to 0-1000\n"
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


def get_segmentation_tool_schema() -> types.Tool:
    """Define the function calling schema for document segmentation."""
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="extract_segments",
                description="Extracts specified segments from a document image based on a user's query.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(
                            type=types.Type.STRING,
                            description="The user's natural language query specifying what to extract. For example: 'the main title', 'all the paragraphs and the chart on the left'.",
                        )
                    },
                    required=["query"],
                ),
            )
        ]
    )
