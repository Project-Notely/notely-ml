import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor as HFTrOCRProcessor, VisionEncoderDecoderModel

from typing import Tuple

from app.services.page_analyzer.models.models import (
    TextBox,
    OCRResult,
    ProcessingResult,
)


class TrOCRProcessor:
    """Simple TrOCR processor for text recognition"""

    MODELS = {
        "handwritten_small": "microsoft/trocr-small-handwritten",
        "handwritten_base": "microsoft/trocr-base-handwritten",
        "handwritten_large": "microsoft/trocr-large-handwritten",
        "printed_base": "microsoft/trocr-base-printed",
        "printed_large": "microsoft/trocr-large-printed",
    }

    def __init__(self, model_type: str = "printed_base", device: str = "cpu"):
        self.model_type = model_type
        self.device = device
        self.processor = None
        self.model = None
        self.initialized = False

    def initialize(self) -> bool:
        """Initialize the TrOCR model"""
        try:
            if self.model_type not in self.MODELS:
                print(
                    f"Unknown model type: {self.model_type}. Available: {list(self.MODELS.keys())}"
                )
                return False

            model_name = self.MODELS[self.model_type]
            print(f"Loading TrOCR model: {model_name}")

            self.processor = HFTrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            self.initialized = True
            return True

        except Exception as e:
            print(f"Failed to initialize TrOCR: {e}")
            return False

    def process_text_region(self, image: np.ndarray) -> ProcessingResult:
        """Process entire image and detect individual words"""
        if not self.initialized:
            return ProcessingResult(success=False, error="Model not initialized")

        try:
            # Split image into horizontal strips for better line processing
            text_lines = self._split_into_lines(image)

            all_text_boxes = []
            all_texts = []
            total_confidence = 0

            for line_info in text_lines:
                line_image, line_y, line_height = line_info

                # Process this line with TrOCR
                line_result = self._process_single_line(line_image)

                if line_result and line_result.strip():
                    # Create word boxes for this line
                    words = line_result.split()
                    if words:
                        # Estimate word positions within the line
                        line_width = line_image.shape[1]
                        word_boxes = self._estimate_word_positions(
                            words, line_y, line_height, line_width
                        )

                        for word, bbox in word_boxes:
                            confidence = 70.0  # Default confidence
                            text_box = TextBox(
                                text=word, confidence=confidence, bbox=bbox
                            )
                            all_text_boxes.append(text_box)
                            all_texts.append(word)
                            total_confidence += confidence

            # Combine results
            if all_text_boxes:
                full_text = " ".join(all_texts)
                avg_confidence = total_confidence / len(all_text_boxes)
                word_count = len(all_text_boxes)
            else:
                full_text = ""
                avg_confidence = 0.0
                word_count = 0

            result = OCRResult(
                full_text=full_text,
                average_confidence=avg_confidence,
                total_words=word_count,
                text_boxes=all_text_boxes,
            )

            return ProcessingResult(success=True, result=result)

        except Exception as e:
            return ProcessingResult(success=False, error=str(e))

    def _split_into_lines(self, image: np.ndarray) -> list[Tuple[np.ndarray, int, int]]:
        """Split image into horizontal text lines"""
        # Convert to grayscale for line detection
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )

        # Simple approach: split image into horizontal strips
        height, width = gray.shape
        line_height = 50  # Approximate line height in pixels

        lines = []
        for y in range(0, height, line_height):
            end_y = min(y + line_height, height)
            if end_y - y > 20:  # Only process if strip is tall enough
                line_image = image[y:end_y, :]
                lines.append((line_image, y, end_y - y))

        return lines

    def _process_single_line(self, line_image: np.ndarray) -> str:
        """Process a single line of text with TrOCR"""
        try:
            # Convert to PIL Image
            if len(line_image.shape) == 3:
                image_rgb = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = line_image

            pil_image = Image.fromarray(image_rgb)

            # Process with TrOCR
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values, max_length=256, num_beams=4
                )

            # Decode text
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[
                0
            ]
            return text.strip()

        except Exception as e:
            print(f"Error processing line: {e}")
            return ""

    def _estimate_word_positions(
        self, words: list[str], line_y: int, line_height: int, line_width: int
    ) -> list[Tuple[str, list[int]]]:
        """Estimate bounding boxes for individual words"""
        word_boxes = []

        if not words:
            return word_boxes

        # Simple estimation: divide line width by number of words
        padding = 10
        available_width = line_width - 2 * padding

        total_chars = sum(len(word) for word in words) + len(words) - 1  # +1 for spaces

        x_offset = padding
        for i, word in enumerate(words):
            # Estimate word width based on character count
            word_width = int((len(word) / total_chars) * available_width)

            # Ensure minimum width
            word_width = max(word_width, 30)

            bbox = [x_offset, line_y, word_width, line_height]
            word_boxes.append((word, bbox))

            # Move to next word position
            x_offset += word_width + 10  # 10px space between words

        return word_boxes

    def cleanup(self):
        """Clean up resources"""
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.initialized = False
