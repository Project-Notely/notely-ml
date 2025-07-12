import difflib
import os
from typing import Literal

import cv2
import easyocr
import google.generativeai as genai
import numpy as np
import pytesseract
from dotenv import load_dotenv
from PIL import Image

from app.services.page_analyzer.engines.trocr_processor import TrOCRProcessor
from app.services.page_analyzer.models.models import (
    OCRResult,
    ProcessingResult,
    TextBox,
    WordMatch,
)

try:
    import paddleocr

    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

load_dotenv(dotenv_path=".env.local")

OCRBackend = Literal["easyocr", "trocr", "tesseract", "paddleocr"]


class GeminiProcessor:
    def __init__(self, ocr_backend: OCRBackend = "easyocr"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.ocr_backend = ocr_backend
        self.easyocr_reader = None
        self.trocr_processor = None
        self.paddleocr_reader = None
        self.initialized = False

    def initialize(self, **kwargs) -> bool:
        try:
            if self.ocr_backend == "easyocr":
                self.easyocr_reader = easyocr.Reader(
                    ["en"], gpu=kwargs.get("gpu", False), verbose=False
                )
            elif self.ocr_backend == "trocr":
                self.trocr_processor = TrOCRProcessor(
                    model_type=kwargs.get("trocr_model_type", "handwritten_base"),
                    device=kwargs.get("device", "cpu"),
                )
                if not self.trocr_processor.initialize():
                    print("❌ Failed to initialize TrOCR processor")
                    return False
            elif self.ocr_backend == "tesseract":
                # Test if tesseract is available
                try:
                    pytesseract.get_tesseract_version()
                except pytesseract.TesseractNotFoundError:
                    print("❌ Tesseract not found. Please install tesseract-ocr")
                    return False
            elif self.ocr_backend == "paddleocr":
                if not PADDLEOCR_AVAILABLE:
                    print(
                        "❌ PaddleOCR not available. "
                        "Install with: pip install paddleocr"
                    )
                    return False
                self.paddleocr_reader = paddleocr.PaddleOCR(
                    use_angle_cls=True, lang="en"
                )

            self.initialized = True
            print(
                f"✅ Initialized GeminiProcessor with {self.ocr_backend.upper()} "
                f"backend"
            )
            return True
        except Exception as e:
            print(f"❌ Error initializing GeminiProcessor: {e}")
            return False

    def switch_ocr_backend(self, new_backend: OCRBackend, **kwargs) -> bool:
        """Switch to a different OCR backend."""
        print(f"🔄 Switching from {self.ocr_backend} to {new_backend}")
        self.ocr_backend = new_backend
        self.initialized = False
        return self.initialize(**kwargs)

    def process_text_region(self, image: Image.Image) -> ProcessingResult:
        if not self.initialized:
            return ProcessingResult(success=False, error="Model not initialized")

        try:
            # Step 1: extract text from image using Gemini (ALWAYS TRUSTED)
            gemini_text = self._extract_text_with_gemini(image)
            if not gemini_text:
                return ProcessingResult(
                    success=False, error="No text extracted from image"
                )

            print(
                f"📝 Gemini extracted: '{gemini_text[:100]}"
                f"{'...' if len(gemini_text) > 100 else ''}'"
            )

            # Step 2: convert PIL to numpy/OpenCV format
            image_np = np.array(image)

            # Handle different image formats properly
            if len(image_np.shape) == 3:
                if image_np.shape[2] == 4:  # RGBA
                    # Convert RGBA to RGB first, then to BGR
                    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
                    image_cv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                elif image_np.shape[2] == 3:  # RGB
                    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                else:
                    image_cv = image_np
            else:
                image_cv = image_np

            # Step 3: get word positions using selected OCR backend
            word_positions = self._get_word_positions_with_backend(image_cv, image)

            # Step 4: match Gemini text with OCR positions
            matched_words = self._match_gemini_with_positions(
                gemini_text, word_positions
            )

            # Step 5: create result objects
            text_boxes = []
            for match in matched_words:
                text_box = TextBox(
                    text=match.gemini_word,  # Use the original Gemini word
                    confidence=match.confidence,
                    bbox=match.bbox,
                )
                text_boxes.append(text_box)

            # Calculate stats
            avg_confidence = (
                sum(match.confidence for match in matched_words) / len(matched_words)
                if matched_words
                else 0
            )

            ocr_result = OCRResult(
                full_text=gemini_text,
                average_confidence=avg_confidence,
                total_words=len(matched_words),
                text_boxes=text_boxes,
            )

            print(
                f"🎯 Successfully processed {len(matched_words)} words with "
                f"{self.ocr_backend.upper()}"
            )
            return ProcessingResult(success=True, result=ocr_result)

        except Exception as e:
            print(f"❌ Error in process_text_region: {e}")
            return ProcessingResult(success=False, error=str(e))

    ####### BACKEND-SPECIFIC WORD POSITIONING #######

    def _get_word_positions_with_backend(
        self, image_cv: np.ndarray, image_pil: Image.Image
    ) -> list[dict]:
        """Get word positions using the selected OCR backend."""
        if self.ocr_backend == "easyocr":
            return self._get_word_positions_easyocr(image_cv)
        elif self.ocr_backend == "trocr":
            return self._get_word_positions_trocr(image_cv)
        elif self.ocr_backend == "tesseract":
            return self._get_word_positions_tesseract(image_cv)
        elif self.ocr_backend == "paddleocr":
            return self._get_word_positions_paddleocr(image_cv)
        else:
            print(f"❌ Unknown OCR backend: {self.ocr_backend}")
            return []

    def _get_word_positions_easyocr(self, image_cv: np.ndarray) -> list[dict]:
        """Get word positions using EasyOCR (original implementation)."""
        try:
            # Get detections from EasyOCR
            results = self.easyocr_reader.readtext(
                image_cv,
                detail=1,
                paragraph=False,
                width_ths=0.7,
                height_ths=0.7,
            )

            # Split lines into individual words
            word_detections = []

            for _i, result in enumerate(results):
                try:
                    # Handle different EasyOCR return formats
                    if len(result) == 3:
                        bbox_points, text, confidence = result
                    elif len(result) == 2:
                        bbox_points, text = result
                        confidence = 0.8  # Default confidence
                    else:
                        continue

                    # Get bounding box
                    x_coords = [point[0] for point in bbox_points]
                    y_coords = [point[1] for point in bbox_points]

                    bbox_x = int(min(x_coords))
                    bbox_y = int(min(y_coords))
                    bbox_w = int(max(x_coords) - min(x_coords))
                    bbox_h = int(max(y_coords) - min(y_coords))

                    # Check if this is a line (very wide) or already a word (narrow)
                    if bbox_w > 200:  # If wider than 200px, treat as line and split
                        words = text.strip().split()
                        if words:
                            # Extract the line region from image for analysis
                            line_region = image_cv[
                                bbox_y : bbox_y + bbox_h, bbox_x : bbox_x + bbox_w
                            ]

                            # Use morphological analysis to find actual word boundaries
                            word_positions = self._find_actual_word_boundaries(
                                line_region, words, bbox_x, bbox_y, confidence
                            )
                            word_detections.extend(word_positions)
                    else:
                        # Use as individual word
                        word_detections.append(
                            {
                                "text": text.strip(),
                                "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                                "confidence": confidence * 100,
                                "center_x": bbox_x + bbox_w // 2,
                                "center_y": bbox_y + bbox_h // 2,
                            }
                        )

                except Exception:
                    continue

            # Sort by reading order
            word_detections.sort(key=lambda d: (d["center_y"] // 20, d["center_x"]))

            return word_detections

        except Exception as e:
            print(f"❌ Error in _get_word_positions_easyocr: {e}")
            return []

    def _get_word_positions_trocr(self, image_cv: np.ndarray) -> list[dict]:
        """Get word positions using TrOCR (great for handwriting)."""
        try:
            # Use TrOCR's built-in word detection
            result = self.trocr_processor.process_text_region(image_cv)

            if not result.success:
                print(f"❌ TrOCR processing failed: {result.error}")
                return []

            word_detections = []
            for text_box in result.result.text_boxes:
                word_detections.append(
                    {
                        "text": text_box.text,
                        "bbox": text_box.bbox,
                        "confidence": text_box.confidence,
                        "center_x": text_box.bbox[0] + text_box.bbox[2] // 2,
                        "center_y": text_box.bbox[1] + text_box.bbox[3] // 2,
                    }
                )

            print(f"🔤 TrOCR detected {len(word_detections)} words")
            return word_detections

        except Exception as e:
            print(f"❌ Error in _get_word_positions_trocr: {e}")
            return []

    def _get_word_positions_tesseract(self, image_cv: np.ndarray) -> list[dict]:
        """Get word positions using Tesseract (classic, reliable)."""
        try:
            # Convert BGR to RGB for Tesseract
            if len(image_cv.shape) == 3:
                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_cv

            # Get word-level data from Tesseract
            data = pytesseract.image_to_data(
                image_rgb,
                output_type=pytesseract.Output.DICT,
                config="--psm 6",  # Uniform block of text
            )

            word_detections = []
            for i in range(len(data["text"])):
                text = data["text"][i].strip()
                if (
                    text and int(data["conf"][i]) > 0
                ):  # Filter out empty and low-confidence
                    x = data["left"][i]
                    y = data["top"][i]
                    w = data["width"][i]
                    h = data["height"][i]
                    conf = int(data["conf"][i])

                    word_detections.append(
                        {
                            "text": text,
                            "bbox": [x, y, w, h],
                            "confidence": conf,
                            "center_x": x + w // 2,
                            "center_y": y + h // 2,
                        }
                    )

            # Sort by reading order
            word_detections.sort(key=lambda d: (d["center_y"] // 20, d["center_x"]))

            print(f"🔤 Tesseract detected {len(word_detections)} words")
            return word_detections

        except Exception as e:
            print(f"❌ Error in _get_word_positions_tesseract: {e}")
            return []

    def _get_word_positions_paddleocr(self, image_cv: np.ndarray) -> list[dict]:
        """Get word positions using PaddleOCR (excellent for handwriting, latest tech)."""
        try:
            # PaddleOCR expects RGB format
            if len(image_cv.shape) == 3:
                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2RGB)

            # Run PaddleOCR
            results = self.paddleocr_reader.ocr(image_rgb, cls=True)

            word_detections = []
            if results and results[0]:
                for line in results[0]:
                    if len(line) >= 2:
                        bbox_points = line[0]
                        text_info = line[1]
                        text = (
                            text_info[0]
                            if isinstance(text_info, list | tuple)
                            else str(text_info)
                        )
                        confidence = text_info[1] if len(text_info) > 1 else 0.8

                        # Convert bbox points to rectangle
                        x_coords = [point[0] for point in bbox_points]
                        y_coords = [point[1] for point in bbox_points]

                        bbox_x = int(min(x_coords))
                        bbox_y = int(min(y_coords))
                        bbox_w = int(max(x_coords) - min(x_coords))
                        bbox_h = int(max(y_coords) - min(y_coords))

                        # Split multi-word lines
                        words = text.strip().split()
                        if len(words) > 1 and bbox_w > 100:  # Multi-word line
                            # Estimate word positions within the line
                            total_chars = sum(len(word) for word in words)
                            x_offset = 0

                            for word in words:
                                word_width = int((len(word) / total_chars) * bbox_w)
                                word_detections.append(
                                    {
                                        "text": word,
                                        "bbox": [
                                            bbox_x + x_offset,
                                            bbox_y,
                                            word_width,
                                            bbox_h,
                                        ],
                                        "confidence": confidence * 100,
                                        "center_x": bbox_x + x_offset + word_width // 2,
                                        "center_y": bbox_y + bbox_h // 2,
                                    }
                                )
                                x_offset += word_width + 10  # Add spacing
                        else:
                            # Single word
                            word_detections.append(
                                {
                                    "text": text,
                                    "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                                    "confidence": confidence * 100,
                                    "center_x": bbox_x + bbox_w // 2,
                                    "center_y": bbox_y + bbox_h // 2,
                                }
                            )

            # Sort by reading order
            word_detections.sort(key=lambda d: (d["center_y"] // 20, d["center_x"]))

            print(f"🔤 PaddleOCR detected {len(word_detections)} words")
            return word_detections

        except Exception as e:
            print(f"❌ Error in _get_word_positions_paddleocr: {e}")
            return []

    ####### EXISTING HELPER FUNCTIONS (keep all existing methods) #######

    def _extract_text_with_gemini(self, image: Image.Image) -> str:
        try:
            response = self.model.generate_content(
                [
                    """Extract ALL text from this image. Requirements:
                    - Preserve exact spacing and line breaks
                    - Include all words, numbers, and punctuation
                    - Maintain original text order (left-to-right, top-to-bottom)
                    - Don't add explanations, just return the extracted text""",
                    image,
                ]
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error extracting text with Gemini: {e}")
            return ""

    def _find_actual_word_boundaries(
        self,
        line_region: np.ndarray,
        words: list[str],
        line_x: int,
        line_y: int,
        line_confidence: float,
    ) -> list[dict]:
        """Use image analysis to find actual word boundaries within a text line."""
        try:
            # Convert to grayscale if needed
            if len(line_region.shape) == 3:
                gray = cv2.cvtColor(line_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = line_region.copy()

            # Binary threshold to get text pixels
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # Create morphological kernel to connect characters within words but not
            # between words
            kernel_width = max(
                2, min(6, line_region.shape[1] // 200)
            )  # Much smaller, adaptive kernel size
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))

            # Close operation to connect characters within words
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Find contours (word regions)
            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Filter and sort contours by x-coordinate
            word_contours = []

            for _i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)

                # Filter out noise (too small regions)
                if w > 8 and h > 5:  # Minimum word size
                    word_contours.append([x, y, w, h])

            # Sort by x-coordinate (left to right)
            word_contours.sort(key=lambda box: box[0])

            # Match words to contours
            word_detections = []

            if len(word_contours) == len(words):
                # Perfect match - use detected contours directly
                for _i, (word, contour_box) in enumerate(
                    zip(words, word_contours, strict=False)
                ):
                    rel_x, rel_y, w, h = contour_box
                    abs_x = line_x + rel_x
                    abs_y = line_y + rel_y

                    word_detections.append(
                        {
                            "text": word,
                            "bbox": [abs_x, abs_y, w, h],
                            "confidence": line_confidence * 100,
                            "center_x": abs_x + w // 2,
                            "center_y": abs_y + h // 2,
                        }
                    )

            elif len(word_contours) > len(words):
                # More contours than words - merge nearby contours
                merged_contours = self._merge_contours_to_words(
                    word_contours, len(words)
                )

                for _i, (word, contour_box) in enumerate(
                    zip(words, merged_contours, strict=False)
                ):
                    rel_x, rel_y, w, h = contour_box
                    abs_x = line_x + rel_x
                    abs_y = line_y + rel_y

                    word_detections.append(
                        {
                            "text": word,
                            "bbox": [abs_x, abs_y, w, h],
                            "confidence": line_confidence * 100,
                            "center_x": abs_x + w // 2,
                            "center_y": abs_y + h // 2,
                        }
                    )

            else:
                # Fewer contours than words - split larger contours
                word_detections = self._split_contours_to_words(
                    word_contours, words, line_x, line_y, line_confidence
                )

            return word_detections

        except Exception as e:
            print(f"❌ Error in _find_actual_word_boundaries: {e}")
            # Fallback to old method
            return self._split_line_into_words(
                words,
                line_x,
                line_y,
                line_region.shape[1],
                line_region.shape[0],
                line_confidence,
            )

    def _merge_contours_to_words(
        self, contours: list[list[int]], target_count: int
    ) -> list[list[int]]:
        """Merge nearby contours to match the target word count."""
        if len(contours) <= target_count:
            return contours

        merged = contours.copy()

        while len(merged) > target_count:
            # Find the pair with smallest gap
            min_gap = float("inf")
            merge_idx = 0

            for i in range(len(merged) - 1):
                gap = merged[i + 1][0] - (
                    merged[i][0] + merged[i][2]
                )  # distance between contours
                if gap < min_gap:
                    min_gap = gap
                    merge_idx = i

            # Merge the pair
            box1 = merged[merge_idx]
            box2 = merged[merge_idx + 1]

            merged_x = box1[0]
            merged_y = min(box1[1], box2[1])
            merged_w = (box2[0] + box2[2]) - box1[0]
            merged_h = max(box1[1] + box1[3], box2[1] + box2[3]) - merged_y

            merged[merge_idx] = [merged_x, merged_y, merged_w, merged_h]
            merged.pop(merge_idx + 1)

        return merged

    def _split_contours_to_words(
        self,
        contours: list[list[int]],
        words: list[str],
        line_x: int,
        line_y: int,
        line_confidence: float,
    ) -> list[dict]:
        """Split larger contours when we have fewer contours than words."""
        word_detections = []

        if not contours:
            # No contours found, fall back to estimation
            return self._split_line_into_words(
                words, line_x, line_y, 500, 30, line_confidence
            )

        # Calculate how many words per contour on average
        words_per_contour = len(words) / len(contours)

        word_idx = 0
        for contour_box in contours:
            rel_x, rel_y, w, h = contour_box

            # Determine how many words this contour should contain
            words_in_this_contour = round(words_per_contour)
            remaining_words = len(words) - word_idx
            remaining_contours = len(contours) - contours.index(contour_box)

            if remaining_contours == 1:
                words_in_this_contour = remaining_words

            # Split this contour among the words
            for i in range(words_in_this_contour):
                if word_idx >= len(words):
                    break

                word = words[word_idx]

                # Calculate position within contour
                word_width = w // words_in_this_contour
                word_x = rel_x + (i * word_width)

                abs_x = line_x + word_x
                abs_y = line_y + rel_y

                word_detections.append(
                    {
                        "text": word,
                        "bbox": [abs_x, abs_y, word_width, h],
                        "confidence": line_confidence * 100,
                        "center_x": abs_x + word_width // 2,
                        "center_y": abs_y + h // 2,
                    }
                )

                word_idx += 1

        return word_detections

    def _split_line_into_words(
        self,
        words: list[str],
        line_x: int,
        line_y: int,
        line_w: int,
        line_h: int,
        line_confidence: float,
    ) -> list[dict]:
        """Split a detected line into individual word bounding boxes."""
        word_detections = []

        if not words:
            return word_detections

        # Calculate total character count for proportional spacing
        total_chars = sum(len(word) for word in words)
        total_spaces = len(words) - 1  # Spaces between words

        # Estimate character width (leaving some margin)
        usable_width = line_w * 0.95  # Use 95% of line width to account for margins
        char_width = (
            usable_width / (total_chars + total_spaces * 0.5) if total_chars > 0 else 12
        )

        current_x = line_x + (line_w * 0.025)  # Start with 2.5% margin

        for word in words:
            # Calculate word width based on character count
            word_width = len(word) * char_width

            # Adjust for character types (simple heuristic)
            wide_chars = sum(1 for c in word if c in "mwMW@")
            narrow_chars = sum(1 for c in word if c in "iIlj")

            if wide_chars > narrow_chars:
                word_width *= 1.1
            elif narrow_chars > wide_chars:
                word_width *= 0.9

            # Ensure minimum width
            word_width = max(word_width, 8)

            # Create word detection
            word_detection = {
                "text": word,
                "bbox": [int(current_x), line_y, int(word_width), line_h],
                "confidence": line_confidence * 100,  # Convert to 0-100
                "center_x": int(current_x + word_width // 2),
                "center_y": line_y + line_h // 2,
            }

            word_detections.append(word_detection)

            # Move to next word position (add word width + space)
            current_x += word_width + (
                char_width * 0.5
            )  # Half-character space between words

        return word_detections

    def _match_gemini_with_positions(
        self, gemini_text: str, ocr_detections: list[dict]
    ) -> list[WordMatch]:
        """Match detected text with positions."""
        if not ocr_detections:
            return []

        gemini_words = gemini_text.split()
        matches = []

        # Method 1: direct matching when counts are similar
        if abs(len(gemini_words) - len(ocr_detections)) <= 2:
            matches = self._direct_word_matching(gemini_words, ocr_detections)
        # Method 2: fuzzy matching for complex cases
        else:
            matches = self._fuzzy_word_matching(gemini_words, ocr_detections)

        return matches

    def _direct_word_matching(
        self, gemini_words: list[str], ocr_detections: list[dict]
    ) -> list[WordMatch]:
        matches = []

        for i, word in enumerate(gemini_words):
            if i >= len(ocr_detections):
                continue
            detection = ocr_detections[i]
            similarity = self._calculate_similarity(word, detection["text"])

            match = WordMatch(
                word=detection["text"],
                bbox=detection["bbox"],
                confidence=detection["confidence"],
                gemini_word=word,
                similarity=similarity,
            )
            matches.append(match)

        return matches

    def _fuzzy_word_matching(
        self, gemini_words: list[str], ocr_detections: list[dict]
    ) -> list[WordMatch]:
        matches = []
        used_detections = set()

        for word in gemini_words:
            best_match = None
            best_similarity = 0
            best_idx = -1

            for i, detection in enumerate(ocr_detections):
                if i in used_detections:
                    continue

                similarity = self._calculate_similarity(word, detection["text"])

                if similarity > best_similarity and similarity > 0.6:
                    best_similarity = similarity
                    best_match = detection
                    best_idx = i

            if best_match:
                used_detections.add(best_idx)
                match = WordMatch(
                    word=best_match["text"],
                    bbox=best_match["bbox"],
                    confidence=best_match["confidence"] * best_similarity,
                    gemini_word=word,
                    similarity=best_similarity,
                )
                matches.append(match)

        return matches

    def _fill_missing_positions(
        self,
        gemini_words: list[str],
        current_matches: list[WordMatch],
        ocr_detections: list[dict],
    ) -> list[WordMatch]:
        if not current_matches:
            return current_matches

        # Calculate average word dimensions
        avg_width = sum(match.bbox[2] for match in current_matches) // len(
            current_matches
        )
        avg_height = sum(match.bbox[3] for match in current_matches) // len(
            current_matches
        )

        # Find missing words
        matched_gemini_words = {match.gemini_word for match in current_matches}
        missing_words = [
            word for word in gemini_words if word not in matched_gemini_words
        ]

        all_matches = current_matches.copy()

        # Estimate positions for missing words
        for word in missing_words:
            # Find best insertion point based on text order
            word_index = gemini_words.index(word)

            # Estimate position based on neighboring matches
            estimated_bbox = self._estimate_word_position(
                word, word_index, all_matches, gemini_words, avg_width, avg_height
            )

            if estimated_bbox:
                estimated_match = WordMatch(
                    word=word,
                    bbox=estimated_bbox,
                    confidence=0.5,  # Lower confidence for estimated positions
                    gemini_word=word,
                    similarity=1.0,
                )
                all_matches.append(estimated_match)

        # Sort matches by original text order
        word_order = {word: i for i, word in enumerate(gemini_words)}
        all_matches.sort(key=lambda m: word_order.get(m.gemini_word, 999))

        return all_matches

    def _estimate_word_position(
        self,
        word: str,
        word_index: int,
        current_matches: list[WordMatch],
        gemini_words: list[str],
        avg_width: int,
        avg_height: int,
    ) -> list[int] | None:
        """Estimate position for a missing word."""
        if not current_matches:
            return [10, 10, len(word) * 12, avg_height]  # Default position

        # Find nearest matched words
        before_matches = [
            m for m in current_matches if gemini_words.index(m.gemini_word) < word_index
        ]
        after_matches = [
            m for m in current_matches if gemini_words.index(m.gemini_word) > word_index
        ]

        if before_matches and after_matches:
            # Interpolate between before and after
            before_match = before_matches[-1]
            after_match = after_matches[0]

            # Calculate estimated position
            before_right = before_match.bbox[0] + before_match.bbox[2]
            after_left = after_match.bbox[0]

            estimated_x = (before_right + after_left) // 2 - (len(word) * 6)
            estimated_y = (before_match.bbox[1] + after_match.bbox[1]) // 2
            estimated_width = max(len(word) * 12, 20)

            return [estimated_x, estimated_y, estimated_width, avg_height]

        elif before_matches:
            # Extend after the last match
            last_match = before_matches[-1]
            estimated_x = last_match.bbox[0] + last_match.bbox[2] + 10
            estimated_y = last_match.bbox[1]
            return [estimated_x, estimated_y, len(word) * 12, avg_height]

        elif after_matches:
            # Place before the first match
            first_match = after_matches[0]
            estimated_x = max(0, first_match.bbox[0] - len(word) * 12 - 10)
            estimated_y = first_match.bbox[1]
            return [estimated_x, estimated_y, len(word) * 12, avg_height]

        return None

    def _calculate_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words."""
        if not word1 or not word2:
            return 0.0

        # Use difflib for sequence matching
        similarity = difflib.SequenceMatcher(None, word1.lower(), word2.lower()).ratio()

        # Boost exact matches
        if word1.lower() == word2.lower():
            similarity = 1.0

        # Boost partial matches for short words
        elif len(word1) <= 3 and word1.lower() in word2.lower():
            similarity = max(similarity, 0.8)

        return similarity
