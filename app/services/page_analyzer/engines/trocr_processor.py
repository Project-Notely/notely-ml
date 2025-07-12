from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import TrOCRProcessor as HFTrOCRProcessor
from transformers import VisionEncoderDecoderModel

from app.services.page_analyzer.models.models import (
    OCRResult,
    ProcessingResult,
    TextBox,
)


class TrOCRProcessor:
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
        if self.model_type not in self.MODELS:
            print(
                f"Unknown model type: {self.model_type}. Available: {list(self.MODELS.keys())}"
            )
            return False

        try:
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
        """Process entire region and detect individual words"""
        if not self.initialized:
            return ProcessingResult(success=False, error="Model not initialized")

        text_lines = self._detect_lines_simple(image)
        all_text_boxes = []
        all_texts = []
        total_confidence = 0

        try:
            for line_info in text_lines:
                line_image, line_y, line_height = line_info

                # process this line with TrOCR
                line_result, line_confidence = self._process_single_line(line_image)

                if line_result and line_result.strip():
                    # Detect word regions in this line
                    word_regions = self._detect_word_regions_simple(line_image)

                    # Match OCR words with detected regions
                    words = line_result.split()
                    if words:
                        matched_boxes = self._match_words_simple(
                            words, word_regions, line_y, line_confidence
                        )

                        for text_box in matched_boxes:
                            all_text_boxes.append(text_box)
                            all_texts.append(text_box.text)
                            total_confidence += text_box.confidence

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

    #########################################################
    # Helper functions
    #########################################################

    def _detect_lines_simple(
        self, image: np.ndarray
    ) -> list[Tuple[np.ndarray, int, int]]:
        """Simple line detection using horizontal morphology
        Args:
            image (np.ndarray): input image
        Returns:
            list[Tuple[np.ndarray, int, int]]: list of tuples containing the line image, the y-coordinate of the line, and the height of the line
        """
        # convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # apply binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # create horizontal kernel to connect text on the same line
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, horizontal_kernel)

        # find contours of text lines
        contours, _ = cv2.findContours(
            horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 10:  # Filter out noise
                # add padding
                padding = 5
                y_start = max(0, y - padding)
                y_end = min(image.shape[0], y + h + padding)
                line_image = image[y_start:y_end, :]
                lines.append((line_image, y_start, y_end - y_start))

        # sort lines by y-coordinate (top to bottom)
        lines.sort(key=lambda x: x[1])

        return lines if lines else [(image, 0, image.shape[0])]

    def _detect_word_regions_simple(self, line_image: np.ndarray) -> list[list[int]]:
        """Simple word detection using morphological operations"""
        # Convert to grayscale
        if len(line_image.shape) == 3:
            gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_image.copy()

        # Apply binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Create small kernel to connect letters within words
        word_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 3))
        word_regions = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, word_kernel)

        # Find contours of words
        contours, _ = cv2.findContours(
            word_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        words = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 5:  # Filter out noise
                words.append([int(x), int(y), int(w), int(h)])

        # Sort words by x-coordinate (left to right)
        words.sort(key=lambda box: box[0])

        return words

    def _match_words_simple(
        self,
        words: list[str],
        regions: list[list[int]],
        line_y: int,
        line_confidence: float,
    ) -> list[TextBox]:
        """Simple word matching with better fallback"""
        text_boxes = []

        if len(regions) == len(words):
            # Perfect match - use detected regions
            for word, region in zip(words, regions):
                x, y_rel, w, h = region
                bbox = [int(x), int(line_y + y_rel), int(w), int(h)]
                confidence = float(line_confidence * 100)
                text_boxes.append(TextBox(text=word, confidence=confidence, bbox=bbox))

        elif len(regions) > len(words) and len(regions) <= len(words) * 2:
            # More regions than words - merge adjacent regions
            merged_regions = self._merge_adjacent_regions(regions, len(words))
            for word, region in zip(words, merged_regions):
                x, y_rel, w, h = region
                bbox = [int(x), int(line_y + y_rel), int(w), int(h)]
                confidence = float(line_confidence * 100)
                text_boxes.append(TextBox(text=word, confidence=confidence, bbox=bbox))

        else:
            # Use intelligent estimation when detection fails
            text_boxes = self._estimate_word_positions_intelligent(
                words, line_y, line_confidence, regions
            )

        return text_boxes

    def _merge_adjacent_regions(
        self, regions: list[list[int]], target_count: int
    ) -> list[list[int]]:
        """Merge adjacent regions to match word count"""
        if len(regions) <= target_count:
            return regions

        merged = regions.copy()

        while len(merged) > target_count:
            # Find the closest pair of regions
            min_gap = float("inf")
            merge_idx = 0

            for i in range(len(merged) - 1):
                gap = merged[i + 1][0] - (merged[i][0] + merged[i][2])
                if gap < min_gap:
                    min_gap = gap
                    merge_idx = i

            # Merge the pair
            region1 = merged[merge_idx]
            region2 = merged[merge_idx + 1]

            merged_x = region1[0]
            merged_y = min(region1[1], region2[1])
            merged_w = (region2[0] + region2[2]) - region1[0]
            merged_h = max(region1[1] + region1[3], region2[1] + region2[3]) - merged_y

            merged[merge_idx] = [merged_x, merged_y, merged_w, merged_h]
            merged.pop(merge_idx + 1)

        return merged

    def _estimate_word_positions_intelligent(
        self,
        words: list[str],
        line_y: int,
        line_confidence: float,
        detected_regions: list[list[int]],
    ) -> list[TextBox]:
        """Intelligent word position estimation"""
        text_boxes = []

        if detected_regions:
            # Use detected regions as a guide
            line_width = max(r[0] + r[2] for r in detected_regions)
            line_height = max(r[3] for r in detected_regions)
            start_x = min(r[0] for r in detected_regions)
        else:
            # Complete fallback
            line_width = 500  # Assume reasonable width
            line_height = 20
            start_x = 10

        # Calculate average character width
        total_chars = sum(len(word) for word in words)
        if total_chars > 0:
            avg_char_width = max(8, (line_width - start_x - 20) // total_chars)
        else:
            avg_char_width = 10

        x_offset = start_x

        for word in words:
            # Calculate word width based on character count
            word_width = len(word) * avg_char_width

            # Add some variation for different character types
            if any(c in "mwMW" for c in word):  # Wide characters
                word_width = int(word_width * 1.2)
            elif any(c in "iIlj" for c in word):  # Narrow characters
                word_width = int(word_width * 0.8)

            bbox = [int(x_offset), int(line_y), int(word_width), int(line_height)]
            confidence = float(line_confidence * 100)
            text_boxes.append(TextBox(text=word, confidence=confidence, bbox=bbox))

            # Move to next word position with spacing
            x_offset += word_width + int(
                avg_char_width * 0.5
            )  # Add space between words

        return text_boxes

    def _process_single_line(self, line_image: np.ndarray) -> Tuple[str, float]:
        """Process a single line of text with TrOCR and return text with confidence"""
        try:
            # check line quality
            line_quality = self._assess_line_quality(line_image)
            if line_quality < 0.3:  # poor quality line
                print(
                    f"Warning: Low quality line detected (quality: {line_quality:.2f})"
                )

            # convert to PIL image
            if len(line_image.shape) == 3:
                image_rgb = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = line_image

            pil_image = Image.fromarray(image_rgb)

            # ensure minimum size for TrOCR
            if pil_image.size[0] < 32 or pil_image.size[1] < 16:
                print(f"Warning: Line image too small: {pil_image.size}")
                print(
                    f"Resizing to minimum size: {max(32, pil_image.size[0])}x{max(16, pil_image.size[1])}"
                )
                # resize to minimum size
                pil_image = pil_image.resize(
                    (max(32, pil_image.size[0]), max(16, pil_image.size[1]))
                )

            # process with TrOCR
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            # Generate text with scores
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    max_length=256,
                    num_beams=4,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            generated_ids = outputs.sequences
            scores = outputs.scores

            # Decode text
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[
                0
            ]

            # Calculate confidence from generation scores with improved method
            confidence = self._calculate_confidence_improved(
                scores, generated_ids, text
            )

            # Apply quality-based confidence adjustment
            confidence = confidence * line_quality

            print(
                f"Line processed: '{text[:50]}...' | Confidence: {confidence:.3f} | Quality: {line_quality:.3f}"
            )

            return text.strip(), confidence

        except Exception as e:
            print(f"Error processing line: {e}")
            return "", 0.0

    def _assess_line_quality(self, line_image: np.ndarray) -> float:
        """Assess the quality of a line image for OCR processing"""
        try:
            # convert to grayscale
            if len(line_image.shape) == 3:
                gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = line_image.copy()

            height, width = gray.shape

            # check size
            if width < 20 or height < 8:
                return 0.1  # too small

            # check if image has sufficient text content
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            text_pixels = np.sum(binary > 0)
            total_pixels = height * width
            text_ratio = text_pixels / total_pixels

            if text_ratio < 0.05:  # less than 5% text
                return 0.2
            elif text_ratio > 0.8:  # too much black (might be inverted or noise)
                return 0.3

            # check for reasonable aspect ratio
            aspect_ratio = width / height
            if aspect_ratio < 2:  # too square/tall
                return 0.4

            # Check contrast
            contrast = np.std(gray)
            if contrast < 20:  # low contrast
                return 0.5

            # calculate quality score
            quality = 1.0

            # penalize extreme text ratios
            if text_ratio < 0.1 or text_ratio > 0.6:
                quality *= 0.7

            # reward good aspect ratios
            if 3 <= aspect_ratio <= 20:
                quality *= 1.0
            else:
                quality *= 0.8

            # reward good contrast
            if contrast > 40:
                quality *= 1.0
            else:
                quality *= 0.9

            return min(quality, 1.0)

        except Exception as e:
            print(f"Error assessing line quality: {e}")
            return 0.5  # default middle quality

    def _calculate_confidence_improved(
        self, scores: tuple, generated_ids: torch.Tensor, text: str
    ) -> float:
        """Improved confidence calculation with better handling of edge cases"""
        try:
            if not scores or len(scores) == 0:
                return 0.5  # Default confidence instead of 0

            if not text or text.strip() == "":
                return 0.1  # Very low confidence for empty text

            # Get the generated tokens (excluding the initial token)
            generated_tokens = generated_ids[0][
                1:
            ]  # Skip the first token (usually BOS)

            if len(generated_tokens) == 0:
                return 0.1

            # Calculate confidence using multiple methods and take the best
            prob_confidence = self._calculate_probability_confidence(
                scores, generated_tokens
            )
            entropy_confidence = self._calculate_entropy_confidence(scores)

            # Combine confidences with weights
            combined_confidence = 0.7 * prob_confidence + 0.3 * entropy_confidence

            # Apply text-based adjustments
            text_confidence_factor = self._assess_text_confidence(text)
            final_confidence = combined_confidence * text_confidence_factor

            # Clamp to reasonable range
            final_confidence = min(max(final_confidence, 0.01), 1.0)

            return final_confidence

        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return 0.3  # Default fallback confidence

    def _calculate_probability_confidence(
        self, scores: tuple, generated_tokens: torch.Tensor
    ) -> float:
        """Calculate confidence based on token probabilities"""
        try:
            total_log_prob = 0.0
            num_tokens = 0

            for i, score_tensor in enumerate(scores):
                if i < len(generated_tokens):
                    # Convert logits to probabilities
                    probs = F.softmax(score_tensor[0], dim=-1)

                    # Get probability of the generated token
                    token_id = generated_tokens[i].item()
                    token_prob = probs[token_id].item()

                    # Add log probability (avoid log(0))
                    if token_prob > 1e-10:
                        total_log_prob += torch.log(torch.tensor(token_prob)).item()
                        num_tokens += 1

            if num_tokens > 0:
                # Convert average log probability back to probability
                avg_log_prob = total_log_prob / num_tokens
                confidence = torch.exp(torch.tensor(avg_log_prob)).item()
                return min(max(confidence, 0.0), 1.0)
            else:
                return 0.1

        except Exception:
            return 0.1

    def _calculate_entropy_confidence(self, scores: tuple) -> float:
        """Calculate confidence based on prediction entropy (lower entropy = higher confidence)"""
        try:
            total_entropy = 0.0
            num_tokens = len(scores)

            for score_tensor in scores:
                # Convert logits to probabilities
                probs = F.softmax(score_tensor[0], dim=-1)

                # Calculate entropy
                entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                total_entropy += entropy.item()

            if num_tokens > 0:
                avg_entropy = total_entropy / num_tokens
                # Convert entropy to confidence (lower entropy = higher confidence)
                # Normalize based on vocabulary size (log of vocab size is max entropy)
                max_entropy = torch.log(torch.tensor(50265.0))  # TrOCR vocab size
                normalized_entropy = avg_entropy / max_entropy.item()
                confidence = 1.0 - normalized_entropy
                return min(max(confidence, 0.0), 1.0)
            else:
                return 0.1

        except Exception:
            return 0.1

    def _assess_text_confidence(self, text: str) -> float:
        """Assess confidence based on the generated text characteristics"""
        try:
            if not text or len(text.strip()) == 0:
                return 0.1

            confidence_factor = 1.0

            # Check for reasonable text patterns
            words = text.split()

            # Penalize very short or very long sequences
            if len(words) < 1:
                confidence_factor *= 0.3
            elif len(words) > 50:  # Very long line might be wrong
                confidence_factor *= 0.7

            # Check for reasonable character patterns
            alpha_chars = sum(1 for c in text if c.isalpha())
            total_chars = len(text)

            if total_chars > 0:
                alpha_ratio = alpha_chars / total_chars
                if alpha_ratio < 0.3:  # Too few letters
                    confidence_factor *= 0.6
                elif alpha_ratio > 0.95:  # Good letter ratio
                    confidence_factor *= 1.1

            # Check for repeated characters (might indicate OCR errors)
            for char in set(text):
                if char.isalnum() and text.count(char) > len(text) * 0.3:
                    confidence_factor *= 0.5  # Too much repetition
                    break

            return min(confidence_factor, 1.0)

        except Exception:
            return 1.0

    def _estimate_word_positions(
        self,
        words: list[str],
        line_y: int,
        line_height: int,
        line_width: int,
        line_image: np.ndarray = None,
    ) -> list[Tuple[str, list[int]]]:
        """Deprecated - word positioning now handled in process_text_region"""
        # This method is kept for backward compatibility but shouldn't be used
        word_boxes = []
        for i, word in enumerate(words):
            x = i * 100  # Simple fallback
            bbox = [x, line_y, len(word) * 12, line_height]
            word_boxes.append((word, bbox))
        return word_boxes

    def cleanup(self):
        """Clean up resources"""
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.initialized = False
