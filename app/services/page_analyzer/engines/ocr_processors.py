import torch
from transformers import TrOCRProcessor as HFTrOCRProcessor, VisionEncoderDecoderModel
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, Union
import logging

from ..interfaces.interfaces import OCRProcessor
from ..models.results import ProcessingResult
from ..models.elements import TextBox, OCRResult
from ..models.config import TrOCRConfig


class TrOCRProcessor(OCRProcessor):
    """TrOCR OCR processor"""

    MODELS = {
        "handwritten_small": TrOCRConfig(
            model_name="microsoft/trocr-small-handwritten",
            processor_name="microsoft/trocr-small-handwritten",
            description="Small model optimized for handwritten text",
            best_for="Fast processing of handwritten text",
        ),
        "handwritten_base": TrOCRConfig(
            model_name="microsoft/trocr-base-handwritten",
            processor_name="microsoft/trocr-base-handwritten",
            description="Base model optimized for handwritten text",
            best_for="Good balance of speed and accuracy for handwriting",
        ),
        "handwritten_large": TrOCRConfig(
            model_name="microsoft/trocr-large-handwritten",
            processor_name="microsoft/trocr-large-handwritten",
            description="Large model optimized for handwritten text",
            best_for="Best accuracy for challenging handwritten text",
        ),
        "printed_base": TrOCRConfig(
            model_name="microsoft/trocr-base-printed",
            processor_name="microsoft/trocr-base-printed",
            description="Base model optimized for printed text",
            best_for="Clean printed text and documents",
        ),
        "printed_large": TrOCRConfig(
            model_name="microsoft/trocr-large-printed",
            processor_name="microsoft/trocr-large-printed",
            description="Large model optimized for printed text",
            best_for="High accuracy printed text recognition",
        ),
    }

    def __init__(self, model_type: str = "handwritten_base", device: str = "cpu"):
        self.model_type = model_type
        self.device = device
        self.processor = None
        self.model = None
        self.initialized = False

        self.logger = logging.getLogger(__name__)

        # config
        self.max_length = 256
        self.num_beams = 4
        self.confidence_threshold = 0.1

    def initialize(self, **kwargs) -> bool:
        """
        Initialize the TrOCR processor

        Args:
            model_type: Override model type
            device: Override device
            max_length: Maximum sequence length
            num_beams: Beam search beams
            confidence_threshold: Minimum confidence threshold

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # update config
            self.model_type = kwargs.get("model_type", self.model_type)
            self.device = kwargs.get("device", self.device)
            self.max_length = kwargs.get("max_length", self.max_length)
            self.num_beams = kwargs.get("num_beams", self.num_beams)
            self.confidence_threshold = kwargs.get(
                "confidence_threshold", self.confidence_threshold
            )

            if self.model_type not in self.MODELS:
                available = list(self.MODELS.keys())
                raise ValueError(
                    f"Unknown model type: {self.model_type}. Available: {available}"
                )

            config = self.MODELS[self.model_type]

            self.logger.info(f"Loading TrOCR model: {config.model_name}")
            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"Description: {config.description}")

            # load processor and model
            self.processor = HFTrOCRProcessor.from_pretrained(config.processor_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(config.model_name)

            # Move model to device
            self.model.to(self.device)
            self.model.eval()  # set to evaluation mode

            self.initialized = True
            self.logger.info("TrOCR processor initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize TrOCR proccessor: {e}")
            return False

    def process_text_region(
        self,
        image_region: np.ndarray,
        preprocessing_options: Optional[Dict[str, Any]] = None,
    ) -> ProcessingResult:
        """
        Process a text region using TrOCR

        Args:
            image_region: Image region to process
            preprocessing_options: Optional preprocessing parameters

        Returns:
            ProcessingResult with OCRResult containing detected text and estimated locations
        """
        if not self.initialized:
            return ProcessingResult(
                success=False,
                result=None,
                error_message="TrOCR processor not initialized. Call initialize() first.",
            )

        try:
            # Preprocess image if needed
            processed_image = self._preprocess_image(
                image_region, preprocessing_options
            )

            # Convert to PIL Image
            pil_image = self._numpy_to_pil(processed_image)

            # Generate text with confidence estimation
            text_result = self._generate_text_with_confidence(pil_image)

            # Create text boxes (TrOCR doesn't provide word-level bounding boxes by default)
            # We'll estimate them or return the full region as one box
            text_boxes = self._create_text_boxes(
                text_result["text"], text_result["confidence"], image_region.shape
            )

            # Create OCR result
            ocr_result = OCRResult(
                full_text=text_result["text"],
                text_boxes=text_boxes,
                average_confidence=text_result["confidence"],
                total_words=(
                    len(text_result["text"].split()) if text_result["text"] else 0
                ),
                languages_used=["auto-detected"],
                processing_config=f"TrOCR-{self.model_type}",
            )

            return ProcessingResult(
                success=True,
                result=ocr_result,
                confidence=text_result["confidence"],
                metadata={
                    "model_type": self.model_type,
                    "device": self.device,
                    "image_shape": processed_image.shape,
                    "generation_config": {
                        "max_length": self.max_length,
                        "num_beams": self.num_beams,
                    },
                },
            )

        except Exception as e:
            self.logger.error(f"Error processing text region with TrOCR: {e}")
            return ProcessingResult(
                success=False,
                result=None,
                error_message=f"TrOCR processing failed: {str(e)}",
            )

    def _preprocess_image(
        self, image: np.ndarray, options: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Preprocess image for TrOCR (minimal preprocessing needed)

        Args:
            image: Input image
            options: Preprocessing options

        Returns:
            Preprocessed image
        """
        if options is None:
            options = {}

        processed = image.copy()

        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        else:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

        height, width = processed.shape[:2]

        # min size for good recognition
        min_size = options.get("min_size", 32)
        if height < min_size or width < min_size:
            scale = max(min_size / height, min_size / width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            processed = cv2.resize(
                processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC
            )

        max_size = options.get("max_size", 1024)
        if height > max_size or width > max_size:
            scale = min(max_size / height, max_size / width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            processed = cv2.resize(
                processed, (new_width, new_width), interpolation=cv2.INTER_AREA
            )

        return processed

    def _numpy_to_pil(self, image: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image"""
        if len(image.shape) == 3:
            return Image.fromarray(image)
        else:
            return Image.fromarray(image).convert("RGB")

    def _generate_text_with_confidence(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Generate text with confidence estimation

        Args:
            pil_image: PIL Image to process

        Returns:
            Dictionary with text and confidence
        """
        # prepare inputs
        pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            # generate with beam search for better quality
            generated_ids = self.model.generate(
                pixel_values,
                max_length=self.max_length,
                num_beams=self.num_beams,
                return_dict_in_generate=True,
                output_scores=True,
            )

            # decode text
            generated_text = self.processor.batch_decode(
                generated_ids.sequences,
                skip_special_tokens=True,
            )[0]

            # estimate confidence from generation scores
            confidence = self._estimate_confidence(generated_ids)

        return {
            "text": generated_text.strip(),
            "confidence": confidence,
        }

    def _estimate_confidence(self, generation_output) -> float:
        """
        Estimate confidence from generation scores

        Args:
            generation_output: Output from model.generate()

        Returns:
            Estimated confidence score (0-100)
        """
        try:
            if hasattr(generation_output, "scores") and generation_output.scores:
                # calculate average probability across all generated tokens
                scores = torch.stack(
                    generation_output.scores, dim=1
                )  # [batch, seq_len, vocab]
                probs = torch.softmax(scores, dim=-1)

                # get the probability of selected tokens
                selected_token_probs = []
                sequences = generation_output.sequences[:, 1:]  # skip start token

                for i in range(sequences.shape[1]):
                    if i < len(generation_output.scores):
                        token_id = sequences[0, i].item()
                        token_prob = probs[0, i, token_id].item()
                        selected_token_probs.append(token_prob)

                if selected_token_probs:
                    avg_prob = np.mean(selected_token_probs)
                    confidence = min(avg_prob * 100, 100.0)  # convert to percentage
                    return max(confidence, 0.0)

        except Exception as e:
            self.logger.warning(f"Could not estimate confidence: {e}")

        # fallback confidence based on text length and complexity
        return 0.0  # return 0 if confidence estimation fails

    def _create_text_boxes(
        self,
        text: str,
        confidence: float,
        image_shape: tuple,
    ) -> list[TextBox]:
        """
        Create text boxes from recognized text
        Note: TrOCR doesn't provide word-level localization by default
        This creates a single box covering the entire region

        Args:
            text: Recognized text
            confidence: Overall confidence
            image_shape: Shape of the processed image

        Returns:
            List of TextBox objects
        """
        text_boxes = []

        if text and len(text.strip()) > 0:
            # create one box for the entire region
            height, width = image_shape[:2]

            # for multi-word text, create separate boxes for each word
            words = text.strip().split()
            if len(words) > 1:
                # estimate word positions (basic approximation)
                word_width = width // len(words)

                for i, word in enumerate(words):
                    x = i * word_width
                    y = 0
                    w = word_width
                    h = height

                    text_boxes.append(
                        TextBox(
                            text=word,
                            confidence=confidence,
                            bbox=(x, y, w, h),
                            word_level=True,
                        )
                    )
            else:
                # single word or phrase - use entire region
                text_boxes.append(
                    TextBox(
                        text=text,
                        confidence=confidence,
                        bbox=(0, 0, width, height),
                        word_level=False,
                    )
                )

        return text_boxes

    def get_supported_languages(self) -> list[str]:
        """
        Get list of supported languages
        
        Returns:
            List of supported languages (TrOCR supports multiple languages)
        """
        # TrOCR models support multiple languages, primarily English
        # but can handle various scripts depending on the model
        return [
            "en",  # English (primary)
            "auto",  # Auto-detect
            "multilingual"  # Mixed languages
        ]

    def cleanup(self) -> None:
        """Clean up resources used by the TrOCR processor"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                # Move model off GPU if it was on GPU
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()
                # Clear model reference
                self.model = None
                
            if hasattr(self, 'processor') and self.processor is not None:
                # Clear processor reference
                self.processor = None
                
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.initialized = False
            self.logger.info("TrOCR processor resources cleaned up")
            
        except Exception as e:
            self.logger.warning(f"Error during TrOCR cleanup: {e}")
