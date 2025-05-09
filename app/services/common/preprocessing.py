import cv2
import numpy as np


def preprocess_image(
    image_path: str, use_clahe: bool = True, denoise: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generic image preprocessing pipeline

    Args:
        image_path: path to image
        use_clahe: whether to apply CLAHE contrast enhancement
        denoise: whether to apply denoising

    Returns:
        tuple of (original image, preprocessed image)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image at {image_path}")
            return None, None

        # convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # preprocessing pipeline
        processed = gray_img

        # enhance contrast with CLAHE if requested
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)

        # apply denoising if requested
        if denoise:
            processed = cv2.fastNlMeansDenoising(processed, None, 10, 7, 21)

        print("Preprocessing complete")
        return img, processed

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None, None
