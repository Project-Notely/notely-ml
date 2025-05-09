import cv2
import numpy as np


def save_output_image(image: np.ndarray, path: str, show: bool = False) -> None:
    """
    Save an image to a file and optionally display it

    Args:
        image (np.ndarray): image to save
        path (str): path to save the image
    """
    cv2.imwrite(path, image)
    print(f"Output image saved to {path}")
    
    if show:
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()