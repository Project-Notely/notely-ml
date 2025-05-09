import cv2
import numpy as np


def draw_bounding_boxes(
    image: np.ndarray,
    regions: list[tuple[int, int, int, int]],
    colour: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    add_labels: bool = True,
    text_colour: tuple[int, int, int] = (255, 255, 255),
    text_bg_colour: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Generic function to draw bounding boxes on an image

    Args:
        image (np.ndarray): image to draw on
        regions (list[tuple[int, int, int, int]]): list of bounding boxes
        colour (tuple[int, int, int], optional): colour of the bounding boxes
        thickness (int, optional): thickness of the bounding boxes
        add_labels (bool, optional): whether to add a label to the bounding boxes

    Returns:
        np.ndarray: image with bounding boxes drawn on it
    """
    # clone to avoid modifying the original
    output_image = image.copy()

    for text, box, confidence in regions:
        x, y, w, h = box

        # draw the rectangle
        cv2.rectangle(output_image, (x, y), (x + w, y + h), colour, thickness)

        # add label if requested
        if add_labels:
            label = f"'{text}' ({confidence:.2f})"
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                output_image, (x, y - text_h - 10), (x + text_w, y), text_bg_colour, -1
            )
            cv2.putText(
                output_image,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_colour,
                1,
            )

    return output_image
