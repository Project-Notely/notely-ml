import cv2
import numpy as np
import layoutparser as lp
from .config import COLOUR_MAP, CONFIDENCE_THRESHOLD


def draw_blocks(
    image, layout, show_element_type=True, show_confidence=True, with_box_on_text=True
):
    """
    Draw the layout blocks on the image with customized visualization

    Args:
        image: Original image (RGB)
        layout: Layout analysis result
        show_element_type: Whether to show the element type
        show_confidence: Whether to show the confidence score
        with_box_on_text: Whether to draw a box around the text

    Returns:
        Image with layout blocks drawn on it
    """
    # convert RGV to BGR for OpenCV
    vis_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    # create a layout figure
    for block in layout:
        # skip low confidence blocks
        if hasattr(block, "score") and block.score < CONFIDENCE_THRESHOLD:
            continue

        # get colour for this type
        colour = COLOUR_MAP.get(
            block.type, (200, 200, 200)
        )  # default gray for unkown types

        # extract coordinates
        x1, y1, x2, y2 = block.coordinates

        # draw rectangle
        cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)

        # prepare label
        label_parts = []
        if show_element_type:
            label_parts.append(block.type)
        if show_confidence:
            label_parts.append(f"{block.score:.2f}")
        if label_parts:
            label = ": ".join(label_parts)
            # Draw label background
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                vis_image,
                (int(x1), int(y1) - text_h - 10),
                (int(x1) + text_w, int(y1)),
                colour,
                -1,
            )
            # Draw label text
            cv2.putText(
                vis_image,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White text
                1,
            )

    return vis_image


def create_block_image(image, layout):
    """
    Create an image with colored blocks for layout visualization

    Args:
        image: Original image
        layout: Layout analysis result

    Returns:
        Image with colored blocks
    """
    # Create a blank image with the same dimensions
    h, w = image.shape[:2]
    block_image = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw filled colored rectangles for each block
    for block in layout:
        # Skip low confidence blocks
        if hasattr(block, "score") and block.score < CONFIDENCE_THRESHOLD:
            continue

        # Get color for this type
        color = COLOUR_MAP.get(
            block.type, (200, 200, 200)
        )  # Default gray for unknown types

        # Extract coordinates
        x1, y1, x2, y2 = block.coordinates

        # Draw filled rectangle
        cv2.rectangle(
            block_image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            -1,  # Filled rectangle
        )

    # Add transparency
    alpha = 0.4
    overlay = image.copy()
    output = cv2.addWeighted(
        cv2.cvtColor(block_image, cv2.COLOR_RGB2BGR),
        alpha,
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        1 - alpha,
        0,
    )

    return output
