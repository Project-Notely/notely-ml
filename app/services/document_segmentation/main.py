import cv2
import numpy as np
from ..common.utils import save_output_image
from .segmentation import (
    load_image,
    analyze_layout,
    detect_tables,
    merge_layouts,
    sort_blocks_reading_order,
    group_text_blocks,
    get_layout_model,
    get_table_model,
)
from .visualization import draw_blocks, create_block_image


def segment_document(image_path, group_paragraphs=True, use_table_model=True):
    """
    Main function to segment a document into layout blocks

    Args:
        image_path: Path to the document image
        group_paragraphs: Whether to group text blocks into paragraphs
        use_table_model: Whether to use the specialized table model

    Returns:
        Tuple of (original image, segmented layout, visualization image)
    """
    # Load the image
    image = load_image(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None, None, None

    # Get the layout model
    layout_model = get_layout_model()

    # Analyze layout
    layout = analyze_layout(image, layout_model)

    # If requested, use specialized table model
    if use_table_model:
        table_model = get_table_model()
        tables = detect_tables(image, table_model)
        layout = merge_layouts(layout, tables)

    # Sort blocks in reading order
    layout = sort_blocks_reading_order(layout)

    # Group text blocks into paragraphs if requested
    if group_paragraphs:
        layout = group_text_blocks(layout)

    # Create visualization
    vis_image = draw_blocks(image, layout)

    return image, layout, vis_image


def segment_and_save(
    image_path, output_path="segmented_document.png", group_paragraphs=True
):
    """
    Segment a document and save the visualization

    Args:
        image_path: Path to the document image
        output_path: Path to save the visualization
        group_paragraphs: Whether to group text blocks into paragraphs

    Returns:
        The segmented layout
    """
    _, layout, vis_image = segment_document(image_path, group_paragraphs)

    if vis_image is not None:
        # Convert BGR to RGB
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        # Save the visualization
        save_output_image(vis_image, output_path, show=True)
        print(f"Segmentation complete. Output saved to {output_path}")
    else:
        print("Segmentation failed")

    return layout


def get_blocks_by_type(layout, block_type):
    """
    Extract blocks of a specific type from the layout

    Args:
        layout: Layout object
        block_type: Type of block to extract (e.g., "Text", "Figure")

    Returns:
        List of blocks matching the type
    """
    return [block for block in layout if block.type == block_type]


def export_layout_data(layout):
    """
    Export layout data as a list of dictionaries

    Args:
        layout: Layout object

    Returns:
        List of block dictionaries with type, coordinates, and confidence
    """
    blocks_data = []

    for block in layout:
        x1, y1, x2, y2 = block.coordinates
        block_dict = {
            "type": block.type,
            "coordinates": {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "width": int(x2 - x1),
                "height": int(y2 - y1),
            },
        }

        if hasattr(block, "score"):
            block_dict["confidence"] = float(block.score)

        if hasattr(block, "text") and block.text:
            block_dict["text"] = block.text

        blocks_data.append(block_dict)

    return blocks_data


if __name__ == "__main__":
    # Example usage
    image_path = "example_document.png"
    layout = segment_and_save(image_path)

    # Print block types
    if layout:
        block_types = set(block.type for block in layout)
        print(f"Document contains blocks of types: {block_types}")

        # Count blocks by type
        for block_type in block_types:
            count = len(get_blocks_by_type(layout, block_type))
            print(f"  - {block_type}: {count} blocks")
