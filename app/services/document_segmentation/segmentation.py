import layoutparser as lp
import numpy as np
import cv2
from .config import DEFAULT_LAYOUT_MODEL, DEFAULT_TABLE_MODEL, LABEL_MAP, COLOUR_MAP, CONFIDENCE_THRESHOLD


# global model cache to avoid reloading models
_layout_model = None
_table_model = None

def get_layout_model(model_path=DEFAULT_LAYOUT_MODEL):
    """Load or retrieve the cached layout analysis model

    Args:
        model_path: Model path or identifier (can use layoutparser's lp:// format)
        
    Returns:
        Loaded layout model
    """
    global _layout_model
    if _layout_model is None:
        print(f"Loading layout model: {model_path}")
        _layout_model = lp.Detectron2LayoutModel(
            model_path,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", CONFIDENCE_THRESHOLD],
            label_map=LABEL_MAP,
        )
    return _layout_model

def get_table_model(model_path=DEFAULT_TABLE_MODEL):
    """Load or retrieve the cached table detection model

    Args:
        model_path: Model path or identifier (can use layoutparser's lp:// format)
        
    Returns:
        Loaded table detection model
    """
    global _table_model
    if _table_model is None:
        print(f"Loading table model: {model_path}")
        _table_model = lp.Detectron2LayoutModel(
            model_path,
            extra_config=['MODEL.ROI_HEADS.SCORE_THRESH_TEST', CONFIDENCE_THRESHOLD]
        )
    return _table_model

def load_image(image_path):
    """Load an image for layout analysis"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        # convert to RGB for compatibility with layout models
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def analyze_layout(image, model=None):
    """Analyze the document layout

    Args:
        image: Image as numpy array (RBG)
        model: Optional pre-loaded model
        
    Returns:
        Layout analysis result with detected blocks
    """
    if model is None:
        model = get_layout_model()
        
    # detect layout components
    layout = model.detect(image)
    print(f"Detected {len(layout)} layout components")
    return layout

def detect_tables(image, model=None):
    """Specialized table detection
    
    Args:
        image: Image as numpy array (RGB)
        model: Optional pre-loaded model

    Returns:
        Detected tables
    """
    if model is None:
        model = get_table_model()
        
    # detect tables
    tables = model.detet(image)
    print(f"Detected {len(tables)} tables")
    return tables

def merge_layouts(general_layout, tables_layout):
    """Merge layout results, prioritizing specialized models

    Args:
        general_layout: General layout from the main model
        tables_layout: Table-specific layout from the specialized model
    
    Returns:
        Merged layout
    """
    # start with general layout
    merged = general_layout
    
    # if we have specialized table detections, replace general tables with specialized ones
    if tables_layout:
        # filter out general table detections
        no_tables = lp.Layout([block for block in general_layout if block.type != 'Table'])
        # add specialized table detections
        for table in tables_layout:
            table.type = "Table"  # ensure correct type
        merged = no_tables + tables_layout
    
    return merged

def sort_blocks_reading_order(layout):
    """
    Sort layout blocks in natural reading order (top to bottom, left to right)
    
    Args:
        layout: Layout object with blocks

    Returns:
        sorted layout
    """
    # use built-in reading order sorting
    return layout.sort(key=lambda b: (b.coordinates[1], b.coordinates[0]))

def group_text_blocks(layout, vertical_threshold=20):
    """
    Group text blocks that belong to the same paragraph

    Args:
        layout: layout with text blocks
        vertical_threshold: max vertical distance to consider blocks part of the same paragraph
        
    Returns:
        layout with grouped text blocks
    """
    # get all text blocks
    text_blocks = [b for b in layout if b.type == 'Text']
    if not text_blocks:
        return layout
    
    # sort by y-position (top to bottom)
    text_blocks.sort(key=lambda b: b.coordinates[1])
    
    # group blocks that are close vertically
    grouped_blocks = []
    current_group = [text_blocks[0]]
    
    for block in text_blocks[1:]:
        prev_block = current_group[-1]
        # if this block is close enough to the previous one, add to the current group
        if block.coordinates[1] - (prev_block.coordinates[1]) + prev_block.height < vertical_threshold:
            current_group.append(block)
        else:
            # create a new grouped block form the current group
            if current_group:
                x1 = min(b.coordinates[0] for b in current_group)
                y1 = min(b.coordinates[1] for b in current_group)
                x2 = max(b.coordinates[0] + b.width for b in current_group)
                y2 = max(b.coordinates[1] + b.height for b in current_group)
                
                grouped_block = lp.TextBlock(
                    block=lp.Rectangle(x1, y1, x2, y2),
                    type='Text',
                    text='Paragraph'  # placeholder
                )
                grouped_blocks.append(grouped_block)
                
            # start a new group
            current_group = [block]
    
    # add the last group
    if current_group:
        x1 = min(b.coordinates[0] for b in current_group)
        y1 = min(b.coordinates[1] for b in current_group)
        x2 = max(b.coordinates[0] + b.width for b in current_group)
        y2 = max(b.coordinates[1] + b.height for b in current_group)
        
        grouped_blocks = lp.TextBlock(
            block=lp.Rectangle(x1, y1, x2, y2),
            type='Text',
            text='Paragraph'  # placeholder
        )
        grouped_blocks.append(grouped_block)
        
    # replace text blocks with grouped blocks
    non_text_blocks = [b for b in layout if b.type != 'Text']
    result_layout = lp.Layout(non_text_blocks + grouped_blocks)
    
    return result_layout             