"""
Configuration for unstructured document segmentation service
"""

from typing import Dict, Tuple

# Color mappings for different element types (R, G, B, Alpha)
ELEMENT_COLORS: Dict[str, Tuple[int, int, int, int]] = {
    'Title': (255, 0, 0, 100),        # Red
    'NarrativeText': (0, 255, 0, 100), # Green  
    'Text': (0, 255, 0, 100),         # Green
    'ListItem': (0, 0, 255, 100),     # Blue
    'Table': (255, 255, 0, 100),      # Yellow
    'Image': (255, 0, 255, 100),      # Magenta
    'Figure': (255, 0, 255, 100),     # Magenta
    'Header': (255, 165, 0, 100),     # Orange
    'Footer': (128, 0, 128, 100),     # Purple
    'PageBreak': (128, 128, 128, 100), # Gray
    'FigureCaption': (0, 255, 255, 100), # Cyan
    'Address': (255, 192, 203, 100),   # Pink
    'EmailAddress': (255, 192, 203, 100), # Pink
    'UncategorizedText': (169, 169, 169, 100), # Dark Gray
}

# Supported file formats
SUPPORTED_FORMATS = {
    '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls',
    '.html', '.htm', '.xml', '.txt', '.md', '.rtf', '.odt',
    '.png', '.jpg', '.jpeg', '.tiff', '.bmp'
}

# Default settings
DEFAULT_STRATEGY = "hi_res"
DEFAULT_DPI = 200
DEFAULT_FONT_SIZE = 12 