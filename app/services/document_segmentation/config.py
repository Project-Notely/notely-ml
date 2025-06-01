DEFAULT_LAYOUT_MODEL = (
    "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"  # general document layout
)
DEFAULT_TABLE_MODEL = (
    "lp://TableBank/faster_rcnn_R_50_FPN_3x/config"  # specialized for tables
)

# label mapping for PubLayNet model
LABEL_MAP = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}

# visualization settings
TEXT_COLOR = (0, 0, 255)  # red
TITLE_COLOR = (255, 0, 0)  # blue
LIST_COLOR = (0, 255, 0)  # green
TABLE_COLOR = (255, 0, 255)  # purple
FIGURE_COLOR = (0, 255, 255)  # yellow

# colour mapping baed on labels
COLOUR_MAP = {
    "Text": TEXT_COLOR,
    "Title": TITLE_COLOR,
    "List": LIST_COLOR,
    "Table": TABLE_COLOR,
    "Figure": FIGURE_COLOR,
}

# default confidence threshold
CONFIDENCE_THRESHOLD = 0.5
