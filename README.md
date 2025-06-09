# Notely TrOCR - Clean Implementation

A focused, clean implementation of TrOCR (Transformer-based Optical Character Recognition) for processing text from images with high accuracy and proper text region highlighting.

## Features

- **Multiple TrOCR Models**: Support for handwritten and printed text models
- **High Accuracy**: Advanced preprocessing pipeline for optimal OCR results
- **Text Highlighting**: Color-coded confidence visualization
- **Clean Architecture**: Well-organized, maintainable codebase
- **Comprehensive Testing**: Full test suite with real image validation

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run tests to verify installation:
```bash
python tests/test_trocr.py
```

## Usage

### Basic Usage

```python
from notely_trocr import TrOCRProcessor
import cv2

# Initialize processor
processor = TrOCRProcessor(model_type="printed_base", device="cpu")
processor.initialize()

# Load and process image
image = cv2.imread("your_image.png")
result = processor.process_text_region(image)

if result.success:
    print(f"Detected text: {result.result.full_text}")
    print(f"Confidence: {result.result.average_confidence:.2f}%")
    print(f"Words detected: {result.result.total_words}")

# Clean up
processor.cleanup()
```

### Available Models

- `handwritten_small`: Fast processing of handwritten text
- `handwritten_base`: Balanced speed/accuracy for handwriting
- `handwritten_large`: High accuracy for handwritten text
- `printed_small`: Fast processing of printed text
- `printed_base`: Balanced speed/accuracy for printed text
- `printed_large`: High accuracy for printed text

### Text Highlighting

```python
from notely_trocr.utils import create_highlighted_image
from pathlib import Path

# Create highlighted image with confidence colors
success = create_highlighted_image(
    image, 
    result.result.text_boxes, 
    Path("highlighted_output.png")
)
```

## Project Structure

```
trocr_clean/
├── notely_trocr/          # Main TrOCR module
│   ├── __init__.py        # Module exports
│   ├── processor.py       # TrOCR processor implementation
│   ├── models.py          # Data models
│   ├── interfaces.py      # Abstract interfaces
│   └── utils.py           # Utility functions
├── tests/                 # Test suite
│   └── test_trocr.py     # Comprehensive tests
├── data/                  # Test data
│   └── paragraph_potato.png
├── output/                # Output directory
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Testing

The test suite includes:

- **Unit Tests**: Core functionality validation
- **Integration Tests**: End-to-end processing
- **Real Image Tests**: Validation with actual images
- **Highlighting Tests**: Visualization functionality

Run all tests:
```bash
python tests/test_trocr.py
```

## Performance

The implementation achieves excellent results on real images:

- **Confidence**: 85%+ on clear printed text
- **Accuracy**: High word-level detection
- **Speed**: Optimized for CPU processing
- **Highlighting**: Perfect color-coded visualization

## Confidence Color Coding

- 🟢 **Green**: 80%+ confidence (high accuracy)
- 🟡 **Yellow**: 60-79% confidence (medium accuracy)
- 🟠 **Orange**: 40-59% confidence (low accuracy)
- 🔴 **Red**: <40% confidence (very low accuracy)

## License

This implementation is provided as-is for educational and research purposes. 