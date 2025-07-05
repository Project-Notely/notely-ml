"""
Tests for segmentation Pydantic models
"""
import pytest
from pydantic import ValidationError
import json

from app.models.segmentation_models import (
    SegmentType,
    BoundingBox,
    DocumentSegment,
    SegmentationResult
)


class TestSegmentType:
    """Test SegmentType enum"""
    
    def test_segment_type_values(self):
        """Test all segment type values"""
        assert SegmentType.TITLE == "title"
        assert SegmentType.PARAGRAPH == "paragraph"
        assert SegmentType.TABLE == "table"
        assert SegmentType.LIST_ITEM == "list_item"
        assert SegmentType.IMAGE == "image"
        assert SegmentType.FIGURE == "figure"
        assert SegmentType.HEADER == "header"
        assert SegmentType.FOOTER == "footer"
        assert SegmentType.CAPTION == "caption"
        assert SegmentType.TEXT == "text"
    
    def test_segment_type_from_string(self):
        """Test creating SegmentType from string"""
        assert SegmentType("title") == SegmentType.TITLE
        assert SegmentType("paragraph") == SegmentType.PARAGRAPH
        assert SegmentType("table") == SegmentType.TABLE
    
    def test_segment_type_invalid_value(self):
        """Test invalid segment type value"""
        with pytest.raises(ValueError):
            SegmentType("invalid_type")
    
    def test_segment_type_iteration(self):
        """Test iterating over segment types"""
        types = list(SegmentType)
        assert len(types) == 10
        assert SegmentType.TITLE in types
        assert SegmentType.PARAGRAPH in types


class TestBoundingBox:
    """Test BoundingBox model"""
    
    def test_valid_bounding_box(self):
        """Test creating valid bounding box"""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50
    
    def test_bounding_box_with_zero_values(self):
        """Test bounding box with zero values"""
        bbox = BoundingBox(x=0, y=0, width=0, height=0)
        
        assert bbox.x == 0
        assert bbox.y == 0
        assert bbox.width == 0
        assert bbox.height == 0
    
    def test_bounding_box_with_negative_coordinates(self):
        """Test bounding box with negative coordinates"""
        # Negative coordinates should be allowed (might be valid in some coordinate systems)
        bbox = BoundingBox(x=-10, y=-20, width=100, height=50)
        
        assert bbox.x == -10
        assert bbox.y == -20
        assert bbox.width == 100
        assert bbox.height == 50
    
    def test_bounding_box_validation_missing_fields(self):
        """Test bounding box validation with missing fields"""
        with pytest.raises(ValidationError):
            BoundingBox(x=10, y=20, width=100)  # Missing height
            
        with pytest.raises(ValidationError):
            BoundingBox(x=10, y=20)  # Missing width and height
    
    def test_bounding_box_validation_wrong_types(self):
        """Test bounding box validation with wrong types"""
        with pytest.raises(ValidationError):
            BoundingBox(x="10", y=20, width=100, height=50)  # x should be int
            
        with pytest.raises(ValidationError):
            BoundingBox(x=10.5, y=20, width=100, height=50)  # x should be int, not float
    
    def test_bounding_box_dict_conversion(self):
        """Test converting bounding box to/from dict"""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        
        # Test to dict
        bbox_dict = bbox.dict()
        expected_dict = {"x": 10, "y": 20, "width": 100, "height": 50}
        assert bbox_dict == expected_dict
        
        # Test from dict
        bbox_from_dict = BoundingBox(**bbox_dict)
        assert bbox_from_dict == bbox
    
    def test_bounding_box_json_serialization(self):
        """Test JSON serialization/deserialization"""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        
        # Test to JSON
        json_str = bbox.json()
        assert isinstance(json_str, str)
        
        # Test from JSON
        bbox_from_json = BoundingBox.parse_raw(json_str)
        assert bbox_from_json == bbox


class TestDocumentSegment:
    """Test DocumentSegment model"""
    
    def test_valid_document_segment(self):
        """Test creating valid document segment"""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        segment = DocumentSegment(
            text="Sample text",
            segment_type=SegmentType.PARAGRAPH,
            bbox=bbox,
            confidence=0.85,
            metadata={"category": "NarrativeText"}
        )
        
        assert segment.text == "Sample text"
        assert segment.segment_type == SegmentType.PARAGRAPH
        assert segment.bbox == bbox
        assert segment.confidence == 0.85
        assert segment.metadata == {"category": "NarrativeText"}
    
    def test_document_segment_without_bbox(self):
        """Test document segment without bounding box"""
        segment = DocumentSegment(
            text="Sample text",
            segment_type=SegmentType.PARAGRAPH,
            confidence=0.85
        )
        
        assert segment.text == "Sample text"
        assert segment.segment_type == SegmentType.PARAGRAPH
        assert segment.bbox is None
        assert segment.confidence == 0.85
        assert segment.metadata == {}  # Should default to empty dict
    
    def test_document_segment_confidence_validation(self):
        """Test confidence validation"""
        # Valid confidence values
        segment1 = DocumentSegment(
            text="Test", 
            segment_type=SegmentType.TEXT, 
            confidence=0.0
        )
        assert segment1.confidence == 0.0
        
        segment2 = DocumentSegment(
            text="Test", 
            segment_type=SegmentType.TEXT, 
            confidence=1.0
        )
        assert segment2.confidence == 1.0
        
        segment3 = DocumentSegment(
            text="Test", 
            segment_type=SegmentType.TEXT, 
            confidence=0.5
        )
        assert segment3.confidence == 0.5
        
        # Invalid confidence values
        with pytest.raises(ValidationError):
            DocumentSegment(
                text="Test", 
                segment_type=SegmentType.TEXT, 
                confidence=-0.1
            )
            
        with pytest.raises(ValidationError):
            DocumentSegment(
                text="Test", 
                segment_type=SegmentType.TEXT, 
                confidence=1.1
            )
    
    def test_document_segment_missing_required_fields(self):
        """Test validation with missing required fields"""
        with pytest.raises(ValidationError):
            DocumentSegment(
                segment_type=SegmentType.TEXT, 
                confidence=0.5
            )  # Missing text
            
        with pytest.raises(ValidationError):
            DocumentSegment(
                text="Test", 
                confidence=0.5
            )  # Missing segment_type
            
        with pytest.raises(ValidationError):
            DocumentSegment(
                text="Test", 
                segment_type=SegmentType.TEXT
            )  # Missing confidence
    
    def test_document_segment_with_all_segment_types(self):
        """Test document segment with all possible segment types"""
        for segment_type in SegmentType:
            segment = DocumentSegment(
                text=f"Sample {segment_type.value}",
                segment_type=segment_type,
                confidence=0.8
            )
            assert segment.segment_type == segment_type
    
    def test_document_segment_metadata_types(self):
        """Test different metadata types"""
        # Test various metadata types
        metadata_cases = [
            {},  # Empty dict
            {"key": "value"},  # String value
            {"number": 42},  # Integer value
            {"float": 3.14},  # Float value
            {"bool": True},  # Boolean value
            {"list": [1, 2, 3]},  # List value
            {"nested": {"key": "value"}},  # Nested dict
            {"mixed": {"str": "test", "num": 42, "list": [1, 2]}},  # Mixed types
        ]
        
        for metadata in metadata_cases:
            segment = DocumentSegment(
                text="Test",
                segment_type=SegmentType.TEXT,
                confidence=0.5,
                metadata=metadata
            )
            assert segment.metadata == metadata
    
    def test_document_segment_dict_conversion(self):
        """Test converting document segment to/from dict"""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        segment = DocumentSegment(
            text="Sample text",
            segment_type=SegmentType.PARAGRAPH,
            bbox=bbox,
            confidence=0.85,
            metadata={"category": "NarrativeText"}
        )
        
        # Test to dict
        segment_dict = segment.dict()
        assert segment_dict["text"] == "Sample text"
        assert segment_dict["segment_type"] == "paragraph"
        assert segment_dict["bbox"] == {"x": 10, "y": 20, "width": 100, "height": 50}
        assert segment_dict["confidence"] == 0.85
        assert segment_dict["metadata"] == {"category": "NarrativeText"}
        
        # Test from dict
        segment_from_dict = DocumentSegment(**segment_dict)
        assert segment_from_dict == segment


class TestSegmentationResult:
    """Test SegmentationResult model"""
    
    def test_valid_segmentation_result_success(self, sample_segments):
        """Test creating valid successful segmentation result"""
        stats = {"total_segments": 4, "segment_types": {"paragraph": 2, "title": 1}}
        
        result = SegmentationResult(
            success=True,
            segments=sample_segments,
            total_segments=len(sample_segments),
            statistics=stats,
            strategy_used="hi_res"
        )
        
        assert result.success is True
        assert len(result.segments) == 4
        assert result.total_segments == 4
        assert result.statistics == stats
        assert result.strategy_used == "hi_res"
        assert result.error is None
    
    def test_valid_segmentation_result_failure(self):
        """Test creating valid failed segmentation result"""
        result = SegmentationResult(
            success=False,
            segments=[],
            error="Processing failed"
        )
        
        assert result.success is False
        assert result.segments == []
        assert result.total_segments == 0  # Should default to 0
        assert result.statistics == {}  # Should default to empty dict
        assert result.strategy_used is None
        assert result.error == "Processing failed"
    
    def test_segmentation_result_missing_required_fields(self):
        """Test validation with missing required fields"""
        with pytest.raises(ValidationError):
            SegmentationResult(segments=[])  # Missing success
            
        with pytest.raises(ValidationError):
            SegmentationResult(success=True)  # Missing segments
    
    def test_segmentation_result_with_empty_segments(self):
        """Test segmentation result with empty segments list"""
        result = SegmentationResult(
            success=True,
            segments=[]
        )
        
        assert result.success is True
        assert result.segments == []
        assert result.total_segments == 0
    
    def test_segmentation_result_segments_validation(self):
        """Test segments list validation"""
        # Valid segments
        valid_segment = DocumentSegment(
            text="Test", 
            segment_type=SegmentType.TEXT, 
            confidence=0.5
        )
        
        result = SegmentationResult(
            success=True,
            segments=[valid_segment]
        )
        assert len(result.segments) == 1
        
        # Invalid segments (wrong type)
        with pytest.raises(ValidationError):
            SegmentationResult(
                success=True,
                segments=["not a DocumentSegment"]
            )
    
    def test_segmentation_result_statistics_types(self):
        """Test different statistics types"""
        statistics_cases = [
            {},  # Empty dict
            {"total": 10},  # Simple stats
            {"types": {"paragraph": 5, "title": 2}},  # Nested dict
            {"confidence": 0.85, "processing_time": 1.23},  # Mixed types
            {"complex": {"nested": {"deep": "value"}}},  # Deep nesting
        ]
        
        for stats in statistics_cases:
            result = SegmentationResult(
                success=True,
                segments=[],
                statistics=stats
            )
            assert result.statistics == stats
    
    def test_segmentation_result_json_serialization(self, sample_segments):
        """Test JSON serialization/deserialization"""
        result = SegmentationResult(
            success=True,
            segments=sample_segments,
            total_segments=len(sample_segments),
            statistics={"total": 4},
            strategy_used="hi_res"
        )
        
        # Test to JSON
        json_str = result.json()
        assert isinstance(json_str, str)
        
        # Test from JSON
        result_from_json = SegmentationResult.parse_raw(json_str)
        assert result_from_json.success == result.success
        assert len(result_from_json.segments) == len(result.segments)
        assert result_from_json.total_segments == result.total_segments
        assert result_from_json.statistics == result.statistics
        assert result_from_json.strategy_used == result.strategy_used


class TestModelsIntegration:
    """Integration tests for all models together"""
    
    def test_complete_segmentation_workflow(self):
        """Test complete workflow with all models"""
        # Create bounding boxes
        bbox1 = BoundingBox(x=10, y=20, width=100, height=30)
        bbox2 = BoundingBox(x=10, y=60, width=200, height=80)
        
        # Create document segments
        segments = [
            DocumentSegment(
                text="Document Title",
                segment_type=SegmentType.TITLE,
                bbox=bbox1,
                confidence=0.95,
                metadata={"category": "Title", "font_size": 16}
            ),
            DocumentSegment(
                text="This is a paragraph with some content.",
                segment_type=SegmentType.PARAGRAPH,
                bbox=bbox2,
                confidence=0.88,
                metadata={"category": "NarrativeText", "word_count": 8}
            )
        ]
        
        # Create segmentation result
        statistics = {
            "total_segments": 2,
            "segment_types": {"title": 1, "paragraph": 1},
            "with_bounding_boxes": 2,
            "average_confidence": 0.915
        }
        
        result = SegmentationResult(
            success=True,
            segments=segments,
            total_segments=2,
            statistics=statistics,
            strategy_used="hi_res"
        )
        
        # Verify the complete structure
        assert result.success is True
        assert len(result.segments) == 2
        assert result.segments[0].segment_type == SegmentType.TITLE
        assert result.segments[1].segment_type == SegmentType.PARAGRAPH
        assert result.segments[0].bbox.width == 100
        assert result.segments[1].bbox.height == 80
        assert result.statistics["average_confidence"] == 0.915
    
    def test_models_dict_round_trip(self, sample_segments):
        """Test converting all models to dict and back"""
        # Test BoundingBox
        bbox = BoundingBox(x=1, y=2, width=3, height=4)
        bbox_dict = bbox.dict()
        bbox_restored = BoundingBox(**bbox_dict)
        assert bbox == bbox_restored
        
        # Test DocumentSegment
        segment = sample_segments[0]
        segment_dict = segment.dict()
        segment_restored = DocumentSegment(**segment_dict)
        assert segment == segment_restored
        
        # Test SegmentationResult
        result = SegmentationResult(
            success=True,
            segments=sample_segments,
            total_segments=len(sample_segments),
            statistics={"test": "value"}
        )
        result_dict = result.dict()
        result_restored = SegmentationResult(**result_dict)
        assert result == result_restored
    
    def test_models_json_round_trip(self, sample_segmentation_result):
        """Test JSON serialization round trip for all models"""
        # Convert to JSON and back
        json_str = sample_segmentation_result.json()
        result_restored = SegmentationResult.parse_raw(json_str)
        
        # Verify all data is preserved
        assert result_restored.success == sample_segmentation_result.success
        assert len(result_restored.segments) == len(sample_segmentation_result.segments)
        assert result_restored.total_segments == sample_segmentation_result.total_segments
        assert result_restored.statistics == sample_segmentation_result.statistics
        assert result_restored.strategy_used == sample_segmentation_result.strategy_used
        
        # Verify segments are preserved correctly
        for original, restored in zip(sample_segmentation_result.segments, result_restored.segments):
            assert original.text == restored.text
            assert original.segment_type == restored.segment_type
            assert original.confidence == restored.confidence
            assert original.metadata == restored.metadata
            if original.bbox:
                assert restored.bbox is not None
                assert original.bbox.x == restored.bbox.x
                assert original.bbox.y == restored.bbox.y
                assert original.bbox.width == restored.bbox.width
                assert original.bbox.height == restored.bbox.height
    
    def test_models_with_complex_nested_data(self):
        """Test models with complex nested metadata"""
        complex_metadata = {
            "processing": {
                "timestamp": "2024-01-01T12:00:00Z",
                "version": "1.0.0",
                "settings": {
                    "resolution": 300,
                    "color_mode": "RGB",
                    "filters": ["blur", "sharpen"]
                }
            },
            "analysis": {
                "confidence_breakdown": {
                    "character_level": 0.95,
                    "word_level": 0.88,
                    "line_level": 0.82
                },
                "languages_detected": ["en", "fr"],
                "estimated_reading_time": 120.5
            }
        }
        
        segment = DocumentSegment(
            text="Complex document with nested metadata",
            segment_type=SegmentType.PARAGRAPH,
            confidence=0.88,
            metadata=complex_metadata
        )
        
        # Test serialization/deserialization preserves complex data
        json_str = segment.json()
        segment_restored = DocumentSegment.parse_raw(json_str)
        
        assert segment_restored.metadata == complex_metadata
        assert segment_restored.metadata["processing"]["settings"]["resolution"] == 300
        assert segment_restored.metadata["analysis"]["languages_detected"] == ["en", "fr"] 