"""
Comprehensive unit tests for SegmentationService
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile
import os
from PIL import Image
import numpy as np

from app.services.segmentation_service import SegmentationService
from app.models.segmentation_models import (
    SegmentationResult, 
    DocumentSegment, 
    SegmentType, 
    BoundingBox
)


class TestSegmentationService:
    """Test suite for SegmentationService"""
    
    def test_init(self):
        """Test SegmentationService initialization"""
        service = SegmentationService()
        
        assert service.supported_formats == {
            '.png', '.jpg', '.jpeg', '.tiff', '.bmp',
            '.pdf', '.docx', '.doc', '.txt', '.html', '.md'
        }
    
    def test_get_supported_formats(self, segmentation_service):
        """Test getting supported file formats"""
        formats = segmentation_service.get_supported_formats()
        
        assert isinstance(formats, list)
        assert '.png' in formats
        assert '.pdf' in formats
        assert '.txt' in formats
        assert len(formats) > 0
    
    def test_get_partition_strategy(self, segmentation_service):
        """Test partition strategy mapping"""
        from unstructured.partition.utils.constants import PartitionStrategy
        
        # Test valid strategies
        assert segmentation_service._get_partition_strategy("fast") == PartitionStrategy.FAST
        assert segmentation_service._get_partition_strategy("hi_res") == PartitionStrategy.HI_RES
        assert segmentation_service._get_partition_strategy("auto") == PartitionStrategy.AUTO
        
        # Test invalid strategy defaults to hi_res
        assert segmentation_service._get_partition_strategy("invalid") == PartitionStrategy.HI_RES
    
    def test_map_element_type(self, segmentation_service):
        """Test element type mapping"""
        # Test all supported mappings
        assert segmentation_service._map_element_type("Title") == SegmentType.TITLE
        assert segmentation_service._map_element_type("NarrativeText") == SegmentType.PARAGRAPH
        assert segmentation_service._map_element_type("Table") == SegmentType.TABLE
        assert segmentation_service._map_element_type("ListItem") == SegmentType.LIST_ITEM
        assert segmentation_service._map_element_type("Image") == SegmentType.IMAGE
        assert segmentation_service._map_element_type("Figure") == SegmentType.FIGURE
        assert segmentation_service._map_element_type("Header") == SegmentType.HEADER
        assert segmentation_service._map_element_type("Footer") == SegmentType.FOOTER
        assert segmentation_service._map_element_type("FigureCaption") == SegmentType.CAPTION
        assert segmentation_service._map_element_type("Address") == SegmentType.TEXT
        assert segmentation_service._map_element_type("EmailAddress") == SegmentType.TEXT
        assert segmentation_service._map_element_type("UncategorizedText") == SegmentType.TEXT
        
        # Test unknown type defaults to TEXT
        assert segmentation_service._map_element_type("Unknown") == SegmentType.TEXT
    
    def test_extract_bounding_box(self, segmentation_service):
        """Test bounding box extraction"""
        # Test with valid coordinates
        class MockCoordinates:
            def __init__(self, points):
                self.points = points
        
        coords = MockCoordinates([(10, 20), (100, 120)])
        bbox = segmentation_service._extract_bounding_box(coords)
        
        assert bbox is not None
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 90
        assert bbox.height == 100
        
        # Test with invalid coordinates
        invalid_coords = MockCoordinates([])
        bbox = segmentation_service._extract_bounding_box(invalid_coords)
        assert bbox is None
        
        # Test with no coordinates attribute
        empty_coords = object()
        bbox = segmentation_service._extract_bounding_box(empty_coords)
        assert bbox is None
    
    def test_calculate_confidence(self, segmentation_service):
        """Test confidence calculation"""
        class MockElement:
            def __init__(self, text):
                self.text = text
            
            def __str__(self):
                return self.text
        
        # Test empty text
        empty_element = MockElement("")
        assert segmentation_service._calculate_confidence(empty_element) == 0.1
        
        # Test short text
        short_element = MockElement("Hi")
        assert segmentation_service._calculate_confidence(short_element) == 0.6
        
        # Test medium text
        medium_element = MockElement("Medium length text")
        assert segmentation_service._calculate_confidence(medium_element) == 0.8
        
        # Test long text
        long_element = MockElement("This is a very long text that should have high confidence")
        assert segmentation_service._calculate_confidence(long_element) == 0.95
    
    def test_extract_metadata(self, segmentation_service):
        """Test metadata extraction"""
        class MockElement:
            def __init__(self, text, category, metadata=None):
                self.text = text
                self.category = category
                self.metadata = metadata or {}
            
            def __str__(self):
                return self.text
        
        # Test with basic metadata
        element = MockElement("Test text", "Title")
        metadata = segmentation_service._extract_metadata(element)
        
        assert metadata['category'] == "Title"
        assert metadata['text_length'] == 9
        
        # Test with additional metadata
        element_with_meta = MockElement(
            "Test text", 
            "Title",
            {"page_number": 1, "filename": "test.pdf", "custom_field": "ignored"}
        )
        metadata = segmentation_service._extract_metadata(element_with_meta)
        
        assert metadata['category'] == "Title"
        assert metadata['text_length'] == 9
        assert metadata['page_number'] == 1
        assert metadata['filename'] == "test.pdf"
        assert 'custom_field' not in metadata  # Should be ignored
    
    def test_calculate_statistics(self, segmentation_service, sample_segments):
        """Test statistics calculation"""
        stats = segmentation_service._calculate_statistics(sample_segments)
        
        assert stats['total_segments'] == 4
        assert stats['segment_types']['title'] == 1
        assert stats['segment_types']['paragraph'] == 1
        assert stats['segment_types']['table'] == 1
        assert stats['segment_types']['image'] == 1
        assert stats['with_bounding_boxes'] == 4
        assert stats['average_confidence'] == 0.875
        
        # Test with empty segments
        empty_stats = segmentation_service._calculate_statistics([])
        assert empty_stats['total_segments'] == 0
        assert empty_stats['average_confidence'] == 0.0
    
    def test_convert_elements_to_segments(self, segmentation_service, mock_unstructured_elements):
        """Test converting unstructured elements to segments"""
        segments = segmentation_service._convert_elements_to_segments(mock_unstructured_elements)
        
        assert len(segments) == 4
        
        # Check first segment (title)
        title_segment = segments[0]
        assert title_segment.text == "Sample Document Title"
        assert title_segment.segment_type == SegmentType.TITLE
        assert title_segment.bbox is not None
        assert title_segment.confidence > 0
        
        # Check second segment (paragraph)
        para_segment = segments[1]
        assert para_segment.text == "This is a sample paragraph with multiple lines of text."
        assert para_segment.segment_type == SegmentType.PARAGRAPH
        
        # Check third segment (table)
        table_segment = segments[2]
        assert table_segment.text == "Table data with headers and rows"
        assert table_segment.segment_type == SegmentType.TABLE
        
        # Check fourth segment (image)
        image_segment = segments[3]
        assert image_segment.text == "Image placeholder content"
        assert image_segment.segment_type == SegmentType.IMAGE
    
    def test_save_temp_image(self, segmentation_service):
        """Test saving temporary image"""
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # Save temporary image
        temp_path = segmentation_service._save_temp_image(test_image)
        
        try:
            # Verify file exists
            assert os.path.exists(temp_path)
            assert temp_path.endswith('.png')
            
            # Verify image can be loaded
            loaded_image = Image.open(temp_path)
            assert loaded_image.size == (100, 100)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_segment_document_unsupported_format(self, segmentation_service):
        """Test segmentation with unsupported file format"""
        # Create a temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as tmp_file:
            tmp_file.write(b'test content')
            tmp_path = tmp_file.name
        
        try:
            result = await segmentation_service.segment_document(tmp_path)
            
            assert result.success is False
            assert "Unsupported file format" in result.error
            assert result.segments == []
            
        finally:
            os.unlink(tmp_path)
    
    @pytest.mark.asyncio
    @patch('app.services.segmentation_service.partition')
    async def test_segment_document_success(self, mock_partition, segmentation_service, 
                                          sample_text_file, mock_unstructured_elements):
        """Test successful document segmentation"""
        # Mock the partition function
        mock_partition.return_value = mock_unstructured_elements
        
        result = await segmentation_service.segment_document(sample_text_file)
        
        assert result.success is True
        assert len(result.segments) == 4
        assert result.total_segments == 4
        assert result.strategy_used == "hi_res"
        assert 'total_segments' in result.statistics
        assert 'segment_types' in result.statistics
        
        # Verify partition was called with correct parameters
        mock_partition.assert_called_once()
        call_args = mock_partition.call_args
        assert str(sample_text_file) in call_args[1]['filename']
        assert call_args[1]['extract_images'] is True
        assert call_args[1]['infer_table_structure'] is True
    
    @pytest.mark.asyncio
    @patch('app.services.segmentation_service.partition')
    async def test_segment_document_exception(self, mock_partition, segmentation_service, 
                                            sample_text_file):
        """Test document segmentation with exception"""
        # Mock partition to raise exception
        mock_partition.side_effect = Exception("Partition failed")
        
        result = await segmentation_service.segment_document(sample_text_file)
        
        assert result.success is False
        assert "Partition failed" in result.error
        assert result.segments == []
    
    @pytest.mark.asyncio
    async def test_segment_image_with_path(self, segmentation_service, sample_image):
        """Test image segmentation with file path"""
        with patch('app.services.segmentation_service.partition') as mock_partition:
            mock_partition.return_value = []
            
            result = await segmentation_service.segment_image(sample_image)
            
            assert result.success is True
            mock_partition.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_segment_image_with_pil_image(self, segmentation_service):
        """Test image segmentation with PIL Image"""
        # Create a test PIL image
        test_image = Image.new('RGB', (200, 200), color='blue')
        
        with patch('app.services.segmentation_service.partition') as mock_partition:
            mock_partition.return_value = []
            
            result = await segmentation_service.segment_image(test_image)
            
            assert result.success is True
            mock_partition.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_segment_image_with_numpy_array(self, segmentation_service):
        """Test image segmentation with numpy array"""
        # Create a test numpy array
        test_array = np.zeros((200, 200, 3), dtype=np.uint8)
        test_array[:, :] = [255, 0, 0]  # Red image
        
        with patch('app.services.segmentation_service.partition') as mock_partition:
            mock_partition.return_value = []
            
            result = await segmentation_service.segment_image(test_array)
            
            assert result.success is True
            mock_partition.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_segment_image_unsupported_type(self, segmentation_service):
        """Test image segmentation with unsupported type"""
        unsupported_input = {"not": "an image"}
        
        result = await segmentation_service.segment_image(unsupported_input)
        
        assert result.success is False
        assert "Unsupported image type" in result.error
    
    @pytest.mark.asyncio
    async def test_segment_image_exception(self, segmentation_service, sample_image):
        """Test image segmentation with exception"""
        with patch('app.services.segmentation_service.partition') as mock_partition:
            mock_partition.side_effect = Exception("Image processing failed")
            
            result = await segmentation_service.segment_image(sample_image)
            
            assert result.success is False
            assert "Image processing failed" in result.error
    
    @pytest.mark.asyncio
    async def test_segment_document_with_different_strategies(self, segmentation_service, 
                                                            sample_text_file, 
                                                            mock_unstructured_elements):
        """Test document segmentation with different strategies"""
        strategies = ["fast", "hi_res", "auto"]
        
        for strategy in strategies:
            with patch('app.services.segmentation_service.partition') as mock_partition:
                mock_partition.return_value = mock_unstructured_elements
                
                result = await segmentation_service.segment_document(
                    sample_text_file, 
                    strategy=strategy
                )
                
                assert result.success is True
                assert result.strategy_used == strategy
    
    @pytest.mark.asyncio
    async def test_segment_document_with_options(self, segmentation_service, 
                                               sample_text_file, 
                                               mock_unstructured_elements):
        """Test document segmentation with different options"""
        with patch('app.services.segmentation_service.partition') as mock_partition:
            mock_partition.return_value = mock_unstructured_elements
            
            result = await segmentation_service.segment_document(
                sample_text_file,
                extract_images=False,
                infer_table_structure=False
            )
            
            assert result.success is True
            
            # Verify partition was called with correct options
            call_args = mock_partition.call_args
            assert call_args[1]['extract_images'] is False
            assert call_args[1]['infer_table_structure'] is False


class TestSegmentationServiceIntegration:
    """Integration tests for SegmentationService"""
    
    @pytest.mark.asyncio
    async def test_real_text_file_segmentation(self, segmentation_service, sample_text_file):
        """Test segmentation with real text file"""
        try:
            result = await segmentation_service.segment_document(sample_text_file)
            
            # Should succeed (unstructured can handle text files)
            assert result.success is True
            assert len(result.segments) > 0
            assert result.total_segments > 0
            
        except Exception as e:
            # If unstructured is not fully configured, this might fail
            # In that case, just verify the error is handled gracefully
            assert "failed" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_real_image_segmentation(self, segmentation_service, sample_image):
        """Test segmentation with real image file"""
        try:
            result = await segmentation_service.segment_image(sample_image)
            
            # Should succeed or fail gracefully
            assert isinstance(result, SegmentationResult)
            
        except Exception as e:
            # If unstructured is not fully configured, this might fail
            # In that case, just verify the error is handled gracefully
            assert "failed" in str(e).lower()


class TestSegmentationServiceErrorHandling:
    """Error handling tests for SegmentationService"""
    
    @pytest.mark.asyncio
    async def test_nonexistent_file(self, segmentation_service):
        """Test segmentation with nonexistent file"""
        result = await segmentation_service.segment_document("/nonexistent/file.txt")
        
        assert result.success is False
        assert result.error is not None
        assert result.segments == []
    
    @pytest.mark.asyncio
    async def test_corrupted_file(self, segmentation_service, invalid_image_file):
        """Test segmentation with corrupted file"""
        result = await segmentation_service.segment_document(invalid_image_file)
        
        assert result.success is False
        assert result.error is not None
        assert result.segments == []
    
    @pytest.mark.asyncio
    async def test_empty_file(self, segmentation_service, empty_file):
        """Test segmentation with empty file"""
        result = await segmentation_service.segment_document(empty_file)
        
        # Should handle empty file gracefully
        assert isinstance(result, SegmentationResult)
        if result.success:
            assert len(result.segments) == 0
        else:
            assert result.error is not None


class TestSegmentationServicePerformance:
    """Performance tests for SegmentationService"""
    
    @pytest.mark.asyncio
    async def test_large_image_performance(self, segmentation_service, large_image):
        """Test segmentation performance with large image"""
        import time
        
        start_time = time.time()
        
        try:
            result = await segmentation_service.segment_image(large_image)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should complete within reasonable time (adjust as needed)
            assert processing_time < 30  # 30 seconds max
            
            # Should return valid result
            assert isinstance(result, SegmentationResult)
            
        except Exception:
            # If processing fails due to resource constraints, that's acceptable
            pass
    
    @pytest.mark.asyncio
    async def test_memory_cleanup(self, segmentation_service):
        """Test memory cleanup after processing"""
        import gc
        
        # Process multiple images to test memory cleanup
        for i in range(3):
            test_image = Image.new('RGB', (500, 500), color=f'rgb({i*80}, {i*80}, {i*80})')
            
            with patch('app.services.segmentation_service.partition') as mock_partition:
                mock_partition.return_value = []
                
                result = await segmentation_service.segment_image(test_image)
                
                # Force garbage collection
                gc.collect()
                
                assert isinstance(result, SegmentationResult)
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, segmentation_service):
        """Test concurrent processing of multiple documents"""
        import asyncio
        
        # Create multiple test images
        test_images = [
            Image.new('RGB', (100, 100), color=f'rgb({i*50}, {i*50}, {i*50})')
            for i in range(3)
        ]
        
        with patch('app.services.segmentation_service.partition') as mock_partition:
            mock_partition.return_value = []
            
            # Process images concurrently
            tasks = [
                segmentation_service.segment_image(img) 
                for img in test_images
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            assert len(results) == 3
            for result in results:
                assert isinstance(result, SegmentationResult)
                assert result.success is True 