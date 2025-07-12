"""
Tests for segmentation controller functions
"""

import asyncio
import io
import os
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from app.controllers.segmentation_controller import (
    get_supported_formats,
    segment_document,
)
from app.models.segmentation_models import SegmentationResult, SegmentType


class MockUploadFile:
    """Mock UploadFile for testing"""

    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self.content = content
        self.file = io.BytesIO(content)

    async def read(self) -> bytes:
        """Read file content"""
        return self.content

    def seek(self, position: int):
        """Seek to position"""
        self.file.seek(position)


class TestSegmentDocument:
    """Test segment_document controller function"""

    @pytest.mark.asyncio
    async def test_segment_document_success(self, sample_segmentation_result):
        """Test successful document segmentation"""
        # Create mock upload file
        mock_file = MockUploadFile("test.txt", b"Sample document content")

        # Mock the segmentation service
        with patch(
            "app.controllers.segmentation_controller.SegmentationService"
        ) as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.segment_document = AsyncMock(
                return_value=sample_segmentation_result
            )

            result = await segment_document(
                file=mock_file,
                strategy="hi_res",
                extract_images=True,
                infer_table_structure=True,
            )

            # Verify result
            assert result.success is True
            assert len(result.segments) == len(sample_segmentation_result.segments)
            assert result.strategy_used == sample_segmentation_result.strategy_used

            # Verify service was called correctly
            mock_service.segment_document.assert_called_once()
            call_args = mock_service.segment_document.call_args
            assert call_args[1]["strategy"] == "hi_res"
            assert call_args[1]["extract_images"] is True
            assert call_args[1]["infer_table_structure"] is True

    @pytest.mark.asyncio
    async def test_segment_document_no_filename(self):
        """Test segmentation with no filename"""
        # Create mock upload file without filename
        mock_file = MockUploadFile(None, b"content")

        result = await segment_document(mock_file)

        # Should return error result
        assert result.success is False
        assert "No file provided" in result.error
        assert result.segments == []

    @pytest.mark.asyncio
    async def test_segment_document_service_failure(self):
        """Test segmentation when service fails"""
        mock_file = MockUploadFile("test.txt", b"Sample content")

        # Mock service to return failure
        failed_result = SegmentationResult(
            success=False, error="Service processing failed", segments=[]
        )

        with patch(
            "app.controllers.segmentation_controller.SegmentationService"
        ) as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.segment_document = AsyncMock(return_value=failed_result)

            result = await segment_document(mock_file)

            assert result.success is False
            assert "Service processing failed" in result.error
            assert result.segments == []

    @pytest.mark.asyncio
    async def test_segment_document_service_exception(self):
        """Test segmentation when service raises exception"""
        mock_file = MockUploadFile("test.txt", b"Sample content")

        with patch(
            "app.controllers.segmentation_controller.SegmentationService"
        ) as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.segment_document = AsyncMock(
                side_effect=Exception("Service crashed")
            )

            result = await segment_document(mock_file)

            assert result.success is False
            assert "Service crashed" in result.error
            assert result.segments == []

    @pytest.mark.asyncio
    async def test_segment_document_file_handling(self):
        """Test proper file handling and cleanup"""
        mock_file = MockUploadFile("test.txt", b"Sample document content")

        with patch(
            "app.controllers.segmentation_controller.SegmentationService"
        ) as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.segment_document = AsyncMock(
                return_value=SegmentationResult(success=True, segments=[])
            )

            # Mock tempfile creation to track file creation/deletion
            created_files = []
            original_tempfile = tempfile.NamedTemporaryFile

            def mock_tempfile(*args, **kwargs):
                temp_file = original_tempfile(*args, **kwargs)
                created_files.append(temp_file.name)
                return temp_file

            with patch("tempfile.NamedTemporaryFile", side_effect=mock_tempfile):
                await segment_document(mock_file)

                # Verify service was called with a file path
                mock_service.segment_document.assert_called_once()
                call_args = mock_service.segment_document.call_args
                file_path = call_args[1]["file_path"]

                # File should be cleaned up after processing
                assert not os.path.exists(file_path)

    @pytest.mark.asyncio
    async def test_segment_document_different_strategies(self):
        """Test segmentation with different strategies"""
        mock_file = MockUploadFile("test.txt", b"content")

        strategies = ["fast", "hi_res", "auto"]

        for strategy in strategies:
            with patch(
                "app.controllers.segmentation_controller.SegmentationService"
            ) as mock_service_class:
                mock_service = mock_service_class.return_value
                mock_service.segment_document = AsyncMock(
                    return_value=SegmentationResult(
                        success=True, segments=[], strategy_used=strategy
                    )
                )

                result = await segment_document(file=mock_file, strategy=strategy)

                assert result.success is True

                # Verify strategy was passed correctly
                call_args = mock_service.segment_document.call_args
                assert call_args[1]["strategy"] == strategy

    @pytest.mark.asyncio
    async def test_segment_document_options(self):
        """Test segmentation with different options"""
        mock_file = MockUploadFile("test.txt", b"content")

        test_cases = [
            {"extract_images": True, "infer_table_structure": True},
            {"extract_images": False, "infer_table_structure": True},
            {"extract_images": True, "infer_table_structure": False},
            {"extract_images": False, "infer_table_structure": False},
        ]

        for options in test_cases:
            with patch(
                "app.controllers.segmentation_controller.SegmentationService"
            ) as mock_service_class:
                mock_service = mock_service_class.return_value
                mock_service.segment_document = AsyncMock(
                    return_value=SegmentationResult(success=True, segments=[])
                )

                result = await segment_document(file=mock_file, **options)

                assert result.success is True

                # Verify options were passed correctly
                call_args = mock_service.segment_document.call_args
                assert call_args[1]["extract_images"] == options["extract_images"]
                assert (
                    call_args[1]["infer_table_structure"]
                    == options["infer_table_structure"]
                )

    @pytest.mark.asyncio
    async def test_segment_document_large_file(self):
        """Test segmentation with large file"""
        # Create a large mock file (1MB)
        large_content = b"x" * (1024 * 1024)
        mock_file = MockUploadFile("large_file.txt", large_content)

        with patch(
            "app.controllers.segmentation_controller.SegmentationService"
        ) as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.segment_document = AsyncMock(
                return_value=SegmentationResult(success=True, segments=[])
            )

            result = await segment_document(mock_file)

            assert result.success is True
            mock_service.segment_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_segment_document_binary_file(self):
        """Test segmentation with binary file"""
        # Create binary content
        binary_content = bytes(range(256))
        mock_file = MockUploadFile("binary_file.bin", binary_content)

        with patch(
            "app.controllers.segmentation_controller.SegmentationService"
        ) as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.segment_document = AsyncMock(
                return_value=SegmentationResult(success=True, segments=[])
            )

            result = await segment_document(mock_file)

            assert result.success is True
            mock_service.segment_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_segment_document_file_permissions(self):
        """Test segmentation with file permission issues"""
        mock_file = MockUploadFile("test.txt", b"content")

        # Mock tempfile to fail
        with patch(
            "tempfile.NamedTemporaryFile",
            side_effect=PermissionError("Permission denied"),
        ):
            result = await segment_document(mock_file)

            assert result.success is False
            assert "Permission denied" in result.error


class TestGetSupportedFormats:
    """Test get_supported_formats controller function"""

    @pytest.mark.asyncio
    async def test_get_supported_formats_success(self):
        """Test getting supported formats successfully"""
        expected_formats = [".png", ".jpg", ".pdf", ".txt"]

        with patch(
            "app.controllers.segmentation_controller.SegmentationService"
        ) as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.get_supported_formats.return_value = expected_formats

            result = await get_supported_formats()

            assert result == expected_formats
            mock_service.get_supported_formats.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_supported_formats_empty(self):
        """Test getting supported formats when empty"""
        with patch(
            "app.controllers.segmentation_controller.SegmentationService"
        ) as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.get_supported_formats.return_value = []

            result = await get_supported_formats()

            assert result == []
            mock_service.get_supported_formats.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_supported_formats_service_exception(self):
        """Test getting supported formats when service raises exception"""
        with patch(
            "app.controllers.segmentation_controller.SegmentationService"
        ) as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.get_supported_formats.side_effect = Exception("Service error")

            # Should propagate the exception
            with pytest.raises(Exception, match="Service error"):
                await get_supported_formats()


class TestControllerIntegration:
    """Integration tests for controller functions"""

    @pytest.mark.asyncio
    async def test_segment_document_real_service_integration(self, sample_text_file):
        """Test controller with real service (mocked unstructured)"""
        # Read real file content
        with open(sample_text_file, "rb") as f:
            content = f.read()

        mock_file = MockUploadFile(sample_text_file.name, content)

        # Mock only the unstructured partition function
        with patch("app.services.segmentation_service.partition") as mock_partition:
            # Create mock elements
            class MockElement:
                def __init__(self, text, category):
                    self.text = text
                    self.category = category
                    self.metadata = {}

                def __str__(self):
                    return self.text

            mock_elements = [
                MockElement("Sample Title", "Title"),
                MockElement("Sample paragraph text", "NarrativeText"),
            ]
            mock_partition.return_value = mock_elements

            result = await segment_document(mock_file)

            assert result.success is True
            assert len(result.segments) == 2
            assert result.segments[0].segment_type == SegmentType.TITLE
            assert result.segments[1].segment_type == SegmentType.PARAGRAPH

    @pytest.mark.asyncio
    async def test_controller_error_handling_chain(self):
        """Test error handling through the entire controller chain"""
        mock_file = MockUploadFile("test.txt", b"content")

        # Test various error scenarios
        error_scenarios = [
            ("File not found", FileNotFoundError("File not found")),
            ("Permission denied", PermissionError("Permission denied")),
            ("IO error", OSError("IO error")),
            ("Generic error", Exception("Generic error")),
        ]

        for error_desc, error in error_scenarios:
            with patch(
                "app.controllers.segmentation_controller.SegmentationService"
            ) as mock_service_class:
                mock_service = mock_service_class.return_value
                mock_service.segment_document = AsyncMock(side_effect=error)

                result = await segment_document(mock_file)

                assert result.success is False
                assert error_desc.split()[0].lower() in result.error.lower()

    @pytest.mark.asyncio
    async def test_concurrent_controller_calls(self):
        """Test concurrent calls to controller functions"""
        mock_files = [
            MockUploadFile(f"test_{i}.txt", f"content_{i}".encode()) for i in range(3)
        ]

        with patch(
            "app.controllers.segmentation_controller.SegmentationService"
        ) as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.segment_document = AsyncMock(
                return_value=SegmentationResult(success=True, segments=[])
            )

            # Run concurrent segmentation
            tasks = [segment_document(file) for file in mock_files]
            results = await asyncio.gather(*tasks)

            # All should succeed
            assert len(results) == 3
            for result in results:
                assert result.success is True

            # Service should be called 3 times
            assert mock_service.segment_document.call_count == 3

    @pytest.mark.asyncio
    async def test_controller_resource_cleanup(self):
        """Test that controller properly cleans up resources"""
        mock_file = MockUploadFile("test.txt", b"content")

        # Track file operations
        deleted_files = []

        original_unlink = os.unlink

        def mock_unlink(path):
            deleted_files.append(path)
            if os.path.exists(path):
                original_unlink(path)

        with (
            patch("os.unlink", side_effect=mock_unlink),
            patch(
                "app.controllers.segmentation_controller.SegmentationService"
            ) as mock_service_class,
        ):
            mock_service = mock_service_class.return_value
            mock_service.segment_document = AsyncMock(
                return_value=SegmentationResult(success=True, segments=[])
            )

            result = await segment_document(mock_file)

            # Should have cleaned up at least one temporary file
            assert len(deleted_files) >= 1
            assert result.success is True


class TestControllerErrorCases:
    """Test various error cases for controller functions"""

    @pytest.mark.asyncio
    async def test_segment_document_memory_error(self):
        """Test handling of memory errors"""
        mock_file = MockUploadFile("test.txt", b"content")

        with patch(
            "app.controllers.segmentation_controller.SegmentationService"
        ) as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.segment_document = AsyncMock(
                side_effect=MemoryError("Out of memory")
            )

            result = await segment_document(mock_file)

            assert result.success is False
            assert "Out of memory" in result.error

    @pytest.mark.asyncio
    async def test_segment_document_timeout_error(self):
        """Test handling of timeout errors"""
        mock_file = MockUploadFile("test.txt", b"content")

        with patch(
            "app.controllers.segmentation_controller.SegmentationService"
        ) as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.segment_document = AsyncMock(
                side_effect=TimeoutError("Processing timeout")
            )

            result = await segment_document(mock_file)

            assert result.success is False
            assert "Processing timeout" in result.error

    @pytest.mark.asyncio
    async def test_segment_document_unicode_filename(self):
        """Test handling of unicode filenames"""
        unicode_filename = "测试文档.txt"
        mock_file = MockUploadFile(unicode_filename, b"content")

        with patch(
            "app.controllers.segmentation_controller.SegmentationService"
        ) as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.segment_document = AsyncMock(
                return_value=SegmentationResult(success=True, segments=[])
            )

            result = await segment_document(mock_file)

            assert result.success is True
            mock_service.segment_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_segment_document_special_characters_filename(self):
        """Test handling of special characters in filename"""
        special_filename = r"test@#$%^&*()_+{}|:<>?[]\;',.txt"
        mock_file = MockUploadFile(special_filename, b"content")

        with patch(
            "app.controllers.segmentation_controller.SegmentationService"
        ) as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.segment_document = AsyncMock(
                return_value=SegmentationResult(success=True, segments=[])
            )

            result = await segment_document(mock_file)

            assert result.success is True
            mock_service.segment_document.assert_called_once()
