"""
Integration tests for segmentation API endpoints
"""
import pytest
import asyncio
from unittest.mock import patch, AsyncMock
import tempfile
import os
from pathlib import Path
import json
from fastapi.testclient import TestClient
from fastapi import UploadFile
import io

from app.main import app
from app.models.segmentation_models import SegmentationResult, DocumentSegment, SegmentType, BoundingBox


class TestSegmentationAPIIntegration:
    """Integration tests for segmentation API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_file_content(self):
        """Sample file content for testing"""
        return b"Sample document content for testing segmentation"
    
    def test_segment_endpoint_success(self, client, sample_file_content, sample_segmentation_result):
        """Test successful segmentation via API endpoint"""
        with patch('app.services.segmentation_service.partition') as mock_partition:
            # Mock unstructured elements
            class MockElement:
                def __init__(self, text, category):
                    self.text = text
                    self.category = category
                    self.metadata = {}
                
                def __str__(self):
                    return self.text
            
            mock_elements = [
                MockElement("Sample Title", "Title"),
                MockElement("Sample paragraph", "NarrativeText")
            ]
            mock_partition.return_value = mock_elements
            
            # Create test file
            files = {"file": ("test.txt", sample_file_content, "text/plain")}
            
            response = client.post("/api/v1/segment", files=files)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert len(data["segments"]) == 2
            assert data["segments"][0]["segment_type"] == "title"
            assert data["segments"][1]["segment_type"] == "paragraph"
            assert data["total_segments"] == 2
    
    def test_segment_endpoint_with_params(self, client, sample_file_content):
        """Test segmentation endpoint with different parameters"""
        with patch('app.services.segmentation_service.partition') as mock_partition:
            mock_partition.return_value = []
            
            files = {"file": ("test.txt", sample_file_content, "text/plain")}
            params = {
                "strategy": "fast",
                "extract_images": False,
                "infer_table_structure": False
            }
            
            response = client.post("/api/v1/segment", files=files, params=params)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
    
    def test_segment_endpoint_no_file(self, client):
        """Test segmentation endpoint without file"""
        response = client.post("/api/v1/segment")
        
        assert response.status_code == 422  # Validation error
    
    def test_segment_endpoint_invalid_strategy(self, client, sample_file_content):
        """Test segmentation endpoint with invalid strategy"""
        files = {"file": ("test.txt", sample_file_content, "text/plain")}
        params = {"strategy": "invalid_strategy"}
        
        # This should still work as the service handles invalid strategies
        response = client.post("/api/v1/segment", files=files, params=params)
        
        # The response might succeed if the service defaults to a valid strategy
        assert response.status_code in [200, 422]
    
    def test_segment_endpoint_large_file(self, client):
        """Test segmentation endpoint with large file"""
        # Create a large file (1MB)
        large_content = b"x" * (1024 * 1024)
        
        with patch('app.services.segmentation_service.partition') as mock_partition:
            mock_partition.return_value = []
            
            files = {"file": ("large_test.txt", large_content, "text/plain")}
            
            response = client.post("/api/v1/segment", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
    
    def test_segment_endpoint_different_file_types(self, client):
        """Test segmentation endpoint with different file types"""
        file_types = [
            ("test.txt", b"Text content", "text/plain"),
            ("test.pdf", b"%PDF-1.4 fake pdf", "application/pdf"),
            ("test.png", b"\x89PNG fake png", "image/png"),
            ("test.docx", b"PK fake docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        ]
        
        for filename, content, mime_type in file_types:
            with patch('app.services.segmentation_service.partition') as mock_partition:
                mock_partition.return_value = []
                
                files = {"file": (filename, content, mime_type)}
                
                response = client.post("/api/v1/segment", files=files)
                
                # Some file types might not be supported
                assert response.status_code in [200, 422]
                
                if response.status_code == 200:
                    data = response.json()
                    assert "success" in data
    
    def test_segment_endpoint_service_error(self, client, sample_file_content):
        """Test segmentation endpoint when service returns error"""
        with patch('app.services.segmentation_service.partition') as mock_partition:
            mock_partition.side_effect = Exception("Service error")
            
            files = {"file": ("test.txt", sample_file_content, "text/plain")}
            
            response = client.post("/api/v1/segment", files=files)
            
            assert response.status_code == 200  # Controller handles errors gracefully
            data = response.json()
            assert data["success"] is False
            assert "error" in data
    
    def test_supported_formats_endpoint(self, client):
        """Test supported formats endpoint"""
        response = client.get("/api/v1/supported-formats")
        
        assert response.status_code == 200
        formats = response.json()
        
        assert isinstance(formats, list)
        assert len(formats) > 0
        assert ".txt" in formats or ".pdf" in formats  # Should have some formats
    
    def test_segment_endpoint_concurrent_requests(self, client, sample_file_content):
        """Test concurrent requests to segmentation endpoint"""
        import threading
        import time
        
        results = []
        
        def make_request():
            with patch('app.services.segmentation_service.partition') as mock_partition:
                mock_partition.return_value = []
                
                files = {"file": ("test.txt", sample_file_content, "text/plain")}
                response = client.post("/api/v1/segment", files=files)
                results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert len(results) == 3
        assert all(status == 200 for status in results)


class TestSegmentationAPIResponseFormat:
    """Test API response format compliance"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_segment_response_format(self, client):
        """Test that segment response follows expected format"""
        with patch('app.services.segmentation_service.partition') as mock_partition:
            # Create mock element with coordinates
            class MockElement:
                def __init__(self, text, category):
                    self.text = text
                    self.category = category
                    self.metadata = {
                        "coordinates": type('obj', (object,), {
                            "points": [(10, 20), (100, 80)]
                        })()
                    }
                
                def __str__(self):
                    return self.text
            
            mock_elements = [MockElement("Test title", "Title")]
            mock_partition.return_value = mock_elements
            
            files = {"file": ("test.txt", b"content", "text/plain")}
            response = client.post("/api/v1/segment", files=files)
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            required_fields = ["success", "segments", "total_segments", "statistics"]
            for field in required_fields:
                assert field in data
            
            # Verify segment structure
            if data["segments"]:
                segment = data["segments"][0]
                segment_fields = ["text", "segment_type", "confidence"]
                for field in segment_fields:
                    assert field in segment
                
                # Check if bbox is present and has correct structure
                if "bbox" in segment and segment["bbox"]:
                    bbox_fields = ["x", "y", "width", "height"]
                    for field in bbox_fields:
                        assert field in segment["bbox"]
    
    def test_segment_response_error_format(self, client):
        """Test error response format"""
        with patch('app.services.segmentation_service.partition') as mock_partition:
            mock_partition.side_effect = Exception("Test error")
            
            files = {"file": ("test.txt", b"content", "text/plain")}
            response = client.post("/api/v1/segment", files=files)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is False
            assert "error" in data
            assert "segments" in data
            assert data["segments"] == []
    
    def test_supported_formats_response_format(self, client):
        """Test supported formats response format"""
        response = client.get("/api/v1/supported-formats")
        
        assert response.status_code == 200
        formats = response.json()
        
        assert isinstance(formats, list)
        for format_str in formats:
            assert isinstance(format_str, str)
            assert format_str.startswith(".")  # Should be file extensions


class TestSegmentationAPIValidation:
    """Test API input validation"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_segment_file_validation(self, client):
        """Test file validation"""
        # Test with no file
        response = client.post("/api/v1/segment")
        assert response.status_code == 422
        
        # Test with empty file
        files = {"file": ("", b"", "text/plain")}
        response = client.post("/api/v1/segment", files=files)
        assert response.status_code in [200, 422]  # Depends on implementation
    
    def test_segment_parameter_validation(self, client):
        """Test parameter validation"""
        files = {"file": ("test.txt", b"content", "text/plain")}
        
        # Test valid boolean parameters
        valid_params = [
            {"extract_images": True},
            {"extract_images": False},
            {"infer_table_structure": True},
            {"infer_table_structure": False},
        ]
        
        for params in valid_params:
            with patch('app.services.segmentation_service.partition') as mock_partition:
                mock_partition.return_value = []
                response = client.post("/api/v1/segment", files=files, params=params)
                assert response.status_code == 200
        
        # Test invalid boolean parameters
        invalid_params = [
            {"extract_images": "not_a_boolean"},
            {"infer_table_structure": "invalid"},
        ]
        
        for params in invalid_params:
            response = client.post("/api/v1/segment", files=files, params=params)
            assert response.status_code == 422


class TestSegmentationAPIPerformance:
    """Performance tests for segmentation API"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_segment_response_time(self, client):
        """Test API response time"""
        import time
        
        with patch('app.services.segmentation_service.partition') as mock_partition:
            mock_partition.return_value = []
            
            files = {"file": ("test.txt", b"content", "text/plain")}
            
            start_time = time.time()
            response = client.post("/api/v1/segment", files=files)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            assert response.status_code == 200
            # API should respond within reasonable time (adjust as needed)
            assert response_time < 5.0  # 5 seconds max
    
    def test_segment_memory_usage(self, client):
        """Test memory usage during processing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        with patch('app.services.segmentation_service.partition') as mock_partition:
            mock_partition.return_value = []
            
            # Process multiple files
            for i in range(5):
                files = {"file": (f"test_{i}.txt", b"content" * 1000, "text/plain")}
                response = client.post("/api/v1/segment", files=files)
                assert response.status_code == 200
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024
    
    def test_segment_with_large_response(self, client):
        """Test handling of large response data"""
        with patch('app.services.segmentation_service.partition') as mock_partition:
            # Create many mock elements
            class MockElement:
                def __init__(self, text, category):
                    self.text = text
                    self.category = category
                    self.metadata = {}
                
                def __str__(self):
                    return self.text
            
            # Create 100 mock elements
            mock_elements = [
                MockElement(f"Sample text {i}", "NarrativeText")
                for i in range(100)
            ]
            mock_partition.return_value = mock_elements
            
            files = {"file": ("test.txt", b"content", "text/plain")}
            response = client.post("/api/v1/segment", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["segments"]) == 100


class TestSegmentationAPIEdgeCases:
    """Test edge cases and unusual scenarios"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_segment_empty_file(self, client):
        """Test segmentation of empty file"""
        with patch('app.services.segmentation_service.partition') as mock_partition:
            mock_partition.return_value = []
            
            files = {"file": ("empty.txt", b"", "text/plain")}
            response = client.post("/api/v1/segment", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert "success" in data
    
    def test_segment_unicode_content(self, client):
        """Test segmentation of unicode content"""
        unicode_content = "测试文档内容 with émojis 🚀".encode('utf-8')
        
        with patch('app.services.segmentation_service.partition') as mock_partition:
            mock_partition.return_value = []
            
            files = {"file": ("unicode.txt", unicode_content, "text/plain")}
            response = client.post("/api/v1/segment", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert "success" in data
    
    def test_segment_binary_content(self, client):
        """Test segmentation of binary content"""
        binary_content = bytes(range(256))
        
        with patch('app.services.segmentation_service.partition') as mock_partition:
            mock_partition.return_value = []
            
            files = {"file": ("binary.bin", binary_content, "application/octet-stream")}
            response = client.post("/api/v1/segment", files=files)
            
            # Should handle gracefully (either succeed or fail with proper error)
            assert response.status_code in [200, 422]
    
    def test_segment_special_filename(self, client):
        """Test segmentation with special characters in filename"""
        special_filename = "test@#$%^&*()_+{}|:<>?[]\\;',.txt"
        
        with patch('app.services.segmentation_service.partition') as mock_partition:
            mock_partition.return_value = []
            
            files = {"file": (special_filename, b"content", "text/plain")}
            response = client.post("/api/v1/segment", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert "success" in data 