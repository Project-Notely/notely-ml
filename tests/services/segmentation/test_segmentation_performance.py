"""
Performance tests for segmentation service
"""

import asyncio
import gc
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, patch

import numpy as np
import psutil
import pytest
from PIL import Image

from app.models.segmentation_models import (
    DocumentSegment,
    SegmentationResult,
    SegmentType,
)
from app.services.segmentation_service import SegmentationService


class TestSegmentationServicePerformance:
    """Performance tests for SegmentationService"""

    @pytest.fixture
    def segmentation_service(self):
        """Create segmentation service instance"""
        return SegmentationService()

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_document_processing_time(
        self, segmentation_service, sample_text_file
    ):
        """Test processing time for a single document"""
        with patch("app.services.segmentation_service.partition") as mock_partition:
            mock_partition.return_value = []

            start_time = time.time()
            result = await segmentation_service.segment_document(sample_text_file)
            end_time = time.time()

            processing_time = end_time - start_time

            assert result.success is True
            # Should process within reasonable time
            assert processing_time < 2.0  # 2 seconds for mocked service

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_multiple_documents_sequential(
        self, segmentation_service, sample_text_file
    ):
        """Test sequential processing of multiple documents"""
        with patch("app.services.segmentation_service.partition") as mock_partition:
            mock_partition.return_value = []

            num_documents = 5
            start_time = time.time()

            for i in range(num_documents):
                result = await segmentation_service.segment_document(sample_text_file)
                assert result.success is True

            end_time = time.time()
            total_time = end_time - start_time
            avg_time_per_doc = total_time / num_documents

            # Average time per document should be reasonable
            assert avg_time_per_doc < 2.0
            assert total_time < 10.0  # Total time reasonable

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_multiple_documents_concurrent(
        self, segmentation_service, sample_text_file
    ):
        """Test concurrent processing of multiple documents"""
        with patch("app.services.segmentation_service.partition") as mock_partition:
            mock_partition.return_value = []

            num_documents = 5
            start_time = time.time()

            # Process documents concurrently
            tasks = [
                segmentation_service.segment_document(sample_text_file)
                for _ in range(num_documents)
            ]

            results = await asyncio.gather(*tasks)

            end_time = time.time()
            total_time = end_time - start_time

            # All should succeed
            assert all(result.success for result in results)

            # Concurrent processing should be faster than sequential
            # (allowing for some overhead)
            assert (
                total_time < 7.0
            )  # Should be significantly faster than 10s sequential

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_image_processing(self, segmentation_service, large_image):
        """Test processing of large images"""
        with patch("app.services.segmentation_service.partition") as mock_partition:
            mock_partition.return_value = []

            start_time = time.time()
            result = await segmentation_service.segment_image(large_image)
            end_time = time.time()

            processing_time = end_time - start_time

            assert result.success is True
            # Large image should still process in reasonable time
            assert processing_time < 10.0

    @pytest.mark.performance
    def test_memory_usage_single_document(self, segmentation_service, sample_text_file):
        """Test memory usage for single document processing"""
        process = psutil.Process(os.getpid())

        # Measure initial memory
        gc.collect()  # Force garbage collection
        initial_memory = process.memory_info().rss

        with patch("app.services.segmentation_service.partition") as mock_partition:
            mock_partition.return_value = []

            # Process document
            asyncio.run(segmentation_service.segment_document(sample_text_file))

        # Measure final memory
        gc.collect()
        final_memory = process.memory_info().rss

        memory_increase = final_memory - initial_memory

        # Memory increase should be minimal for mocked processing
        assert memory_increase < 50 * 1024 * 1024  # Less than 50MB

    @pytest.mark.performance
    def test_memory_usage_multiple_documents(
        self, segmentation_service, sample_text_file
    ):
        """Test memory usage for multiple document processing"""
        process = psutil.Process(os.getpid())

        gc.collect()
        initial_memory = process.memory_info().rss

        with patch("app.services.segmentation_service.partition") as mock_partition:
            mock_partition.return_value = []

            # Process multiple documents
            async def process_multiple():
                for i in range(10):
                    await segmentation_service.segment_document(sample_text_file)

            asyncio.run(process_multiple())

        gc.collect()
        final_memory = process.memory_info().rss

        memory_increase = final_memory - initial_memory

        # Memory should not grow excessively with multiple documents
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_cleanup_after_processing(self, segmentation_service):
        """Test that memory is properly cleaned up after processing"""
        process = psutil.Process(os.getpid())

        with patch("app.services.segmentation_service.partition") as mock_partition:
            mock_partition.return_value = []

            memory_measurements = []

            for i in range(5):
                gc.collect()
                memory_before = process.memory_info().rss

                # Create a large temporary image
                large_image = Image.new("RGB", (1000, 1000), color="white")
                result = await segmentation_service.segment_image(large_image)
                assert result.success is True

                # Delete the image
                del large_image

                gc.collect()
                memory_after = process.memory_info().rss

                memory_measurements.append(memory_after - memory_before)

            # Memory usage should be relatively consistent (not growing significantly)
            avg_memory_use = sum(memory_measurements) / len(memory_measurements)
            max_memory_use = max(memory_measurements)

            # Max should not be significantly higher than average (no major leaks)
            assert max_memory_use < avg_memory_use * 2

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cpu_usage_during_processing(
        self, segmentation_service, sample_text_file
    ):
        """Test CPU usage during processing"""
        process = psutil.Process(os.getpid())

        with patch("app.services.segmentation_service.partition") as mock_partition:
            # Make the mock take some time to simulate processing
            async def slow_partition(*args, **kwargs):
                await asyncio.sleep(0.1)  # Simulate processing time
                return []

            mock_partition.side_effect = slow_partition

            # Start monitoring CPU
            cpu_percentages = []
            monitoring = True

            def monitor_cpu():
                while monitoring:
                    cpu_percentages.append(process.cpu_percent())
                    time.sleep(0.05)

            # Start CPU monitoring in background
            monitor_thread = threading.Thread(target=monitor_cpu)
            monitor_thread.start()

            # Process document
            result = await segmentation_service.segment_document(sample_text_file)

            # Stop monitoring
            monitoring = False
            monitor_thread.join()

            assert result.success is True

            # CPU usage should be reasonable (not pegged at 100%)
            if cpu_percentages:
                avg_cpu = sum(cpu_percentages) / len(cpu_percentages)
                max_cpu = max(cpu_percentages)

                # Should not constantly use 100% CPU
                assert avg_cpu < 50.0
                assert max_cpu < 100.0

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_throughput_documents_per_second(
        self, segmentation_service, sample_text_file
    ):
        """Test throughput in documents per second"""
        with patch("app.services.segmentation_service.partition") as mock_partition:
            mock_partition.return_value = []

            num_documents = 20
            start_time = time.time()

            # Process documents concurrently
            tasks = [
                segmentation_service.segment_document(sample_text_file)
                for _ in range(num_documents)
            ]

            results = await asyncio.gather(*tasks)

            end_time = time.time()
            total_time = end_time - start_time

            # Calculate throughput
            throughput = num_documents / total_time

            assert all(result.success for result in results)
            # Should achieve reasonable throughput
            assert (
                throughput > 5.0
            )  # At least 5 documents per second for mocked service

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_scalability_with_document_size(self, segmentation_service):
        """Test how performance scales with document size"""
        with patch("app.services.segmentation_service.partition") as mock_partition:
            mock_partition.return_value = []

            # Test different document sizes
            sizes = [1024, 10240, 102400, 1024000]  # 1KB, 10KB, 100KB, 1MB
            processing_times = []

            for size in sizes:
                # Create temporary file of specific size
                content = b"x" * size
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".txt"
                ) as tmp_file:
                    tmp_file.write(content)
                    tmp_path = tmp_file.name

                try:
                    start_time = time.time()
                    result = await segmentation_service.segment_document(tmp_path)
                    end_time = time.time()

                    processing_time = end_time - start_time
                    processing_times.append(processing_time)

                    assert result.success is True

                finally:
                    os.unlink(tmp_path)

            # Processing time should scale reasonably with size
            # (not exponentially for mocked service)
            for i in range(1, len(processing_times)):
                # Each step up should not be more than 10x slower
                assert processing_times[i] < processing_times[i - 1] * 10

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_load_handling(
        self, segmentation_service, sample_text_file
    ):
        """Test handling of high concurrent load"""
        with patch("app.services.segmentation_service.partition") as mock_partition:
            mock_partition.return_value = []

            # Simulate high load with many concurrent requests
            num_concurrent = 50

            start_time = time.time()

            # Create many concurrent tasks
            tasks = [
                segmentation_service.segment_document(sample_text_file)
                for _ in range(num_concurrent)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.time()
            total_time = end_time - start_time

            # Count successful results
            successful_results = [
                r for r in results if isinstance(r, SegmentationResult) and r.success
            ]
            failed_results = [
                r
                for r in results
                if not (isinstance(r, SegmentationResult) and r.success)
            ]

            # Most requests should succeed
            success_rate = len(successful_results) / len(results)
            assert success_rate > 0.8  # At least 80% success rate

            # Should handle load in reasonable time
            assert total_time < 30.0  # Complete within 30 seconds

    @pytest.mark.performance
    def test_thread_safety(self, segmentation_service, sample_text_file):
        """Test thread safety of the service"""
        with patch("app.services.segmentation_service.partition") as mock_partition:
            mock_partition.return_value = []

            results = []
            exceptions = []

            def process_document():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(
                        segmentation_service.segment_document(sample_text_file)
                    )
                    results.append(result)
                    loop.close()
                except Exception as e:
                    exceptions.append(e)

            # Create multiple threads
            threads = []
            num_threads = 10

            for i in range(num_threads):
                thread = threading.Thread(target=process_document)
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            # Should not have any exceptions from threading issues
            assert len(exceptions) == 0
            assert len(results) == num_threads
            assert all(result.success for result in results)


class TestSegmentationServiceStress:
    """Stress tests for segmentation service"""

    @pytest.fixture
    def segmentation_service(self):
        """Create segmentation service instance"""
        return SegmentationService()

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_long_running_processing(
        self, segmentation_service, sample_text_file
    ):
        """Test service behavior during long-running processing"""
        with patch("app.services.segmentation_service.partition") as mock_partition:
            mock_partition.return_value = []

            # Run for extended period
            duration = 60  # 1 minute
            start_time = time.time()
            processed_count = 0

            while time.time() - start_time < duration:
                result = await segmentation_service.segment_document(sample_text_file)
                assert result.success is True
                processed_count += 1

                # Brief pause to avoid overwhelming
                await asyncio.sleep(0.1)

            # Should process many documents without degradation
            assert (
                processed_count > 100
            )  # Should process at least 100 documents in 1 minute

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, segmentation_service):
        """Test behavior under memory pressure"""
        with patch("app.services.segmentation_service.partition") as mock_partition:
            mock_partition.return_value = []

            # Create many large images in succession
            for i in range(20):
                # Create progressively larger images
                size = 500 + i * 100  # Start at 500x500, grow to 2400x2400
                large_image = Image.new(
                    "RGB", (size, size), color=f"rgb({i*10}, {i*10}, {i*10})"
                )

                result = await segmentation_service.segment_image(large_image)

                # Should handle even under memory pressure
                assert result.success is True

                # Clean up immediately
                del large_image
                gc.collect()

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_rapid_successive_requests(
        self, segmentation_service, sample_text_file
    ):
        """Test handling of rapid successive requests"""
        with patch("app.services.segmentation_service.partition") as mock_partition:
            mock_partition.return_value = []

            # Make rapid requests without waiting
            num_requests = 100

            start_time = time.time()

            tasks = []
            for i in range(num_requests):
                task = segmentation_service.segment_document(sample_text_file)
                tasks.append(task)

                # Very small delay between requests
                if i % 10 == 0:
                    await asyncio.sleep(0.01)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.time()
            total_time = end_time - start_time

            # Count successful results
            successful = [
                r for r in results if isinstance(r, SegmentationResult) and r.success
            ]

            # Should handle most requests successfully
            success_rate = len(successful) / len(results)
            assert success_rate > 0.7  # At least 70% success under stress

            # Should complete in reasonable time
            assert total_time < 60.0  # Within 1 minute


class TestSegmentationServiceResourceLimits:
    """Test service behavior at resource limits"""

    @pytest.fixture
    def segmentation_service(self):
        """Create segmentation service instance"""
        return SegmentationService()

    @pytest.mark.limit
    @pytest.mark.asyncio
    async def test_maximum_file_size_handling(self, segmentation_service):
        """Test handling of very large files"""
        with patch("app.services.segmentation_service.partition") as mock_partition:
            mock_partition.return_value = []

            # Create very large file (10MB)
            large_content = b"x" * (10 * 1024 * 1024)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                tmp_file.write(large_content)
                tmp_path = tmp_file.name

            try:
                start_time = time.time()
                result = await segmentation_service.segment_document(tmp_path)
                end_time = time.time()

                processing_time = end_time - start_time

                # Should handle or fail gracefully
                assert isinstance(result, SegmentationResult)

                if result.success:
                    # If successful, should complete in reasonable time
                    assert processing_time < 30.0
                else:
                    # If failed, should have meaningful error
                    assert result.error is not None

            finally:
                os.unlink(tmp_path)

    @pytest.mark.limit
    @pytest.mark.asyncio
    async def test_maximum_concurrent_connections(
        self, segmentation_service, sample_text_file
    ):
        """Test maximum concurrent connections handling"""
        with patch("app.services.segmentation_service.partition") as mock_partition:
            mock_partition.return_value = []

            # Test with very high concurrency
            max_concurrent = 200

            tasks = [
                segmentation_service.segment_document(sample_text_file)
                for _ in range(max_concurrent)
            ]

            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()

            total_time = end_time - start_time

            # Should complete or fail gracefully
            successful = [
                r for r in results if isinstance(r, SegmentationResult) and r.success
            ]

            # Some percentage should succeed
            success_rate = len(successful) / len(results)
            assert success_rate > 0.3  # At least 30% under extreme load

            # Should not hang indefinitely
            assert total_time < 120.0  # Complete within 2 minutes


# Performance benchmarking utilities
class PerformanceBenchmark:
    """Utility class for performance benchmarking"""

    def __init__(self):
        self.results = {}

    def benchmark_function(self, func_name: str, func, *args, **kwargs):
        """Benchmark a function call"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss

        self.results[func_name] = {
            "duration": end_time - start_time,
            "memory_delta": end_memory - start_memory,
            "result": result,
        }

        return result

    def get_summary(self):
        """Get performance summary"""
        summary = {}
        for func_name, metrics in self.results.items():
            summary[func_name] = {
                "avg_duration": metrics["duration"],
                "memory_usage_mb": metrics["memory_delta"] / (1024 * 1024),
            }
        return summary


@pytest.mark.benchmark
def test_performance_benchmark_suite(sample_text_file, sample_image):
    """Comprehensive performance benchmark suite"""
    service = SegmentationService()
    benchmark = PerformanceBenchmark()

    with patch("app.services.segmentation_service.partition") as mock_partition:
        mock_partition.return_value = []

        # Benchmark different operations
        async def run_benchmarks():
            # Single document
            await benchmark.benchmark_function(
                "single_document", service.segment_document, sample_text_file
            )

            # Single image
            await benchmark.benchmark_function(
                "single_image", service.segment_image, sample_image
            )

            # Multiple documents concurrent
            tasks = [service.segment_document(sample_text_file) for _ in range(5)]
            await benchmark.benchmark_function(
                "concurrent_documents", asyncio.gather, *tasks
            )

        asyncio.run(run_benchmarks())

        # Get and verify benchmark results
        summary = benchmark.get_summary()

        # All operations should complete in reasonable time
        for operation, metrics in summary.items():
            assert metrics["avg_duration"] < 10.0  # Less than 10 seconds
            assert metrics["memory_usage_mb"] < 100  # Less than 100MB memory increase
