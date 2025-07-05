# Segmentation Service Test Suite

This directory contains a comprehensive test suite for the document segmentation service that uses unstructured.io for document layout analysis.

## 📋 Test Structure

```
tests/services/segmentation/
├── __init__.py                          # Test module initialization
├── conftest.py                          # Pytest fixtures and configuration
├── test_segmentation_service.py         # Unit tests for SegmentationService
├── test_segmentation_models.py          # Tests for Pydantic models
├── test_segmentation_controller.py      # Tests for controller functions
├── test_segmentation_integration.py     # API integration tests
├── test_segmentation_performance.py     # Performance and stress tests
├── run_segmentation_tests.py           # Test runner script
└── README.md                           # This file
```

## 🧪 Test Categories

### 1. Unit Tests (`test_segmentation_service.py`)
- **Service Initialization**: Tests service setup and configuration
- **Document Processing**: Tests document segmentation logic
- **Image Processing**: Tests image segmentation functionality  
- **Error Handling**: Tests exception handling and edge cases
- **File Type Support**: Tests various file format handling
- **Utility Functions**: Tests helper methods and utilities

**Key Test Classes:**
- `TestSegmentationService`: Core service functionality
- `TestSegmentationServiceIntegration`: Integration scenarios
- `TestSegmentationServiceErrorHandling`: Error conditions
- `TestSegmentationServicePerformance`: Basic performance checks

### 2. Model Tests (`test_segmentation_models.py`)
- **Pydantic Validation**: Tests data model validation
- **Serialization**: Tests JSON serialization/deserialization
- **Type Checking**: Tests enum and type validation
- **Edge Cases**: Tests boundary conditions and invalid inputs

**Key Test Classes:**
- `TestSegmentType`: Enum validation
- `TestBoundingBox`: Coordinate model tests
- `TestDocumentSegment`: Segment model tests
- `TestSegmentationResult`: Result model tests
- `TestModelsIntegration`: Cross-model integration

### 3. Controller Tests (`test_segmentation_controller.py`)
- **File Upload Handling**: Tests file processing and validation
- **Parameter Validation**: Tests API parameter handling
- **Error Response**: Tests error handling and responses
- **Resource Management**: Tests file cleanup and resource handling

**Key Test Classes:**
- `TestSegmentDocument`: Document segmentation endpoint
- `TestGetSupportedFormats`: Supported formats endpoint
- `TestControllerIntegration`: End-to-end scenarios
- `TestControllerErrorCases`: Error handling

### 4. Integration Tests (`test_segmentation_integration.py`)
- **API Endpoints**: Tests FastAPI endpoint integration
- **Request/Response Format**: Tests API contract compliance
- **Concurrent Processing**: Tests multi-user scenarios
- **Performance Metrics**: Tests response times and throughput

**Key Test Classes:**
- `TestSegmentationAPIIntegration`: Core API functionality
- `TestSegmentationAPIResponseFormat`: Response validation
- `TestSegmentationAPIValidation`: Input validation
- `TestSegmentationAPIPerformance`: API performance
- `TestSegmentationAPIEdgeCases`: Edge case handling

### 5. Performance Tests (`test_segmentation_performance.py`)
- **Throughput Testing**: Documents per second metrics
- **Memory Usage**: Memory consumption analysis
- **Concurrent Load**: High-load scenario testing
- **Resource Limits**: Boundary condition testing

**Key Test Classes:**
- `TestSegmentationServicePerformance`: Basic performance metrics
- `TestSegmentationServiceStress`: Stress testing
- `TestSegmentationServiceResourceLimits`: Resource boundary tests

## 🚀 Running Tests

### Quick Start
```bash
# Run all tests
python tests/services/segmentation/run_segmentation_tests.py

# Run specific test categories
python tests/services/segmentation/run_segmentation_tests.py unit
python tests/services/segmentation/run_segmentation_tests.py integration
python tests/services/segmentation/run_segmentation_tests.py performance

# Run with coverage
python tests/services/segmentation/run_segmentation_tests.py all --coverage

# Run in verbose mode
python tests/services/segmentation/run_segmentation_tests.py all --verbose
```

### Using pytest directly
```bash
# Run all segmentation tests
pytest tests/services/segmentation/ -v

# Run specific test file
pytest tests/services/segmentation/test_segmentation_service.py -v

# Run specific test class
pytest tests/services/segmentation/test_segmentation_service.py::TestSegmentationService -v

# Run specific test method
pytest tests/services/segmentation/test_segmentation_service.py::TestSegmentationService::test_init -v

# Run with markers
pytest -m "not performance" tests/services/segmentation/  # Skip performance tests
pytest -m "performance" tests/services/segmentation/      # Only performance tests
```

### Test Dependencies Check
```bash
python tests/services/segmentation/run_segmentation_tests.py --check-deps
```

## 🏷️ Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.asyncio`: Async tests
- `@pytest.mark.performance`: Performance tests  
- `@pytest.mark.stress`: Stress tests
- `@pytest.mark.limit`: Resource limit tests
- `@pytest.mark.benchmark`: Benchmarking tests

## 📊 Test Coverage

The test suite aims for comprehensive coverage:

- **Service Logic**: 95%+ coverage of core segmentation logic
- **Models**: 100% coverage of Pydantic models
- **Controllers**: 90%+ coverage of API controllers
- **Error Paths**: Comprehensive error condition testing
- **Edge Cases**: Boundary and edge case coverage

### Generating Coverage Reports
```bash
# Generate HTML coverage report
pytest --cov=app.services.segmentation_service \
       --cov=app.models.segmentation_models \
       --cov=app.controllers.segmentation_controller \
       --cov-report=html \
       tests/services/segmentation/

# View report
open htmlcov/index.html
```

## 🧰 Test Fixtures

### Core Fixtures (`conftest.py`)
- `segmentation_service`: SegmentationService instance
- `sample_image`: Generated test image with multiple elements
- `sample_pdf_content`: Mock PDF content
- `sample_text_file`: Temporary text file with sample content
- `sample_segments`: Pre-built DocumentSegment objects
- `sample_segmentation_result`: Complete SegmentationResult object
- `mock_unstructured_elements`: Mock unstructured.io elements
- `temp_directory`: Temporary directory for test files
- `large_image`: Large test image for performance testing
- `invalid_image_file`: Corrupted file for error testing
- `empty_file`: Empty file for edge case testing

### Fixture Scopes
- **Function**: Most fixtures (fresh for each test)
- **Class**: Expensive setup shared within test class
- **Session**: One-time setup for entire test session

## 🔧 Configuration

### pytest.ini Configuration
```ini
[tool:pytest]
markers =
    performance: Performance tests
    stress: Stress tests  
    limit: Resource limit tests
    benchmark: Benchmarking tests
    slow: Slow-running tests

asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

### Test Environment Variables
```bash
# Optional: Configure test behavior
export SEGMENTATION_TEST_TIMEOUT=30
export SEGMENTATION_TEST_CONCURRENT_LIMIT=50
export SEGMENTATION_TEST_SKIP_SLOW=true
```

## 📈 Performance Benchmarks

### Benchmark Targets
- **Single Document**: < 2 seconds processing time
- **Concurrent Processing**: > 5 documents/second throughput
- **Memory Usage**: < 100MB increase per document batch
- **API Response Time**: < 5 seconds for standard documents
- **Concurrent API Load**: 80%+ success rate under high load

### Running Benchmarks
```bash
# Run performance tests only
python tests/services/segmentation/run_segmentation_tests.py performance

# Run specific benchmark
pytest -m benchmark tests/services/segmentation/test_segmentation_performance.py
```

## 🐛 Debugging Tests

### Common Issues
1. **Unstructured.io Import Errors**: Ensure unstructured is installed
2. **Async Test Failures**: Check pytest-asyncio is installed
3. **Timeout Issues**: Increase timeout for slow systems
4. **Memory Errors**: Run fewer concurrent tests
5. **File Permission Issues**: Check temp directory permissions

### Debug Commands
```bash
# Run with detailed output
pytest -vvv --tb=long tests/services/segmentation/

# Run single test with debugging
pytest -s tests/services/segmentation/test_segmentation_service.py::TestSegmentationService::test_init

# Run with pdb debugging
pytest --pdb tests/services/segmentation/test_segmentation_service.py
```

## 🔍 Test Data

### Generated Test Data
- **Images**: Programmatically generated with PIL
- **Documents**: Template-based text content
- **PDFs**: Mock PDF structures (when reportlab available)
- **Binary Files**: Byte sequences for edge testing

### External Test Data
- **Sample Images**: Place in `tests/data/images/`
- **Sample Documents**: Place in `tests/data/documents/`
- **Expected Results**: JSON files in `tests/data/expected/`

## 📝 Writing New Tests

### Test Naming Convention
```python
def test_[component]_[scenario]_[expected_outcome]:
    """
    Clear description of what is being tested
    """
    # Arrange
    # Act  
    # Assert
```

### Example Test Structure
```python
@pytest.mark.asyncio
async def test_segment_document_success(self, segmentation_service, sample_text_file):
    """Test successful document segmentation"""
    # Arrange
    expected_segments = 2
    
    # Act
    result = await segmentation_service.segment_document(sample_text_file)
    
    # Assert
    assert result.success is True
    assert len(result.segments) == expected_segments
    assert result.total_segments == expected_segments
```

### Best Practices
1. **Use descriptive test names**
2. **Include docstrings explaining test purpose**
3. **Follow Arrange-Act-Assert pattern**
4. **Mock external dependencies**
5. **Test both success and failure paths**
6. **Include edge cases and boundary conditions**
7. **Use appropriate fixtures**
8. **Add performance markers for slow tests**

## 🛠️ Maintenance

### Updating Tests
- **New Features**: Add corresponding test coverage
- **Bug Fixes**: Add regression tests
- **API Changes**: Update integration tests
- **Performance Changes**: Update benchmark expectations

### Test Review Checklist
- [ ] All new code has test coverage
- [ ] Tests are properly categorized with markers
- [ ] Performance tests have realistic expectations
- [ ] Error conditions are tested
- [ ] Documentation is updated
- [ ] Tests pass in CI/CD pipeline

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Pydantic Testing](https://pydantic-docs.helpmanual.io/usage/validation_errors/)
- [unstructured.io Documentation](https://unstructured-io.github.io/unstructured/)

---

## 🤝 Contributing

When contributing to the test suite:

1. **Run existing tests** to ensure no regressions
2. **Add tests for new functionality**
3. **Follow established patterns and conventions**
4. **Update documentation** as needed
5. **Consider performance implications** of new tests

For questions or issues with the test suite, please refer to the project documentation or open an issue. 