# Testing Framework

This directory contains all test files for the LAMMPS input generation project. The testing framework follows best practices for Python testing with pytest and includes comprehensive test coverage for all components.

## 📁 Directory Structure

### 🧪 Unit Tests
- **`unit/`** (✅ Exists): Unit tests for individual components
  - Test individual functions and classes in isolation
  - Mock external dependencies
  - Fast execution for development workflow

### 🔗 Integration Tests (🔧 Planned)
- **`integration/`**: Integration tests for component interactions
  - Test component interactions and data flow
  - End-to-end workflow testing
  - RAG system integration testing

### 🚀 System Tests (🔧 Planned)
- **`system/`**: Full system testing
  - Complete LAMMPS generation pipeline testing
  - Performance and load testing
  - Real LAMMPS execution validation

### 📊 Test Data (🔧 Planned)
- **`test_data/`**: Test datasets and fixtures
  - Sample LAMMPS scripts for testing
  - Mock documentation for RAG testing
  - Expected outputs for validation

## 🏗️ Testing Framework

### Test Categories
1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Component interaction testing
3. **System Tests**: Full pipeline testing
4. **Performance Tests**: Speed and resource usage testing

### Testing Standards
- **Coverage Target**: 90%+ code coverage
- **Test Naming**: `test_<component>_<functionality>.py`
- **Fixtures**: Reusable test data and mock objects
- **Assertions**: Clear, descriptive assertions

## 🚀 Running Tests

### All Tests
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Specific Test Categories
```bash
# Run only unit tests
python -m pytest tests/unit/

# Run integration tests (when implemented)
python -m pytest tests/integration/

# Run system tests (when implemented)
python -m pytest tests/system/
```

### Specific Components
```bash
# Test LAMMPS generators
python -m pytest tests/unit/test_lammps_generators.py

# Test RAG system (when implemented)
python -m pytest tests/unit/test_rag_system.py

# Test model components (when implemented)
python -m pytest tests/unit/test_model_components.py
```

## 🔧 Test Configuration

### pytest Configuration
Located in `pytest.ini` or `pyproject.toml`:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --strict-markers
markers =
    unit: Unit tests
    integration: Integration tests
    system: System tests
    slow: Slow running tests
```

### Fixtures
Common test fixtures for reusable test data:
```python
# conftest.py
import pytest

@pytest.fixture
def sample_lammps_script():
    return """
    # Sample LAMMPS script for testing
    units real
    dimension 3
    boundary p p p
    atom_style full
    """

@pytest.fixture
def mock_rag_documents():
    return ["LAMMPS documentation text...", "More docs..."]
```

## 📝 Writing Tests

### Unit Test Example
```python
# tests/unit/test_lammps_generators.py
import pytest
from src.lammps_generators.md_generators import NVTGenerator

class TestNVTGenerator:
    def test_generate_basic_nvt(self, sample_lammps_script):
        generator = NVTGenerator()
        result = generator.generate("Create NVT simulation")
        
        assert "fix nvt" in result
        assert "run" in result
        assert result.strip().endswith("run 10000")
    
    def test_invalid_input_handling(self):
        generator = NVTGenerator()
        
        with pytest.raises(ValueError):
            generator.generate("")
```

### Integration Test Example
```python
# tests/integration/test_rag_pipeline.py
import pytest
from src.rag.document_processing import DocumentProcessor
from src.rag.retrieval import SimilaritySearch

class TestRAGPipeline:
    def test_document_to_generation_pipeline(self, mock_rag_documents):
        processor = DocumentProcessor()
        retriever = SimilaritySearch()
        
        # Test full pipeline
        processed_docs = processor.process(mock_rag_documents)
        results = retriever.search("NVT simulation", processed_docs)
        
        assert len(results) > 0
        assert "simulation" in results[0].lower()
```

## 🎯 Testing Checklist

### Before Submitting Code
- [ ] All new functions have unit tests
- [ ] Integration tests cover component interactions
- [ ] Tests pass locally with 90%+ coverage
- [ ] No test dependencies on external services
- [ ] Test data is properly mocked or fixtures
- [ ] Performance tests for critical paths

### Test Quality Standards
- [ ] Tests are deterministic (no random failures)
- [ ] Tests are isolated (no test dependencies)
- [ ] Tests are fast (unit tests < 1s each)
- [ ] Tests are readable and well-documented
- [ ] Edge cases and error conditions covered

## 🔍 Testing Tools

### Core Testing Framework
- **pytest**: Main testing framework
- **pytest-cov**: Code coverage reporting
- **pytest-mock**: Enhanced mocking capabilities
- **pytest-xdist**: Parallel test execution

### Additional Tools
- **factory_boy**: Test data factories
- **responses**: HTTP request mocking
- **freezegun**: Time/date mocking
- **pytest-benchmark**: Performance testing

## 📊 Coverage Reporting

### Generate Coverage Reports
```bash
# HTML coverage report
python -m pytest tests/ --cov=src --cov-report=html

# Terminal coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing

# XML coverage for CI/CD
python -m pytest tests/ --cov=src --cov-report=xml
```

### Coverage Targets
- **Overall**: 90%+ coverage
- **Critical Components**: 95%+ coverage
- **New Code**: 100% coverage required

## 🚀 Continuous Integration

### GitHub Actions (Example)
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: python -m pytest tests/ --cov=src
```

## 🔧 Development Workflow

### Test-Driven Development
1. **Write Test First**: Define expected behavior
2. **Run Test**: Verify it fails (red)
3. **Write Code**: Implement minimal functionality
4. **Run Test**: Verify it passes (green)
5. **Refactor**: Improve code while keeping tests green

### Adding New Tests
1. Identify the component to test
2. Create appropriate test file in `tests/unit/`
3. Write comprehensive test cases
4. Ensure tests pass and add coverage
5. Update this README if needed

## 🚨 Important Notes

1. **Test Data**: Use fixtures and mocks instead of real external data
2. **Isolation**: Tests should not depend on each other
3. **Speed**: Keep unit tests fast for rapid development
4. **Coverage**: Aim for high coverage but focus on quality over quantity
5. **Documentation**: Test names should clearly describe what they test

---

For more information on testing best practices, see the project documentation in `docs/`. 