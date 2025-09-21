# Hugging Face Spaces Testing Suite

This directory contains comprehensive testing tools and scripts specifically designed for validating PolyID deployment on Hugging Face Spaces.

## 🎯 Purpose

This testing suite is separate from the original PolyID unit tests (`../tests/`) and focuses specifically on:
- Hugging Face Spaces deployment validation
- Chemistry stack integration testing
- Performance monitoring and analysis
- End-to-end functionality verification
- Remote API testing

## 🧪 Test Files

### Core Testing Scripts

- **`test_polyid_comprehensive.py`** - Comprehensive end-to-end testing of the complete PolyID pipeline
- **`test_polyid_space_functionality.py`** - Specific HF Spaces functionality and UI testing
- **`test_polyid_deployment.py`** - Deployment configuration and environment testing
- **`test_chemistry_stack.py`** - Chemistry package integration testing (RDKit, NFP, m2p)
- **`test_hf_space_remote.py`** - Remote API testing for deployed Spaces

### Analysis and Monitoring

- **`performance_monitor.py`** - Performance monitoring framework with metrics collection
- **`hf_space_chemistry_analysis.py`** - Chemistry stack analysis and compatibility checking
- **`verify_hf_space_chemistry.py`** - Chemistry component verification script
- **`verify_deployment.py`** - Deployment status and configuration verification

### Specialized Testing

- **`simplified_space_test.py`** - Lightweight testing for quick validation
- **`test_polyid_pipeline.py`** - Pipeline-specific testing and validation

## 🚀 Usage

### Running Individual Tests

```bash
# Comprehensive testing
python hf-spaces-testing/test_polyid_comprehensive.py

# Performance monitoring
python hf-spaces-testing/performance_monitor.py

# Chemistry stack verification
python hf-spaces-testing/verify_hf_space_chemistry.py
```

### Running Full Test Suite

```bash
# Run all HF Spaces tests
pytest hf-spaces-testing/

# Run with verbose output
pytest -v hf-spaces-testing/
```

### Performance Analysis

```bash
# Generate performance report
python hf-spaces-testing/performance_monitor.py --generate-report

# Analyze chemistry stack
python hf-spaces-testing/hf_space_chemistry_analysis.py
```

## 📊 Test Coverage

### Functionality Testing
- ✅ SMILES input validation
- ✅ Molecular graph generation
- ✅ Neural network model loading
- ✅ Property prediction accuracy
- ✅ Multi-model ensemble behavior
- ✅ Confidence estimation
- ✅ Error handling and fallback modes

### Performance Testing
- ✅ Response time measurement
- ✅ Memory usage analysis
- ✅ GPU utilization monitoring
- ✅ Throughput benchmarking
- ✅ Resource efficiency evaluation

### Integration Testing
- ✅ RDKit molecular processing
- ✅ NFP neural fingerprint generation
- ✅ TensorFlow model execution
- ✅ Gradio interface functionality
- ✅ API endpoint validation

## 📋 Generated Reports

Test results and analysis reports are automatically saved to the `../reports/` directory:
- Performance metrics and benchmarks
- Chemistry stack compatibility reports
- Deployment validation results
- Error logs and debugging information

## 🔧 Configuration

### Environment Variables
- `HF_TOKEN` - Hugging Face authentication token for private Spaces
- `POLYID_TEST_MODE` - Set to 'mock' for testing without full chemistry stack

### Test Configuration
Tests can be configured for different deployment scenarios:
- Standard GPU Spaces testing
- ZeroGPU compatibility validation
- Local development environment testing
- Production deployment verification

## 🚦 Test Status Interpretation

### Exit Codes
- `0` - All tests passed successfully
- `1` - Test failures detected
- `2` - Configuration or environment issues
- `3` - Missing dependencies or setup problems

### Mock vs Real Testing
When chemistry dependencies are unavailable, tests automatically switch to mock mode to validate:
- Application structure and flow
- Error handling and graceful degradation
- UI responsiveness and interface elements
- API contract compliance

## 🤝 Contributing

When adding new tests:

1. **Naming Convention**: Use `test_` prefix for pytest compatibility
2. **Documentation**: Include docstrings explaining test purpose and scope
3. **Categorization**: Group related tests in logical modules
4. **Mock Support**: Ensure tests can run in mock mode when dependencies are missing
5. **Reporting**: Generate appropriate reports in `../reports/` directory

## 🔗 Related

- **[Main Tests](../tests/)** - Original PolyID unit tests
- **[Reports](../reports/)** - Generated analysis reports and results
- **[Documentation](../docs/)** - Project documentation and guides

---

*This testing suite ensures robust deployment validation for PolyID on Hugging Face Spaces platforms.*