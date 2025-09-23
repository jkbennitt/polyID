# Performance Testing Suite

Comprehensive performance benchmarking for the PolyID Hugging Face Space.

## Features

- Cold start time measurement
- Concurrent user load testing
- Stress testing with various request rates
- Response time analysis
- Memory usage profiling
- Performance visualization

## Running Tests

```bash
cd performance_tests
python load_testing_suite.py
```

## Output

- Performance plots: `performance_plots/`
  - `performance_overview.png` - High-level metrics
  - `detailed_analysis.png` - Detailed breakdowns
- Performance report: `performance_report.md`

## Metrics Measured

- Cold start latency
- Prediction response times
- Concurrent user capacity
- Request throughput
- Error rates
- System resource utilization

## Test Scenarios

1. **Cold Start Testing** - First request timing
2. **Batch Prediction Testing** - Various complexity levels
3. **Concurrent Load Testing** - 2, 5, 10 simultaneous users
4. **Stress Testing** - 1, 2, 5 requests/second sustained load

## Dependencies

- aiohttp
- matplotlib
- seaborn
- pandas
- numpy
- psutil