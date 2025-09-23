# Testing Suite

This directory contains various testing scripts for the PolyID Hugging Face Space.

## API Testing Scripts

- `api_discovery.py` - Discovers available Gradio API endpoints
- `simple_api_test.py` - Basic connectivity and API testing
- `correct_api_test.py` - Advanced API testing with proper Gradio format
- `live_space_test.py` - Comprehensive live space functionality testing

## Usage

Run individual scripts:
```bash
python testing/simple_api_test.py
python testing/api_discovery.py
```

These scripts test the live deployment at:
https://jkbennitt-polyid-private.hf.space