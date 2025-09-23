# Playwright UI Testing Suite

Comprehensive UI testing for the PolyID Hugging Face Space using Playwright.

## Setup

```bash
pip install playwright
playwright install
```

## Running Tests

```bash
# Run all UI tests
cd playwright-tests
pytest test_polyid_ui.py -v

# Run specific test categories
pytest -m ui          # UI interaction tests
pytest -m prediction  # Prediction workflow tests
pytest -m mobile      # Mobile responsiveness tests
```

## Test Coverage

- Space loading and initialization
- Model loading indicators
- Input validation and form handling
- Prediction workflow end-to-end
- Error handling with invalid inputs
- Mobile responsiveness
- Cross-browser compatibility (Chromium, Firefox, WebKit)

## Output

- Screenshots saved to `screenshots/`
- Videos saved to `videos/` (on failure)
- Test reports in `test-results/`

## Configuration

See `pytest.ini` for test configuration and markers.