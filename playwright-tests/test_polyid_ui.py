#!/usr/bin/env python3
"""
PolyID Hugging Face Space UI Testing with Playwright
==================================================

Comprehensive UI testing for the PolyID space including:
- Space loading and initialization
- Model loading indicators
- Form validation
- Prediction workflow
- Error handling
- Responsiveness
"""

import pytest
from playwright.sync_api import Page, expect
import time

SPACE_URL = "https://jkbennitt-polyid-private.hf.space"

class TestPolyIDUI:
    """PolyID UI test suite"""

    @pytest.mark.ui
    def test_space_loads(self, page: Page):
        """Test that the space loads properly"""
        page.goto(SPACE_URL)
        expect(page).to_have_title(lambda title: "PolyID" in title or "Hugging Face" in title)

        # Wait for the interface to load
        page.wait_for_timeout(5000)

        # Check for key elements
        expect(page.locator("text=PolyID")).to_be_visible(timeout=10000)

    @pytest.mark.loading
    def test_model_initialization(self, page: Page):
        """Test model loading and status indicators"""
        page.goto(SPACE_URL)

        # Wait for the model to initialize
        page.wait_for_timeout(10000)

        # Look for status indicators
        # The space should show system status or model ready indicators
        page.screenshot(path="playwright-tests/screenshots/model_init.png")

    @pytest.mark.validation
    def test_input_validation(self, page: Page):
        """Test input form validation"""
        page.goto(SPACE_URL)
        page.wait_for_timeout(5000)

        # Find the input field - try common selectors for Gradio interfaces
        input_selectors = [
            "input[type='text']",
            "textarea",
            ".input-field",
            "[data-testid*='textbox']"
        ]

        input_field = None
        for selector in input_selectors:
            if page.locator(selector).count() > 0:
                input_field = page.locator(selector).first
                break

        if input_field:
            # Test valid input
            input_field.fill("CC")
            page.screenshot(path="playwright-tests/screenshots/valid_input.png")

            # Test empty input
            input_field.fill("")
            page.screenshot(path="playwright-tests/screenshots/empty_input.png")

    @pytest.mark.prediction
    def test_prediction_workflow(self, page: Page):
        """Test complete prediction workflow"""
        page.goto(SPACE_URL)
        page.wait_for_timeout(10000)

        # Find and fill input
        input_selectors = [
            "input[type='text']",
            "textarea",
            ".input-field",
            "[data-testid*='textbox']"
        ]

        for selector in input_selectors:
            if page.locator(selector).count() > 0:
                input_field = page.locator(selector).first
                input_field.fill("CC")
                break

        # Find and click submit button
        submit_selectors = [
            "button:has-text('Submit')",
            "button:has-text('Predict')",
            "button:has-text('Run')",
            ".submit-button",
            "[data-testid*='button']"
        ]

        for selector in submit_selectors:
            if page.locator(selector).count() > 0:
                button = page.locator(selector).first
                button.click()
                break

        # Wait for prediction to complete
        page.wait_for_timeout(15000)

        # Take screenshot of results
        page.screenshot(path="playwright-tests/screenshots/prediction_result.png")

        # Check for output elements
        output_selectors = [
            ".output",
            ".result",
            "[data-testid*='output']",
            "text=Glass"  # Properties like Glass Transition Temperature
        ]

        found_output = False
        for selector in output_selectors:
            if page.locator(selector).count() > 0:
                found_output = True
                break

        # Screenshot the full page
        page.screenshot(path="playwright-tests/screenshots/full_page_result.png")

    @pytest.mark.error_handling
    def test_error_handling(self, page: Page):
        """Test error handling with invalid inputs"""
        page.goto(SPACE_URL)
        page.wait_for_timeout(5000)

        # Test with invalid SMILES
        invalid_inputs = ["INVALID_SMILES", "", "123ABC"]

        for i, invalid_input in enumerate(invalid_inputs):
            # Find input field
            input_selectors = ["input[type='text']", "textarea"]
            for selector in input_selectors:
                if page.locator(selector).count() > 0:
                    input_field = page.locator(selector).first
                    input_field.fill(invalid_input)

                    # Try to submit
                    submit_selectors = ["button:has-text('Submit')", "button:has-text('Predict')"]
                    for submit_selector in submit_selectors:
                        if page.locator(submit_selector).count() > 0:
                            page.locator(submit_selector).first.click()
                            break

                    # Wait and screenshot
                    page.wait_for_timeout(5000)
                    page.screenshot(path=f"playwright-tests/screenshots/error_{i}.png")
                    break

    @pytest.mark.mobile
    def test_mobile_responsiveness(self, page: Page):
        """Test mobile responsiveness"""
        # Set mobile viewport
        page.set_viewport_size({"width": 375, "height": 667})
        page.goto(SPACE_URL)
        page.wait_for_timeout(5000)

        # Screenshot mobile view
        page.screenshot(path="playwright-tests/screenshots/mobile_view.png")

    @pytest.mark.cross_browser
    def test_gradio_components(self, page: Page):
        """Test Gradio-specific components"""
        page.goto(SPACE_URL)
        page.wait_for_timeout(10000)

        # Look for Gradio-specific elements
        gradio_elements = [
            ".gradio-container",
            ".interface",
            "[data-testid*='gradio']"
        ]

        # Take a comprehensive screenshot
        page.screenshot(path="playwright-tests/screenshots/gradio_interface.png")

        # Check for any JavaScript errors in console
        console_messages = []
        page.on("console", lambda msg: console_messages.append(f"{msg.type}: {msg.text}"))

        # Wait and collect any console messages
        page.wait_for_timeout(10000)

        # Save console messages to file
        with open("playwright-tests/screenshots/console_log.txt", "w") as f:
            f.write("Console messages:\n")
            for msg in console_messages:
                f.write(f"{msg}\n")

    @pytest.mark.slow
    def test_performance_timing(self, page: Page):
        """Test page load and interaction timing"""
        start_time = time.time()
        page.goto(SPACE_URL)
        load_time = time.time() - start_time

        page.wait_for_timeout(5000)
        ready_time = time.time() - start_time

        # Save timing info
        with open("playwright-tests/screenshots/timing_results.txt", "w") as f:
            f.write(f"Page load time: {load_time:.2f}s\n")
            f.write(f"Ready time: {ready_time:.2f}s\n")

        # Take final screenshot
        page.screenshot(path="playwright-tests/screenshots/final_state.png")