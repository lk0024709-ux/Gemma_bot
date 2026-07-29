"""Tests for Pollinations image endpoint routing."""

import unittest
from unittest.mock import MagicMock, patch

import requests

from ai_fallback import IRAProviderError
from ai_router import generate_image_router


class GenerateImageRouterTests(unittest.TestCase):
    @patch("ai_router.requests.get")
    def test_calls_pollinations_with_encoded_prompt(self, get):
        get.return_value = MagicMock(status_code=200, content=b"image")

        self.assertEqual(
            generate_image_router("a red fox & a blue moon"), b"image"
        )

        get.assert_called_once()
        called_url = get.call_args.args[0]
        self.assertTrue(
            called_url.startswith("https://image.pollinations.ai/prompt/")
        )
        # The prompt segment must be fully URL-encoded: no raw spaces or '&'.
        prompt_segment = called_url[len("https://image.pollinations.ai/prompt/"):]
        self.assertNotIn(" ", prompt_segment)
        self.assertNotIn("&", prompt_segment)
        self.assertIn("%20", prompt_segment)  # encoded spaces
        self.assertIn("%26", prompt_segment)  # encoded '&'
        # A timeout must be supplied so the call cannot hang forever.
        self.assertEqual(get.call_args.kwargs.get("timeout"), 60)

    @patch("ai_router.requests.get")
    def test_returns_image_bytes_on_success(self, get):
        get.return_value = MagicMock(status_code=200, content=b"jpeg-bytes")

        self.assertEqual(generate_image_router("a mountain"), b"jpeg-bytes")

    @patch("ai_router.requests.get")
    def test_raises_provider_error_on_http_error(self, get):
        get.return_value = MagicMock(status_code=502, text="bad gateway")

        with self.assertRaisesRegex(IRAProviderError, "502"):
            generate_image_router("a mountain")

    @patch("ai_router.requests.get")
    def test_raises_provider_error_on_network_error(self, get):
        get.side_effect = requests.ConnectionError("boom")

        with self.assertRaisesRegex(IRAProviderError, "network error"):
            generate_image_router("a mountain")


if __name__ == "__main__":
    unittest.main()
