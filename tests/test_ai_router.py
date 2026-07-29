"""Tests for Hugging Face image endpoint routing."""

import os
import unittest
from unittest.mock import MagicMock, patch

from ai_fallback import IRAProviderError
from ai_router import generate_image_router


class GenerateImageRouterTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"HF_TOKEN": "test-token"}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    @patch("ai_router.requests.post")
    def test_uses_flux_dev_endpoint_first(self, post):
        post.return_value = MagicMock(status_code=200, content=b"image")

        self.assertEqual(generate_image_router("a mountain"), b"image")

        post.assert_called_once_with(
            "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev",
            headers={"Authorization": "Bearer test-token"},
            json={"inputs": "a mountain"},
            timeout=60,
        )

    @patch("ai_router.requests.post")
    def test_falls_back_to_stable_diffusion_when_flux_fails(self, post):
        post.side_effect = [
            MagicMock(status_code=410, text="deprecated"),
            MagicMock(status_code=200, content=b"fallback-image"),
        ]

        self.assertEqual(generate_image_router("a mountain"), b"fallback-image")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(
            post.call_args_list[1].args[0],
            "https://api-inference.huggingface.co/models/"
            "stabilityai/stable-diffusion-3.5-large",
        )

    @patch("ai_router.requests.post")
    def test_raises_provider_error_after_both_endpoints_fail(self, post):
        post.side_effect = [
            MagicMock(status_code=410, text="deprecated"),
            MagicMock(status_code=503, text="busy"),
        ]

        with self.assertRaisesRegex(IRAProviderError, "503.*busy"):
            generate_image_router("a mountain")


if __name__ == "__main__":
    unittest.main()
