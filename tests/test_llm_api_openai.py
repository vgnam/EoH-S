from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from llm4ad.tools.llm.llm_api_openai import OpenAIAPI  # noqa: E402


class OpenAIAPITests(unittest.TestCase):
    def test_draw_sample_does_not_send_reasoning_parameters(self):
        create_calls = []

        def create(**kwargs):
            create_calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="result"))],
                usage=None,
            )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with patch(
            "llm4ad.tools.llm.llm_api_openai.openai.OpenAI",
            return_value=client,
        ):
            api = OpenAIAPI(
                base_url="https://example.test/v1",
                api_key="test-key",
                model="test-model",
            )

        result = api.draw_sample(
            " prompt ",
            temperature=0.5,
            reasoning={"effort": "high"},
            reasoning_effort="high",
            extra_body={
                "reasoning": {"enabled": True},
                "reasoning_effort": "high",
                "provider_option": True,
            },
        )

        self.assertEqual(result, "result")
        self.assertEqual(len(create_calls), 1)
        request = create_calls[0]
        self.assertNotIn("reasoning", request)
        self.assertNotIn("reasoning_effort", request)
        self.assertEqual(request["extra_body"], {"provider_option": True})
        self.assertEqual(request["temperature"], 0.5)
        self.assertEqual(request["messages"][0]["content"], "prompt")

    def test_deepseek_v4_flash_explicitly_disables_thinking(self):
        create_calls = []

        def create(**kwargs):
            create_calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="result"))],
                usage=None,
            )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with patch(
            "llm4ad.tools.llm.llm_api_openai.openai.OpenAI",
            return_value=client,
        ):
            api = OpenAIAPI(
                base_url="https://api.deepseek.com",
                api_key="test-key",
                model="deepseek-v4-flash",
            )

        api.draw_sample(
            "prompt",
            reasoning_effort="max",
            extra_body={
                "thinking": {"type": "enabled"},
                "provider_option": True,
            },
        )

        request = create_calls[0]
        self.assertNotIn("reasoning_effort", request)
        self.assertEqual(
            request["extra_body"],
            {
                "thinking": {"type": "disabled"},
                "provider_option": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
