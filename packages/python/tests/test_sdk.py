import asyncio
import importlib
import importlib.util
import os
import unittest
from unittest.mock import patch

from sdk_helpers import QUESTIONS, api


class ClientTests(unittest.TestCase):
    def sdk(self):
        self.assertIsNotNone(importlib.util.find_spec("zils_sdk"), "Zils SDK is missing")
        return importlib.import_module("zils_sdk")

    def test_sync_client_preserves_answers_alias_and_billable_usage(self):
        sdk = self.sdk()
        with api() as service, sdk.ZilsClient(api_key="valid-key", base_url=service.url) as client:
            self.assertEqual(
                {m.name for m in client.models.list().models}, {"zils-test-r1", "zils-shared"}
            )
            result = client.system_one(state={"ticket": "Refund please"}, questions=QUESTIONS)
            self.assertEqual(result.model, "zils-test-r1")
            self.assertEqual(result.nouls["eligible"].noul, 0.5)
            self.assertEqual(result.scores["urgency"].legend, {0: {"level": "low"}, 1: ["high"]})
            self.assertEqual(result.usage.billable_input_tokens, 73)
            self.assertEqual(result.usage.input_tokens, 30)

    def test_async_client_preserves_usage_and_shares_configuration(self):
        sdk = self.sdk()

        async def run(url):
            async with sdk.AsyncZilsClient(api_key="valid-key", base_url=url) as client:
                return await client.system_one("Refund", QUESTIONS)

        with api() as service:
            result = asyncio.run(run(service.url))
            self.assertEqual(result.usage.billable_input_tokens, 73)
            self.assertEqual(set(result.answers), set(QUESTIONS))

    def test_zils_environment_overrides_upstream_defaults(self):
        sdk = self.sdk()
        with (
            api() as service,
            patch.dict(
                os.environ,
                {
                    "ZILS_API_KEY": "valid-key",
                    "ZILS_BASE_URL": service.url,
                    "ZILS_DEFAULT_MODEL": "zils-test-r1",
                    "TYPESAFE_API_KEY": "wrong",
                    "TYPESAFE_BASE_URL": "https://invalid.example",
                    "TYPESAFE_DEFAULT_MODEL": "wrong",
                },
            ),
        ):
            with sdk.ZilsClient() as client:
                client.system_one("Refund", QUESTIONS)
            self.assertEqual(service.calls[-1][3]["model"], "zils-test-r1")

    def test_missing_zils_key_does_not_use_typesafe_key(self):
        sdk = self.sdk()
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "private-upstream-key"}, clear=True):
            with self.assertRaisesRegex(ValueError, "ZILS_API_KEY"):
                sdk.ZilsClient()

    def test_explicit_invalid_configuration_is_not_silently_replaced(self):
        sdk = self.sdk()
        with patch.dict(os.environ, {"ZILS_API_KEY": "valid-key"}):
            for options in (
                {"api_key": ""},
                {"api_key": "secret\nheader"},
                {"model": ""},
                {"base_url": "http://remote.example"},
                {"base_url": "https://user:secret@host.example"},
                {"base_url": "https://host.example?key=secret"},
                {"base_url": "https://host.example/#fragment"},
            ):
                with self.subTest(options=options), self.assertRaises(ValueError) as error:
                    sdk.ZilsClient(**options)
                self.assertNotIn("secret", str(error.exception))

    def test_rejected_decision_is_not_automatically_retried(self):
        sdk = self.sdk()
        with api() as service, sdk.ZilsClient(api_key="valid-key", base_url=service.url) as client:
            service.store.limit = True
            with self.assertRaises(sdk.APIError) as caught:
                client.system_one("Refund", QUESTIONS)
            self.assertEqual(caught.exception.status, 429)
            self.assertEqual(len(service.calls), 1)


if __name__ == "__main__":
    unittest.main()
