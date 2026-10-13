import asyncio
import math
import unittest
from unittest.mock import patch

from ranking_fixture import CASES, GROUPED_CASES, ranking_api
from sdk_helpers import api
from zils_sdk import APIError, AsyncZilsClient, ZilsClient, ZilsError


class RankingTests(unittest.TestCase):
    def client(self, url, kind=ZilsClient):
        client = kind(api_key="test-key", base_url=url)
        self.assertTrue(callable(getattr(client, "rank", None)), "SDK rank method is missing")
        return client

    def assert_ranking(self, result, expected):
        self.assertEqual([item.index for item in result.ranking], [row[0] for row in expected])
        for item, (_, score) in zip(result.ranking, expected, strict=True):
            self.assertAlmostEqual(item.score, score, places=6)

    def test_recovers_known_strengths_ties_cycles_and_order_bias(self):
        for case in CASES:
            with self.subTest(case=case["name"]), ranking_api(case) as service:
                with self.client(service.url) as client:
                    result = client.rank(case["items"], "Most useful first", model="private-model")
                self.assert_ranking(result, case["ranking"])
                self.assertEqual(result.model, "zils-test-r1")
                self.assertEqual(result.comparisons, 6)
                self.assertEqual(result.usage.input_tokens, 1234)
                self.assertEqual(result.usage.billable_input_tokens, 456)
                self.assertEqual(len(service.calls), 1)
                path, auth, body = service.calls[0]
                self.assertEqual((path, auth), ("/v1/systemone", "Bearer test-key"))
                self.assertEqual(body["state"], {"criterion": "Most useful first"})
                self.assertEqual(body["model"], "private-model")
                pairs = [
                    (q["criteria"]["a"]["item"], q["criteria"]["b"]["item"])
                    for q in body["questions"].values()
                ]
                self.assertEqual(len(set(pairs)), 6)
                self.assertTrue(all(a != b and (b, a) in pairs for a, b in pairs))

    def test_reordering_inputs_changes_indices_but_not_the_ranking(self):
        with ranking_api(CASES[0]) as service, self.client(service.url) as client:
            result = client.rank(["high", "low", "medium"], "Most useful first")
        self.assert_ranking(result, [[0, 1], [2, 0.5], [1, 0.25]])

    def test_async_client_returns_the_same_result(self):
        async def run(url):
            async with self.client(url, AsyncZilsClient) as client:
                return await client.rank(CASES[0]["items"], "Most useful first")

        with ranking_api(CASES[0]) as service:
            result = asyncio.run(run(service.url))
        self.assert_ranking(result, CASES[0]["ranking"])
        self.assertEqual(result.usage.billable_input_tokens, 456)

    def test_grouped_ranking_recovers_strengths_and_balances_positions(self):
        for case in GROUPED_CASES:
            with self.subTest(case=case["name"]), ranking_api(case) as service:
                with self.client(service.url) as client:
                    result = client.rank(
                        case["items"],
                        "Most useful first",
                        group_size=case["group_size"],
                        model="private-model",
                    )
                self.assert_ranking(result, case["ranking"])
                self.assertEqual(result.comparisons, len(case["items"]))
                self.assertEqual(result.usage.billable_input_tokens, 456)
                self.assertEqual(result.usage.input_tokens, 1234)
                self.assertEqual(len(service.calls), 1)
                body = service.calls[0][2]
                self.assertEqual(body["model"], "private-model")
                self.assertEqual(body["state"], {"criterion": "Most useful first"})
                groups = [list(q["criteria"].values()) for q in body["questions"].values()]
                size = min(case["group_size"], len(case["items"]))
                self.assertTrue(all(len(group) == size for group in groups))
                for position in range(size):
                    self.assertCountEqual(
                        [group[position]["item"] for group in groups], case["items"]
                    )

    def test_grouped_async_ranking_snapshots_input_and_supports_reordering(self):
        case = GROUPED_CASES[2]
        items = ["high", "low", "medium"]

        async def run(url):
            async with self.client(url, AsyncZilsClient) as client:
                return await client.rank(items, "Most useful first", group_size=4)

        with ranking_api(case, lambda value: items.clear()) as service:
            result = asyncio.run(run(service.url))
        self.assert_ranking(result, [[0, 1], [2, 0.5], [1, 0.25]])
        self.assertEqual(result.comparisons, 3)

    def test_invalid_group_sizes_never_send_a_paid_request(self):
        with ranking_api(GROUPED_CASES[0]) as service, self.client(service.url) as client:
            for size in (0, 1, 11, 2.5, True, "4"):
                with self.subTest(size=size), self.assertRaises(ValueError):
                    client.rank(["a", "b", "c"], "Rank", group_size=size)
            self.assertEqual(service.calls, [])

    def test_grouped_incomplete_or_invalid_answers_fail_without_retry(self):
        def wrong_labels(value):
            answer = next(iter(value["answers"].values()))
            key = next(iter(answer["probabilities"]))
            answer["probabilities"]["unexpected"] = answer["probabilities"].pop(key)

        for transform in (
            lambda v: v["answers"].pop(next(iter(v["answers"]))),
            lambda v: v["answers"].update(extra=next(iter(v["answers"].values()))),
            wrong_labels,
            lambda v: next(iter(v["answers"].values())).update(choice="unexpected"),
            lambda v: next(iter(v["answers"].values())).update(confidence=-1),
            lambda v: next(iter(v["answers"].values())).update(type="noul"),
            lambda v: next(iter(v["answers"].values()))["probabilities"].update(a=2),
        ):
            with (
                self.subTest(transform=transform),
                ranking_api(GROUPED_CASES[0], transform) as service,
            ):
                with self.client(service.url) as client, self.assertRaises((ValueError, ZilsError)):
                    client.rank(GROUPED_CASES[0]["items"], "Rank", group_size=4)
                self.assertEqual(len(service.calls), 1)

    def test_grouped_zero_one_probabilities_keep_finite_strengths(self):
        def decisive(value):
            for answer in value["answers"].values():
                answer["probabilities"] = {
                    key: float(key == answer["choice"]) for key in answer["probabilities"]
                }

        with ranking_api(GROUPED_CASES[0], decisive) as service, self.client(service.url) as client:
            result = client.rank(GROUPED_CASES[0]["items"], "Rank", group_size=4)
        self.assertEqual(result.ranking[0].index, 9)
        self.assertTrue(all(math.isfinite(r.score) and 0 < r.score <= 1 for r in result.ranking))

    def test_grouped_ranking_uses_the_existing_gateway_contract(self):
        with api() as service, ZilsClient(api_key="valid-key", base_url=service.url) as client:
            result = client.rank(["a", "b", "c", "d", "e"], "Rank", group_size=4)
        self.assert_ranking(result, [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]])
        self.assertEqual(result.comparisons, 5)

    def test_rejects_invalid_input_before_any_paid_request(self):
        circular = []
        circular.append(circular)
        with ranking_api(CASES[0]) as service, self.client(service.url) as client:
            for items, instructions in (
                ([], "Rank"),
                (["a"], "Rank"),
                (["a"] * 11, "Rank"),
                ("ab", "Rank"),
                (["a", "b"], " "),
                (["a", "b"], None),
                (["a", float("nan")], "Rank"),
                (["a", object()], "Rank"),
                (["a", circular], "Rank"),
            ):
                with self.subTest(items=type(items), instructions=instructions):
                    with self.assertRaises(ValueError):
                        client.rank(items, instructions)
            self.assertEqual(service.calls, [])

    def test_non_json_object_keys_are_rejected_without_lossy_conversion(self):
        with ZilsClient(api_key="test-key") as client:
            with patch.object(
                client, "system_one", side_effect=AssertionError("Input reached API")
            ):
                with self.assertRaises(ValueError):
                    client.rank([{1: "would be lost", "1": "kept"}, "b"], "Most useful first")

    def test_all_ten_items_are_compared_with_a_bounded_request(self):
        case = {"items": list(range(10)), "probabilities": [[0.5] * 10 for _ in range(10)]}
        with ranking_api(case) as service, self.client(service.url) as client:
            result = client.rank(case["items"], "Most useful first")
        self.assertEqual(result.comparisons, 90)
        self.assertEqual(len(service.calls[0][2]["questions"]), 90)
        self.assert_ranking(result, [[i, 1] for i in range(10)])

    def test_zero_one_probabilities_still_produce_finite_scores(self):
        case = {"items": ["bad", "good"], "probabilities": [[None, 0], [1, None]]}
        with ranking_api(case) as service, self.client(service.url) as client:
            result = client.rank(case["items"], "Most useful first")
        self.assertEqual([item.index for item in result.ranking], [1, 0])
        self.assertEqual(result.ranking[0].score, 1)
        self.assertTrue(
            all(math.isfinite(item.score) and 0 < item.score <= 1 for item in result.ranking)
        )

    def test_ten_decisive_items_converge_without_collapsing_small_scores(self):
        case = {
            "items": list(range(10)),
            "probabilities": [[float(i > j) for j in range(10)] for i in range(10)],
        }
        with ranking_api(case) as service, self.client(service.url) as client:
            result = client.rank(case["items"], "Most useful first")
        self.assertEqual([item.index for item in result.ranking], list(reversed(range(10))))
        self.assertTrue(
            all(math.isfinite(item.score) and 0 < item.score <= 1 for item in result.ranking)
        )

    def test_async_input_list_is_snapshotted_before_the_request(self):
        items = list(CASES[0]["items"])

        async def run(url):
            async with self.client(url, AsyncZilsClient) as client:
                return await client.rank(items, "Most useful first")

        with ranking_api(CASES[0], lambda value: items.pop()) as service:
            result = asyncio.run(run(service.url))
        self.assert_ranking(result, CASES[0]["ranking"])

    def test_structured_and_duplicate_items_keep_their_original_indices(self):
        items = [{"text": "返金", "tags": [True, None]}, {"text": "返金", "tags": [True, None]}]
        case = {"items": items, "probabilities": [[0.5, 0.5], [0.5, 0.5]]}
        with ranking_api(case) as service, self.client(service.url) as client:
            result = client.rank(items, "Most useful first")
        self.assert_ranking(result, [[0, 1], [1, 1]])
        self.assertEqual(
            service.calls[0][2]["questions"]["pair_0_1"]["criteria"]["a"]["item"], items[0]
        )

    def test_rank_uses_the_existing_gateway_contract(self):
        with api() as service, ZilsClient(api_key="valid-key", base_url=service.url) as client:
            result = client.rank(["a", "b", "c"], "Most useful first")
        self.assert_ranking(result, [[0, 1], [1, 1], [2, 1]])
        self.assertEqual(result.usage.billable_input_tokens, 73)

    def test_malformed_or_incomplete_comparisons_never_return_a_ranking(self):
        def remove_answer(value):
            value["answers"].pop(next(iter(value["answers"])))

        def change_probabilities(probabilities):
            def change(value):
                next(iter(value["answers"].values()))["probabilities"] = probabilities

            return change

        for transform in (
            remove_answer,
            lambda v: next(iter(v["answers"].values())).pop("confidence"),
            lambda v: next(iter(v["answers"].values())).update(confidence=-1),
            change_probabilities({"a": 0.9}),
            change_probabilities({"a": -0.1, "b": 1.1}),
            change_probabilities({"a": 0.8, "b": 0.8}),
        ):
            with self.subTest(transform=transform), ranking_api(CASES[0], transform) as service:
                with self.client(service.url) as client:
                    with self.assertRaises((ValueError, ZilsError)):
                        client.rank(CASES[0]["items"], "Most useful first")
                self.assertEqual(len(service.calls), 1)

    def test_absent_billable_usage_is_not_invented(self):
        def remove_billing(value):
            value["usage"].pop("billable_input_tokens")

        with ranking_api(CASES[0], remove_billing) as service, self.client(service.url) as client:
            result = client.rank(CASES[0]["items"], "Most useful first")
        self.assertIsNone(result.usage.billable_input_tokens)
        self.assertEqual(result.usage.input_tokens, 1234)

    def test_rate_limit_propagates_without_a_second_paid_request(self):
        with ranking_api(CASES[0], status=429) as service, self.client(service.url) as client:
            with self.assertRaises(APIError) as caught:
                client.rank(CASES[0]["items"], "Most useful first")
            self.assertEqual(caught.exception.status, 429)
            self.assertEqual(len(service.calls), 1)

    def test_missing_or_invalid_model_and_usage_metadata_never_return_a_ranking(self):
        for transform in (
            lambda v: v.pop("model"),
            lambda v: v.update(model=""),
            lambda v: v.pop("usage"),
            lambda v: v["usage"].pop("input_tokens"),
            lambda v: v["usage"].update(input_tokens=-1),
            lambda v: v["usage"].update(output_tokens=0.5),
            lambda v: v["usage"].update(billable_input_tokens="bad"),
        ):
            with self.subTest(transform=transform), ranking_api(CASES[0], transform) as service:
                with self.client(service.url) as client:
                    with self.assertRaises((ValueError, ZilsError)):
                        client.rank(CASES[0]["items"], "Most useful first")
                self.assertEqual(len(service.calls), 1)
