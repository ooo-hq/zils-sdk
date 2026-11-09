"""Coding agents get the whole path over MCP: find decisions, measure, prepare training data from logs, train."""

import json
import tempfile
import unittest
from pathlib import Path

import test_mcp
from sdk_helpers import api
from test_fit import APP
from test_prepare_log import QUESTIONS, game_logs


class McpFitTests(unittest.TestCase):
    env = test_mcp.McpTests.env  # the stdio session helpers, without re-running McpTests' own tests
    run_session = test_mcp.McpTests.run_session

    def test_agents_are_told_about_the_fit_tools_and_the_training_path(self):
        with tempfile.TemporaryDirectory() as directory:

            async def check(session):
                names = {t.name for t in (await session.list_tools()).tools}
                self.assertLessEqual({"fit_scan", "fit_run", "train_prepare", "train_guide"}, names)
                guide = (await session.call_tool("train_guide", {})).structuredContent["guide"]
                # The guide covers decisions made inside the user's program, end to end.
                for step in ("record", "fit_scan", "fit_run", "train_prepare", "episode"):
                    self.assertIn(step, guide)

            self.run_session(self.env(directory), check)

    def test_fit_scan_finds_the_decisions_and_drafts_their_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory) / "app"
            app.mkdir()
            (app / "triage.py").write_text(APP["app/triage.py"])

            async def check(session):
                result = await session.call_tool("fit_scan", {"path": str(app)})
                self.assertFalse(result.isError, result.content)
                found = result.structuredContent
                self.assertEqual(
                    [f["name"] for f in found["findings"] if f["kind"] == "decision"], ["is_urgent"]
                )
                self.assertEqual(found["questions"]["is_urgent"]["type"], "noul")

            self.run_session(self.env(directory), check)

    def test_train_prepare_takes_a_folder_of_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = game_logs(root / "logs")

            async def check(session):
                result = await session.call_tool(
                    "train_prepare",
                    {
                        "input": str(logs),
                        "questions": QUESTIONS,
                        "decide": "next_action",
                        "out": str(root / "out"),
                    },
                )
                self.assertFalse(result.isError, result.content)
                self.assertEqual(
                    (
                        result.structuredContent["episodes"],
                        result.structuredContent["repeats_left_out"],
                    ),
                    (12, 1),
                )
                table_options = await session.call_tool(
                    "train_prepare",
                    {"input": str(logs), "read": ["health"], "out": str(root / "x")},
                )
                self.assertTrue(table_options.isError)
                self.assertIn("--read and --group are for tables", table_options.content[0].text)

            self.run_session(self.env(directory), check)

    def test_fit_run_measures_a_model_on_a_log_within_the_cap(self):
        with tempfile.TemporaryDirectory() as directory, api() as service:
            root = Path(directory)
            log = root / "fit-log.jsonl"
            log.write_text(
                "".join(
                    json.dumps(
                        {"question": "is_urgent", "state": f"ticket {i}", "answer": i % 2 == 0}
                    )
                    + "\n"
                    for i in range(6)
                )
            )
            questions = {"is_urgent": {"type": "noul", "instructions": "Is this ticket urgent?"}}
            env = self.env(directory)
            env.update(ZILS_API_KEY="valid-key", ZILS_BASE_URL=service.url)

            async def check(session):
                capped = await session.call_tool(
                    "fit_run", {"questions": questions, "log": str(log), "max_rows": 3}
                )
                self.assertTrue(capped.isError)
                self.assertIn("exceed", capped.content[0].text)
                self.assertEqual(len(service.calls), 0)  # nothing was sent
                result = await session.call_tool(
                    "fit_run",
                    {
                        "questions": questions,
                        "log": str(log),
                        "model": "zils-adapter-trained",
                        "max_rows": 10,
                    },
                )
                self.assertFalse(result.isError, result.content)
                report = result.structuredContent
                self.assertEqual((report["rows"], report["model"]), (6, "zils-adapter-trained"))
                self.assertIn("is_urgent", report["questions"])
                asked = [body for method, _, _, body in service.calls if method == "POST"]
                self.assertEqual(len(asked), 6)
                self.assertEqual({body["model"] for body in asked}, {"zils-adapter-trained"})

            self.run_session(env, check)


class SkillTests(unittest.TestCase):
    def test_the_fit_skill_goes_on_to_training(self):
        from importlib.resources import files

        skill = files("zils_sdk").joinpath("skills/zils-fit/SKILL.md").read_text()
        self.assertIn("zils fit train", skill)
        self.assertIn("zils fit run --model", skill.replace("\n", " "))


if __name__ == "__main__":
    unittest.main()
