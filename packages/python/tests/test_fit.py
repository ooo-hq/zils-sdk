import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sdk_helpers import api
from zils_sdk import record
from zils_sdk.fit import draft_questions, measure, read_log, scan

APP = {
    "app/triage.py": """from openai import OpenAI
client = OpenAI()

def is_urgent(ticket: str) -> bool:
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": "Does this ticket need a response today?"}],
    )
    return resp.choices[0].message.content.strip().lower().startswith("yes")
""",
    "app/mood.py": """import anthropic
claude = anthropic.Anthropic()

def frustration(ticket: str) -> int:
    msg = claude.messages.create(
        model="claude-haiku-4-5",
        system="How frustrated is the customer? Reply 0 calm, 1 impatient, 2 angry.",
        messages=[{"role": "user", "content": ticket}],
    )
    return int(msg.content[0].text.strip())
""",
    "app/reply.py": """def draft_reply(ticket: str) -> str:
    msg = claude.messages.create(model="x", system="Write a kind reply.", messages=[])
    return msg.content[0].text
""",
    "web/route.ts": """export async function routeTicket(ticket: string) {
  const { object } = await generateObject({
    schema: z.object({ department: z.enum(["billing", "technical", "sales"]) }),
    prompt: `Which team should handle this ticket?\\n\\n${ticket}`,
  });
  return object.department;
}
""",
    "app/reader.py": """async def decide(self, model, state, questions):
    return await self._client.post(
        "/inference/alpha/decisions",
        json_data={"model": model, "state": state, "questions": questions},
    )
""",
    # Inbox Zero patterns: arrow functions, schemas above the call, string label fields,
    # providers named in comments, several provider cases in one file, agent .decide().
    "web/status.ts": """export const threadStatus = async ({ thread }: Args) => {
  const statuses = definitions.map((d) => d.systemType);
  // Model: typesafe/jev-1.13 is mentioned here only in a comment.
  const schema = z.object({
    status: z.enum(statuses),
    rationale: z.string(),
  });
  const aiResponse = await generateObject({ ...modelOptions, prompt, schema });
  return aiResponse.object;
};
""",
    "web/category.ts": """export async function categorizeSenderWithLlm<T>({ sender }: Args) {
  const aiResponse = await generateObject({
    prompt,
    schema: z.object({ rationale: z.string(), category: z.string() }),
  });
  return aiResponse.object;
}
""",
    "web/providers.ts": """export function createConfig(provider: string) {
  switch (provider) {
    case "typesafe":
      return systemOne({ url: "https://api.typesafe.ai/v1/systemone", questions });
    case "openrouter":
      return systemOne({ url: "https://openrouter.ai/api/alpha/decisions", questions });
    case "other":
      return systemOne({ url: "https://example.com/v1/systemone", questions });
  }
}
""",
    "worker/agent.ts": """async step(observation) {
  const decision = await this.options.decide(observation, history, signal);
  return decision;
}
""",
    "tests/test_triage.py": """def test_mock():
    client.chat.completions.create(model="x", messages=[])
""",
}


class RecordTests(unittest.TestCase):
    def test_logs_sync_and_async_calls_privately_and_merges_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "fit-log.jsonl"
            os.environ["ZILS_FIT_LOG"] = str(log)
            self.addCleanup(os.environ.pop, "ZILS_FIT_LOG", None)

            @record("urgent")
            def urgent(ticket):
                return "outage" in ticket

            @record("team")
            async def team(ticket):
                return "technical"

            @record("broken")
            def broken(ticket):
                raise RuntimeError("upstream down")

            self.assertIs(urgent("outage now"), True)
            self.assertEqual(asyncio.run(team("outage now")), "technical")
            urgent("can wait")
            with self.assertRaises(RuntimeError):
                broken("x")
            self.assertEqual(log.stat().st_mode & 0o777, 0o600)
            rows = {r["state"]: r["labels"] for r in read_log(log)}
            self.assertEqual(
                rows,
                {
                    "outage now": {"urgent": True, "team": "technical"},
                    "can wait": {"urgent": False},
                },
            )

    def test_unloggable_input_never_breaks_the_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["ZILS_FIT_LOG"] = str(Path(tmp) / "fit-log.jsonl")
            self.addCleanup(os.environ.pop, "ZILS_FIT_LOG", None)
            self.assertEqual(record("q")(lambda value: 7)(object()), 7)


class ScanTests(unittest.TestCase):
    def test_finds_decisions_drop_ins_and_skips_text_and_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in APP.items():
                (Path(tmp) / name).parent.mkdir(parents=True, exist_ok=True)
                (Path(tmp) / name).write_text(text)
            found = {f["name"]: f for f in scan(tmp)}
            self.assertEqual(
                {n: f["kind"] for n, f in found.items()},
                {
                    "frustration": "decision",
                    "draft_reply": "text",
                    "decide": "drop-in",
                    "is_urgent": "decision",
                    "department": "decision",
                    "status": "decision",
                    "category": "decision",
                    "createConfig": "drop-in",
                },
            )
            by_name = {f["name"]: f for f in scan(tmp)}
            self.assertEqual(by_name["status"]["function"], "threadStatus")
            self.assertEqual(by_name["category"]["function"], "categorizeSenderWithLlm")
            self.assertNotIn("step", {f["function"] for f in scan(tmp)})  # agent .decide()
            questions = draft_questions(scan(tmp))
            self.assertEqual(
                questions["is_urgent"],
                {"type": "noul", "instructions": "Does this ticket need a response today?"},
            )
            self.assertEqual(
                list(questions["department"]["criteria"]), ["billing", "technical", "sales"]
            )
            self.assertEqual(questions["frustration"]["criteria"], ["calm", "impatient", "angry"])
            self.assertIn("test_mock", {f["name"] for f in scan(tmp, include_tests=True)})


class MeasureTests(unittest.TestCase):
    def rows(self, n=200):
        return [{"id": f"r{i}", "state": f"s{i}", "labels": {"q": i % 2 == 0}} for i in range(n)]

    def test_threshold_keeps_confident_agreement_and_checks_the_other_half(self):
        # Confident rows agree with the label; unsure rows disagree.
        def predict(state, asked):
            i = int(state[1:])
            sure = i % 4 != 3
            p = (0.99 if i % 2 == 0 else 0.01) if sure else (0.4 if i % 2 == 0 else 0.6)
            return {"q": {"noul": p}}

        questions = {"q": {"type": "noul"}, "unlogged": {"type": "noul"}}
        report = measure(self.rows(), questions, predict, target=0.95)
        self.assertEqual(report["questions"]["unlogged"]["status"], "not_logged")
        q = report["questions"]["q"]
        self.assertEqual(q["status"], "ok")
        self.assertEqual(q["accuracy_handled"], 1.0)
        self.assertAlmostEqual(q["handled"], 0.75, delta=0.1)
        self.assertAlmostEqual(q["agreement_all"], 0.75)
        self.assertEqual(report["rows_fully_handled"], q["handled"])

    def test_failed_requests_are_reported_not_counted_and_no_threshold_escalates(self):
        def predict(state, asked):
            if state == "s0":
                raise ConnectionError("reset")
            return {"q": {"noul": 0.5}}

        report = measure(self.rows(), {"q": {"type": "noul"}}, predict)
        self.assertEqual(report["errors"], [{"id": "r0", "error": "ConnectionError: reset"}])
        self.assertEqual(report["rows"], 199)
        self.assertEqual(report["questions"]["q"]["status"], "no_threshold")
        self.assertEqual(report["rows_fully_handled"], 0.0)


class StopEarlyTests(unittest.TestCase):
    def test_stops_after_the_first_failures_without_sending_the_rest(self):
        calls = []

        def predict(state, asked):
            calls.append(state)
            raise ConnectionError("503 Model connection failed")

        rows = [{"id": f"r{i}", "state": f"s{i}", "labels": {"q": True}} for i in range(50)]
        with self.assertRaisesRegex(ValueError, "first 5 requests failed.*503"):
            measure(rows, {"q": {"type": "noul"}}, predict)
        self.assertEqual(len(calls), 5)


class FitCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("ZILS_", "TYPESAFE_"))}
        self.env["ZILS_CONFIG_PATH"] = str(self.tmp / "credentials.json")

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "zils_sdk.cli", "fit", *args],
            env=self.env,
            text=True,
            capture_output=True,
            timeout=30,
            cwd=self.tmp,
        )

    def test_scan_writes_questions_and_run_replays_the_log(self):
        (self.tmp / "app").mkdir()
        (self.tmp / "app/triage.py").write_text(APP["app/triage.py"])
        result = self.cli("scan", "app", "--questions", "questions.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Decisions Zils can answer  1", result.stdout)
        self.assertRegex(result.stdout, r"yes/no\s+is_urgent\s+triage.py:\d+")
        self.assertNotIn("\033[", result.stdout)  # piped output stays plain
        log = self.tmp / "fit-log.jsonl"
        log.write_text(
            "".join(
                json.dumps({"question": "is_urgent", "state": f"ticket {i}", "answer": True}) + "\n"
                for i in range(4)
            )
        )
        with api() as service:
            self.env.update(ZILS_API_KEY="valid-key", ZILS_BASE_URL=service.url)
            capped = self.cli("run", "--questions", "questions.json", "--max-rows", "3")
            self.assertEqual(capped.returncode, 2)
            self.assertIn("exceed --max-rows", capped.stderr)
            result = self.cli("run", "--questions", "questions.json", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual((report["rows"], report["model"]), (4, "zils-shared"))
        self.assertEqual(report["questions"]["is_urgent"]["status"], "too_few_rows")
        self.assertEqual(report["questions"]["is_urgent"]["inputs_needed"], 146)
        self.assertIn("Sending 4 requests to zils-shared.", result.stderr)

    def test_run_refuses_unfinished_drafts(self):
        (self.tmp / "questions.json").write_text(
            json.dumps({"q": {"type": "noul", "instructions": "TODO"}})
        )
        result = self.cli("run", "--questions", "questions.json")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Replace every TODO", result.stderr)


if __name__ == "__main__":
    unittest.main()
