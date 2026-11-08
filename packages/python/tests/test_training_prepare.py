import asyncio
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from zils_sdk import training_data, training_prepare

TEAMS = {
    "Billing": [
        "I was charged twice this month",
        "Please send the invoice for March",
        "My card was declined at renewal",
    ],
    "Tech support": [
        "The app crashes when I open it",
        "Export fails with an error",
        "The login link never arrives",
    ],
    "Sales": [
        "Can we add twenty more seats",
        "Do you offer an annual plan",
        "We would like a demo for the team",
    ],
}


def tickets(n=30):
    """n support tickets: an id, the message, a channel, the team that handled it, and the case it belongs to."""
    rows = []
    for i in range(n):
        team = list(TEAMS)[i % 3]
        rows.append(
            {
                "ticket_id": f"T-{i}",
                "message": f"{TEAMS[team][i % 3]} (ref {i})",
                "channel": ["email", "chat"][i % 2],
                "team": team,
                "case": f"C-{i // 2}",
            }
        )
    return rows


def write_csv(path, rows, delimiter=","):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_split(directory, split):
    return [
        json.loads(line)
        for line in (directory / f"{split}.jsonl").read_text().splitlines()
        if line.strip()
    ]


class PrepareTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.csv = write_csv(self.root / "tickets.csv", tickets())

    def prepare(self, source=None, **options):
        return training_prepare.prepare(
            source or self.csv,
            options.pop("decide", "team"),
            out=options.pop("out", self.root / "out"),
            **options,
        )

    def test_writes_three_splits_the_validator_accepts(self):
        result = self.prepare(read=["message", "channel"], group="case")
        out = self.root / "out"
        files = {split: out / f"{split}.jsonl" for split in training_data.SPLITS}
        checked = training_data.check(files)
        self.assertEqual(sum(v["count"] for v in checked["files"].values()), 30)
        self.assertEqual(result["ceiling_usd"], checked["ceiling_usd"])
        self.assertEqual({k: v["count"] for k, v in checked["files"].items()}, result["counts"])
        self.assertIn("zils train submit", result["next"])

    def test_cases_follow_the_website_format(self):
        self.prepare(
            read=["message", "channel"],
            group="case",
            name="support-routing",
            question="Which team should handle this ticket?",
        )
        case = read_split(self.root / "out", "train")[0]
        self.assertEqual(set(case), {"id", "group_id", "family", "state", "question", "label"})
        self.assertEqual(case["family"], "support-routing")
        self.assertEqual(case["state"]["decision"], "Which team should handle this ticket?")
        self.assertEqual(set(case["state"]["information"]), {"channel", "message"})
        self.assertEqual(
            case["question"], {"type": "choice", "criteria": {t: t for t in sorted(TEAMS)}}
        )
        self.assertTrue(case["id"].startswith("case-") and case["group_id"].startswith("group-"))

    def test_every_answer_reaches_every_split_and_groups_stay_together(self):
        self.prepare(read=["message"], group="case")
        out = self.root / "out"
        seen = {}
        for split in training_data.SPLITS:
            rows = read_split(out, split)
            self.assertEqual({r["label"] for r in rows}, set(TEAMS), split)
            for r in rows:
                self.assertEqual(seen.setdefault(r["group_id"], split), split)

    def test_the_same_file_always_gives_the_same_splits(self):
        self.prepare(read=["message"], group="case", out=self.root / "a")
        self.prepare(read=["message"], group="case", out=self.root / "b")
        for split in training_data.SPLITS:
            self.assertEqual(
                (self.root / "a" / f"{split}.jsonl").read_bytes(),
                (self.root / "b" / f"{split}.jsonl").read_bytes(),
            )

    def test_reads_every_other_column_unless_told_but_never_an_id(self):
        result = self.prepare(group="case")
        self.assertEqual(
            result["read"], ["channel", "message"]
        )  # ticket_id is an id; team is the answer; case groups
        self.assertIn("ticket_id", result["skipped"])

    def test_a_column_that_gives_the_answer_away_is_left_out_unless_named(self):
        # Who handled a ticket is only known after the team was chosen, and gives the team away.
        agents = {"Billing": ["Ana", "Bo"], "Tech support": ["Raj", "Kim"], "Sales": ["Lee", "Max"]}
        rows = [{**row, "agent": agents[row["team"]][i % 2]} for i, row in enumerate(tickets())]
        source = write_csv(self.root / "leaky.csv", rows)
        result = self.prepare(source, group="case")
        self.assertEqual(result["read"], ["channel", "message"])
        self.assertIn("agent", result["skipped"])
        self.assertEqual([leak["column"] for leak in result["leaks"]], ["agent"])
        self.assertIn("gives away", result["leaks"][0]["reason"])
        # Named on purpose, it is read, with the warning kept.
        named = self.prepare(
            source, group="case", read=["message", "agent"], out=self.root / "named"
        )
        self.assertIn("agent", named["read"])
        self.assertEqual([leak["column"] for leak in named["leaks"]], ["agent"])

    def test_people_see_the_leak_warning_too(self):
        from zils_sdk.training_cli import render_prepared

        agents = {"Billing": ["Ana", "Bo"], "Tech support": ["Raj", "Kim"], "Sales": ["Lee", "Max"]}
        rows = [{**row, "agent": agents[row["team"]][i % 2]} for i, row in enumerate(tickets())]
        result = self.prepare(write_csv(self.root / "leaky.csv", rows), group="case")
        self.assertIn("agent", render_prepared(result).split("Reads:")[1])
        self.assertIn("gives away", render_prepared(result))

    def test_ordinary_columns_are_not_called_leaks(self):
        # A common answer is not a leak: most tickets go to Billing, and channel says little about the team.
        rows = [
            {**row, "team": "Billing" if i % 10 else row["team"]}
            for i, row in enumerate(tickets(90))
        ]
        source = write_csv(self.root / "skewed.csv", rows)
        self.assertEqual(self.prepare(source, group="case")["leaks"], [])
        self.assertEqual(self.prepare(group="case", out=self.root / "plain")["leaks"], [])

    def test_tsv_json_and_jsonl_inputs(self):
        rows = tickets()
        sources = [
            write_csv(self.root / "t.tsv", rows, "\t"),
            self.root / "t.json",
            self.root / "t.jsonl",
        ]
        sources[1].write_text(json.dumps(rows))
        sources[2].write_text("".join(json.dumps(r) + "\n" for r in rows))
        for i, source in enumerate(sources):
            with self.subTest(source=source.suffix):
                self.assertEqual(
                    sum(
                        self.prepare(
                            source, read=["message"], group="case", out=self.root / f"o{i}"
                        )["counts"].values()
                    ),
                    30,
                )

    def test_nested_json_values_become_text_it_reads(self):
        rows = [
            {
                "state": {"health": 60 + i, "enemy": ["imp", "demon", "none"][i % 3]},
                "action": ["FIRE", "MOVE", "USE"][i % 3],
            }
            for i in range(18)
        ]
        source = self.root / "doom.jsonl"
        source.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.prepare(source, decide="action")
        case = read_split(self.root / "out", "train")[0]
        self.assertIsInstance(case["state"]["information"]["state"], str)
        self.assertIn("health", case["state"]["information"]["state"])

    def test_refusals_explain_what_to_fix(self):
        rows = tickets()
        cases = {
            "not a column": ({"decide": "queue"}, rows, "queue"),
            "answer is missing": ({}, rows[:5] + [{**rows[5], "team": ""}] + rows[6:], "row 7"),
            "conflicting duplicate": (
                {"read": ["message"]},
                rows + [{**rows[0], "team": "Sales", "ticket_id": "T-99", "case": "C-99"}],
                "conflicting",
            ),
            "too rare": (
                {},
                [r for r in rows if r["team"] != "Sales"]
                + [r for r in rows if r["team"] == "Sales"][:2],
                "Sales",
            ),
            "free text answer": ({"decide": "message"}, rows, "different values"),
            "only one answer": (
                {},
                [{**r, "team": "Billing"} for r in rows],
                "two different answers",
            ),
            "reads the answer": ({"read": ["message", "team"]}, rows, "answer"),
        }
        for label, (options, data, phrase) in cases.items():
            with self.subTest(label), self.assertRaisesRegex(ValueError, phrase):
                self.prepare(
                    write_csv(self.root / "bad.csv", data),
                    out=self.root / label.replace(" ", "-"),
                    **{"group": "case", **options},
                )

    def test_never_overwrites_existing_files(self):
        out = self.root / "out"
        out.mkdir()
        (out / "test.jsonl").write_text("keep me\n")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare(group="case")
        self.assertEqual((out / "test.jsonl").read_text(), "keep me\n")
        self.assertFalse((out / "train.jsonl").exists())


class PrepareCommandTests(unittest.TestCase):
    def test_cli_prepares_without_signing_in(self):
        with tempfile.TemporaryDirectory() as root:
            source = write_csv(Path(root) / "tickets.csv", tickets())
            out = Path(root) / "splits"
            # No credentials anywhere: a command that needed a login would fail here.
            env = {
                **os.environ,
                "HOME": root,
                "ZILS_CONFIG_PATH": str(Path(root) / "none.json"),
                "ZILS_API_KEY": "",
            }
            done = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "zils_sdk.cli",
                    "train",
                    "prepare",
                    str(source),
                    "--decide",
                    "team",
                    "--read",
                    "message",
                    "--group",
                    "case",
                    "--out",
                    str(out),
                    "--json",
                ],
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            result = json.loads(done.stdout)
            self.assertEqual(sum(result["counts"].values()), 30)
            self.assertTrue((out / "calibration.jsonl").exists())

    def test_mcp_tool_prepares_too(self):
        from zils_sdk import mcp

        with tempfile.TemporaryDirectory() as root:
            source = write_csv(Path(root) / "tickets.csv", tickets())
            server = mcp.build_server()
            names = {tool.name for tool in asyncio.run(server.list_tools())}
            self.assertIn("train_prepare", names)
            _, result = asyncio.run(
                server.call_tool(
                    "train_prepare",
                    {
                        "input": str(source),
                        "decide": "team",
                        "read": ["message"],
                        "group": "case",
                        "out": str(Path(root) / "splits"),
                    },
                )
            )
            self.assertEqual(sum(result["counts"].values()), 30)


if __name__ == "__main__":
    unittest.main()
