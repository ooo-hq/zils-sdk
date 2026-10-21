"""`zils fit`: a guided first run. Read the code, show the questions Zils can answer, try one."""

import os
import re
import sys
import time
from pathlib import Path

from .fit import scan, source_files

# The wordmark, sampled from brand-assets/zils-logo-purple.png at 44 columns, two pixels per cell.
LOGO = """\
                   ▄███  ▄▄▄▄▄
  ████████████▀ ▄█ ▀██▀  ▀████
  ███    ████▀ ██         ████
  ▀▀    ████ ▄██  ▄███▄   ████   ▄▄██████▄
  ▄▄▄▄▄███▀ ▄██▄▄  ████   ████  ███▀ ▀▀██
  ███████▀ ██▀▀▀▀  ████   ████  ████▄▄
    ████ ▄██       ████   ████   ▀██████▄
   ████ ▄█▀    ██  ████   ████  ▄▄  ▀█████
  ███▀ ███    ▄██  ████   ████  ██▄   ████
  ██▀ ███████████  ▀████  █████ ████████▀"""
PURPLE = "#7f78f2"  # the brand purple #4b44c8, brightened to read on dark terminals
SPECTRUM = ["#ff7a3d", "#ffae80", "#ff8fc7", "#c39bff", "#8fa8ff"]  # orange to periwinkle
WIDTH = 44
ANSWER_KIND = {"noul": "yes or no", "choice": "one of a list", "score": "a rating"}


class Style:
    """Colour only for an interactive terminal; 24-bit when the terminal says so, else 256."""

    def __init__(self, stream):
        self.on = (
            stream.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"
        )
        self.true = os.environ.get("COLORTERM", "") in ("truecolor", "24bit")

    def code(self, text, code):
        return f"\033[{code}m{text}\033[0m" if self.on else text

    def rgb(self, text, hex_colour):
        r, g, b = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
        if self.true:
            return self.code(text, f"38;2;{r};{g};{b}")
        cube = 16 + 36 * round(r / 51) + 6 * round(g / 51) + round(b / 51)
        return self.code(text, f"38;5;{cube}")

    def bold(self, text):
        return self.code(text, "1")

    def dim(self, text):
        return self.code(text, "2")

    def spectrum(self, width):
        """A rule that runs through the brand colours, one cell at a time."""
        if not self.on:
            return "─" * width
        cells = []
        for i in range(width):
            pos = i / max(width - 1, 1) * (len(SPECTRUM) - 1)
            a, b = SPECTRUM[int(pos)], SPECTRUM[min(int(pos) + 1, len(SPECTRUM) - 1)]
            t = pos - int(pos)
            mix = "#" + "".join(
                f"{round(int(a[k : k + 2], 16) * (1 - t) + int(b[k : k + 2], 16) * t):02x}"
                for k in (1, 3, 5)
            )
            cells.append(self.rgb("━", mix))
        return "".join(cells)


def title(item):
    """The question in plain words: the prompt's own question if the scan found one."""
    question = item["question"]
    asked = (question.get("instructions") or "").strip()
    # Use the prompt's own question only when it reads as one plain sentence.
    if re.fullmatch(r"[A-Z][^{}$`<>\n]{4,58}\?", asked):
        return asked
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", item["name"]).replace("_", " ").lower().split()
    words = [w for w in words if w not in ("ai", "llm", "with", "get")]
    if question["type"] == "noul":
        if words and words[0] in ("is", "has", "should", "can"):
            return f"{words[0].capitalize()} this {' '.join(words[1:])}?"
        return f"Is this {' '.join(words)}?"
    if words and words[-1] == "name":
        words = words[:-1]
    return (
        f"Which {' '.join(words)}?" if question["type"] == "choice" else f"How {' '.join(words)}?"
    )


VERBS = {
    "categorize": "Categorize",
    "classify": "Classify",
    "determine": "Decide",
    "decide": "Decide",
    "check": "Check",
    "choose": "Choose",
    "pick": "Pick",
    "select": "Choose",
    "route": "Route",
    "detect": "Spot",
    "label": "Label",
    "rate": "Rate",
    "score": "Rate",
    "assign": "Assign",
    "find": "Find",
    "match": "Match",
    "triage": "Triage",
}
NOISE = {"ai", "llm", "with", "get", "response", "run", "do", "handle", "fn", "the"}


def words(name):
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ").replace("-", " ")
    return [w for w in spaced.lower().split() if w]


def describe(item):
    """What the decision does in the product, in plain words, from the names the code uses."""
    kind = item["question"]["type"]
    action = [w for w in words(item["function"]) if w not in NOISE]
    field = [w for w in words(item["name"]) if w not in NOISE and w != "name"]
    if kind == "noul":
        rest = action[1:] if action[:1] in (["is"], ["has"], ["should"]) else action or field
        return f"Check whether it's {' '.join(rest)}" if rest else "Answer a yes-or-no question"
    if action and action[0] in VERBS:
        return f"{VERBS[action[0]]} {' '.join(action[1:])}".strip()
    extra = [w for w in action if w not in field]
    verb = "Rate" if kind == "score" else "Choose"
    return f"{verb} {' '.join(field)}" + (f" ({' '.join(extra)})" if extra else "")


def relative(location, base):
    path, line = location.rsplit(":", 1)
    try:
        return f"{Path(path).resolve().relative_to(base)}:{line}"
    except ValueError:
        return location


def say(text="", indent=2):
    print(" " * indent + text if text else "")


def step(style, number, heading):
    print()
    say(f"{style.rgb(str(number), PURPLE)}  {style.bold(heading)}")


def ask(prompt):
    sys.stderr.write(prompt)
    sys.stderr.flush()
    line = sys.stdin.readline()
    if not line:
        raise EOFError
    return line.strip()


def bar(style, value, width=20):
    filled = round(value * width)
    return style.rgb("█" * filled, SPECTRUM[-1]) + style.dim("░" * (width - filled))


def try_question(style, item, client):
    """Ask Zils one found question about an example the user pastes."""
    question = dict(item["question"])
    question["instructions"] = title(item)
    kind = question["type"]
    if kind == "choice" and not question.get("criteria"):
        options = ask("     What answers can it give? Separate them with commas: ")
        question["criteria"] = {o.strip(): None for o in options.split(",") if o.strip()}
        if len(question["criteria"]) < 2:
            say("Give at least two answers to choose from.", 5)
            return
    if kind == "score" and not question.get("criteria"):
        levels = ask("     List the ratings from lowest to highest, separated by commas: ")
        question["criteria"] = [level.strip() for level in levels.split(",") if level.strip()]
        if len(question["criteria"]) < 2:
            say("Give at least two ratings.", 5)
            return
    question = {k: v for k, v in question.items() if k in ("type", "instructions", "criteria")}
    say(style.dim("Paste an example of what this code sees, then press Enter twice:"), 5)
    lines = []
    while True:
        line = sys.stdin.readline()
        if not line or (not line.strip() and lines):
            break
        if line.strip():
            lines.append(line.rstrip("\n"))
    example = "\n".join(lines).strip()
    if not example:
        say("Nothing to ask about yet. Paste an example next time.", 5)
        return
    started = time.monotonic()
    result = client.system_one(state=example, questions={"q": question})
    seconds = time.monotonic() - started
    answer = result.model_dump(mode="json")["answers"]["q"]
    if kind == "noul":
        p = answer["noul"]
        verdict, sure = ("yes", p) if p >= 0.5 else ("no", 1 - p)
    else:
        probabilities = answer["probabilities"]
        best = max(probabilities, key=probabilities.get)
        sure = probabilities[best]
        verdict = best if kind == "choice" else question["criteria"][int(best)]
    print()
    say(f"Zils says  {style.bold(str(verdict))}", 5)
    say(f"{bar(style, sure)}  {sure:.0%} sure, in {seconds:.1f} s", 5)


def try_loop(style, decisions):
    """Let the user ask Zils any listed question; explain instead when that isn't possible."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        say("Run zils fit in a terminal to ask Zils one of these questions.", 5)
        return
    try:
        from . import ZilsClient
        from .credentials import load_credentials

        credentials = load_credentials()
    except ValueError:
        say("Trying a question needs a Zils account. Run zils login, then zils fit again.", 5)
        return
    with ZilsClient(**credentials) as client:
        while True:
            try:
                choice = ask(
                    f"     Pick a question to try (1–{len(decisions)}), or press Enter to finish: "
                )
            except EOFError:
                return
            if not choice:
                return
            if not choice.isdigit() or not 1 <= int(choice) <= len(decisions):
                say(f"Enter a number from 1 to {len(decisions)}.", 5)
                continue
            item = decisions[int(choice) - 1]
            say(f"{style.bold(describe(item))}  {style.dim(title(item))}", 5)
            try:
                try_question(style, item, client)
            except EOFError:
                return
            except Exception as error:
                say(f"Zils couldn't answer that: {error}", 5)
            print()


def guide(path, include_tests=False):
    style = Style(sys.stdout)
    root = Path(path).resolve()
    base = root if root.is_dir() else root.parent
    print()
    for line in LOGO.splitlines():
        print(style.rgb(line, PURPLE))
    print("  " + style.spectrum(WIDTH - 2))
    print()
    say("Zils is a small model that answers yes-or-no and pick-one questions")
    say("inside your app. This finds where your code already asks an AI those")
    say(f"questions, so you can see what Zils would take over in {style.bold(root.name)}.")

    files = source_files(root, include_tests)
    found = scan(root, include_tests, files=files)
    decisions = [f for f in found if f["kind"] == "decision"]
    switches = [f for f in found if f["kind"] == "drop-in"]
    text = [f for f in found if f["kind"] == "text"]

    step(style, 1, "Read your code")
    calls = f"{len(found)} place{'' if len(found) == 1 else 's'} call" + (
        "s" if len(found) == 1 else ""
    )
    say(
        f"Read {len(files):,} files. {calls} an AI model."
        if found
        else f"Read {len(files):,} files. Nothing here calls an AI model yet.",
        5,
    )
    if not found:
        print()
        return

    step(style, 2, "Find the questions")
    if decisions:
        many = len(decisions) != 1
        say(f"Your app asks an AI to make {len(decisions)} quick decision{'s' if many else ''}.", 5)
        say(f"Zils could make {'these' if many else 'it'} instead:", 5)
        print()
        for n, item in enumerate(decisions, 1):
            say(f"{style.bold(str(n).rjust(2))}  {describe(item)}", 5)
            where = Path(item["location"]).name  # file:line; zils fit scan shows full paths
            say(f"    {style.dim(item['function'] + '()  ' + where)}", 5)
    if switches:
        print()
        say("Your app already sends some decisions to a decision model. Your", 5)
        say("developer can point it at Zils by changing the address and model name.", 5)
        for item in switches:
            say(f"  {style.dim(relative(item['location'], base))}", 5)  # the one file to change
    if text:
        print()
        if len(text) == 1:
            say("The other one writes text, like a reply or a summary.", 5)
            say("Zils doesn't write text, so that stays as it is.", 5)
        else:
            say(f"The other {len(text)} write text, like replies and summaries.", 5)
            say("Zils doesn't write text, so those stay as they are.", 5)
    print()
    say(style.dim("Developers: zils fit scan lists the exact code for each."), 5)
    if not decisions:
        print()
        return

    step(style, 3, "Try one")
    try_loop(style, decisions)

    step(style, 4, "Check it on your real traffic")
    say("Before switching anything, record what your current model answers for a", 5)
    say("few days, then compare Zils on the same inputs. Start with", 5)
    say(style.dim("zils fit scan --questions questions.json"), 7)
    say(style.dim("zils fit run --help"), 7)
    print()


def main_args(argv):
    """`zils fit` and `zils fit <folder>` run the guide; `scan` and `run` stay as subcommands."""
    if argv[:1] == ["fit"] and (
        len(argv) == 1 or argv[1] not in ("scan", "run", "start", "-h", "--help")
    ):
        return ["fit", "start", *argv[1:]]
    return argv
