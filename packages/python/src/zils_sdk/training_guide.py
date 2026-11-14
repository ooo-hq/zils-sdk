"""The training walkthrough for an agent that has never used Zils: the MCP server's train_guide tool returns
GUIDE, and `zils train guide` prints it after COMMANDS for agents working from a shell."""

COMMANDS = """\
# From a shell

The guide below names the Zils MCP tools. Without them, each step is a `zils` command, and a tool's arguments
are flags of the same name (min_accuracy is --min-accuracy; max_cost_usd is --max-cost). Add --help to any
command for its flags. Never put a key in a command; sign in with `zils login`.

- train_prepare: `zils train prepare FILE --decide COLUMN`. It prints the cost ceiling and the submit command.
- train_check: no command of its own. `zils train prepare` reports the ceiling, and `zils train submit` checks
  the files and refuses before sending if the job could cost more than --max-cost.
- train_submit: `zils train submit`.
- train_list, train_status, train_cancel: `zils train list`, `zils train status JOB_ID` (add --watch to wait),
  `zils train cancel JOB_ID`.
- fit_scan, fit_run: `zils fit scan`, `zils fit run`.
- decide: `zils decide`.

"""

GUIDE = """\
# Training a Zils model for the user

A Zils model makes one repeated decision (which team gets a ticket, is a claim eligible) from the information
available when the decision is made. You train it from the user's past decisions. Work through these steps and
check with the user where it says so.

## 1. Look at the data
Read the file's columns and a few rows. Find the column holding the right answer (the decision) and the case
column, if any: one customer, account or thread whose rows belong together.

## 2. Prepare the splits (train_prepare)
Call train_prepare with:
- decide: the answer column.
- read: only the columns known at the moment of deciding. Leave out IDs, columns that give away the answer (the
  agent who handled a ticket, the final status), and anything filled in after the decision (resolution time,
  satisfaction score). train_prepare lists suspected give-aways under "leaks"; drop them unless the user says the
  value really is known when deciding.
- group: the case column, so one customer's rows stay in one split and the test is honest. Left out, a column
  named like a case (conversation_id, customer_id, ...) is used when its values repeat; "group" and "note" in
  the result say which. Name it yourself when the column has another name.
Then tell the user which columns the model reads and which it skips, and point out any answer with few test
examples in the "answers" counts: results for it will be rough.

## 3. Agree the pass bars
Propose min_accuracy and min_brier_improvement, say what they mean, and agree them with the user before
submitting. The trained model is only delivered if, on the held-out test split, its accuracy reaches
min_accuracy (0-1) and its Brier score (how well its confidence matches reality; lower is better) beats the
untrained base model by at least min_brier_improvement. A common start is 0.8 and 0.01; set min_accuracy from
what a wrong answer costs the user, not from what seems easy to pass.

## 4. Ask open or private
Ask the user; never choose for them.
- open: free. Zils network miners train it and receive the training split; they must delete it within 7 days.
  Pass allow_open_tier_training true only after the user agrees to that.
  Limits: 3 open jobs per account in 24 hours, up to 5,000 examples and 5,000,000 training tokens each; past them, train privately.
  The trained model is served by Zils and used through the API; only private models can be downloaded.
- private: paid ($0.20 per million training tokens, $2 minimum), trained only on Zils's own workers. Agree a
  max_cost_usd with the user; train_check and train_prepare report ceiling_usd, the most the job can cost.

## 5. Submit (train_submit)
Submit once. If a submission's outcome is uncertain, check train_list before trying again; never resubmit
blindly.

## 6. Wait for it to finish (train_status)
Nothing notifies you or the user when training ends; check train_status. A job moves through validating,
awaiting_approval (waiting for a training worker), queued, running and evaluating to completed or failed.
Most jobs take 5 to 25 minutes once a worker starts them; waiting for a worker comes on top and cannot be
predicted. train_status shows expected_seconds, the estimate for this job, and then started_at and a rough
expected_finish_at (UTC): tell the user those rather than guessing, and say they are estimates. Give the user
the job id so they can follow it on the Training page at zils.ai/train, and check back rather than polling
nonstop.

## 7. Report the result
When status is completed, the result says whether the model was accepted and shows the base model's and the
trained model's accuracy and Brier score on the test split. Report both honestly: if the base model was
already as accurate, say the gain is in calibration only. If it was not accepted, it did not clear the bars
agreed in step 3; say which.

## Decisions made inside the user's program
When the decisions happen in the user's code (an LLM call that picks a label, a rule, a game or agent choosing
its next move) rather than in a file of past examples, collect the examples from the program first:
- fit_scan finds model calls that make typed decisions and drafts their questions; read the code around each
  call and replace every TODO. A rule or scripted player is not found by the scan: wrap that function with
  record() instead (Python `from zils_sdk import record`, TypeScript `import { record } from '@zils/sdk'`).
- record() logs each call's state and answer. For games, sessions or conversations, log each episode to its own
  file (set ZILS_FIT_LOG per episode) so one episode never lands in both training and test.
- fit_run measures the untrained model against the logged answers. It sends one paid request per distinct
  input: agree the number with the user and pass it as max_rows.
- To train, call train_prepare with the log or folder as input, the questions, and decide set to the question;
  then continue from step 3 (pass bars, open or private, submit, wait, report). Train only on answers that are
  right (real outcomes, or the program's answers when the goal is to copy it), never on inputs you wrote to test.
- After training, run fit_run again with the trained model's API id as model and show the user both results.

## 8. Use the model
An accepted model has an API model id (train_status shows it). Call it with decide, passing that id as model.
decide needs an inference API key, separate from the training sign-in: the user runs `zils login --api-key`
(or creates a key in the Zils dashboard). Each decide call is billed per input token.
"""


def command_line():
    """What `zils train guide` prints."""
    return COMMANDS + GUIDE
