"""The training walkthrough the MCP server's train_guide tool returns to an agent that has never used Zils."""

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
- group: the case column, so one customer's rows stay in one split and the test is honest.
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
- private: paid ($0.20 per million training tokens, $2 minimum), trained only on Zils's own workers. Agree a
  max_cost_usd with the user; train_check and train_prepare report ceiling_usd, the most the job can cost.

## 5. Submit (train_submit)
Submit once. If a submission's outcome is uncertain, check train_list before trying again; never resubmit
blindly.

## 6. Wait for it to finish (train_status)
Nothing notifies you or the user when training ends; check train_status. A job moves through validating,
awaiting_approval (waiting for a training worker), queued, running and evaluating to completed or failed.
Training usually takes a few minutes once a worker starts it; waiting for a worker can add more. Give the user
the job id so they can follow it on the Training page at zils.ai/train, and check back rather than polling
nonstop.

## 7. Report the result
When status is completed, the result says whether the model was accepted and shows the base model's and the
trained model's accuracy and Brier score on the test split. Report both honestly: if the base model was
already as accurate, say the gain is in calibration only. If it was not accepted, it did not clear the bars
agreed in step 3; say which.

## 8. Use the model
An accepted model has an API model id (train_status shows it). Call it with decide, passing that id as model.
decide needs an inference API key, separate from the training sign-in: the user runs `zils login --api-key`
(or creates a key in the Zils dashboard). Each decide call is billed per input token.
"""
