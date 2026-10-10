import { ZilsClient, noul, choice, score } from '../src';

async function typed(client: ZilsClient) {
  const result = await client.systemOne({
    state: 'A refund request',
    questions: {
      refund: noul('Refund?'),
      team: choice('Team?', { billing: 'Payments', other: null }),
      urgency: score('Urgency?', ['Low', 'High']),
    },
  });
  const probability: number = result.answers.refund.noul;
  const label: 'billing' | 'other' = result.answers.team.choice;
  const billable: number | undefined = result.usage.billable_input_tokens;
  // @ts-expect-error Answers retain the actual question names.
  result.answers.unknown;
  // @ts-expect-error A yes/no answer is not a choice answer.
  result.answers.refund.choice;
  return { probability, label, billable };
}
void typed;
