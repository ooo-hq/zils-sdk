import { ZilsClient, noul, choice, score, type RankResult } from '../src';

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
  const ranked: RankResult = await client.rank({ items: [{ title: 'A' }, 'B'], instructions: 'Most useful first' });
  const index: number = ranked.ranking[0].index;
  const relativeStrength: number = ranked.ranking[0].score;
  const grouped: RankResult = await client.rank({ items: ['A', 'B', 'C'], instructions: 'Rank', groupSize: 4 });
  // @ts-expect-error Group size must be numeric.
  client.rank({ items: ['A', 'B'], instructions: 'Rank', groupSize: '4' });
  // @ts-expect-error Ranking requires a list, not a single string.
  client.rank({ items: 'A', instructions: 'Most useful first' });
  // @ts-expect-error Ranking items must be JSON values.
  client.rank({ items: ['A', undefined], instructions: 'Most useful first' });
  // @ts-expect-error Ranking requires a criterion.
  client.rank({ items: ['A', 'B'] });
  return { probability, label, billable, index, relativeStrength, grouped };
}
void typed;
