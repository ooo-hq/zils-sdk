import { appendFileSync } from 'node:fs';

/**
 * Shadow-log a decision function for `zils fit run`. Behaviour is unchanged: each call's
 * input and answer are appended to fit-log.jsonl (or ZILS_FIT_LOG). The log holds real
 * inputs, so it is created owner-only. Failed calls are not logged, and a logging error
 * never reaches the caller.
 */
export function record<A extends unknown[], R>(question: string, fn: (...args: A) => R): (...args: A) => R {
  return (...args: A): R => {
    const answer = fn(...args);
    const state = args.length === 1 ? args[0] : { args };
    if (answer instanceof Promise) {
      return answer.then(value => { write(question, state, value); return value; }) as R;
    }
    write(question, state, answer);
    return answer;
  };
}

function write(question: string, state: unknown, answer: unknown): void {
  try {
    const line = JSON.stringify({ question, state, answer, ts: Date.now() / 1000 });
    appendFileSync(process.env.ZILS_FIT_LOG || 'fit-log.jsonl', line + '\n', { mode: 0o600 });
  } catch (error) {
    console.error(`zils fit: call not recorded (${(error as Error).name})`);
  }
}
