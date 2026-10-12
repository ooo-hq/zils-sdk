import {
  TypeSafeClient, type TypeSafeClientConfig, type APIPromise, type Questions,
  type SystemOneRequest, type SystemOneResult as BaseResult, type RequestOptions,
  type Usage as BaseUsage,
} from '@typesafe-ai/sdk';
import { prepareRank, rankAnswers, type RankRequest, type RankResult } from './ranking.js';

export type { RankRequest, RankResult, RankedItem } from './ranking.js';

export {
  noul, choice, score, APIError, APIConnectionError, APITimeoutError,
  APIUserAbortError, AuthenticationError, RateLimitError, TypeSafeError as ZilsError,
} from '@typesafe-ai/sdk';
export type {
  Question, Questions, SystemOneRequest, RequestOptions, RetryPolicy, ModelCard,
  NoulQuestion, ChoiceQuestion, ScoreQuestion, NoulResponse, ChoiceResponse, ScoreResponse,
  JsonValue, EntryType,
} from '@typesafe-ai/sdk';

export interface Usage extends BaseUsage {
  /** Logical billing tokens; absent on older runtimes. input_tokens counts resources. */
  readonly billable_input_tokens?: number;
}
export interface SystemOneResult<Q extends Questions> extends BaseResult<Q> {
  readonly usage: Usage;
}
export type ZilsClientConfig = Omit<TypeSafeClientConfig, 'dangerouslyAllowBrowser'>;

const DEFAULT_BASE_URL = 'https://training.zils.ai/decision';
function env(name: string): string | undefined {
  return typeof process === 'undefined' ? undefined : process.env[name]?.trim() || undefined;
}
function baseURL(value: string): string {
  let parsed: URL;
  try { parsed = new URL(value); }
  catch { throw new Error('Use a valid HTTPS base URL.'); }
  const loopback = parsed.hostname === 'localhost' || parsed.hostname === '[::1]' || /^127(?:\.\d{1,3}){3}$/.test(parsed.hostname);
  if ((parsed.protocol !== 'https:' && !(parsed.protocol === 'http:' && loopback)) ||
      parsed.username || parsed.password || parsed.search || parsed.hash || /\s/.test(value) || parsed.port === '0') {
    throw new Error('Use an HTTPS base URL without credentials, query, or fragment; HTTP is allowed only on loopback.');
  }
  return parsed.href.replace(/\/+$/, '');
}

/** Zils defaults over the compatible TypeSafe client; automatic retries are off. */
export class ZilsClient extends TypeSafeClient {
  constructor(config: ZilsClientConfig = {}) {
    const apiKey = config.apiKey ?? env('ZILS_API_KEY') ?? '';
    if (!/^[\x21-\x7E]+$/.test(apiKey)) throw new Error('Set ZILS_API_KEY to a nonempty API key without whitespace.');
    const model = config.defaultModel ?? env('ZILS_DEFAULT_MODEL') ?? 'zils-shared';
    if (!model.trim() || model.length > 128) throw new Error('Model must contain 1–128 characters.');
    const transport = config.fetch ?? globalThis.fetch;
    super({
      ...config, apiKey,
      baseURL: baseURL(config.baseURL ?? env('ZILS_BASE_URL') ?? DEFAULT_BASE_URL),
      defaultModel: model,
      timeout: config.timeout ?? 60_000,
      retry: { ...config.retry, maxRetries: config.retry?.maxRetries ?? 0 },
      logLevel: config.logLevel ?? 'off',
      dangerouslyAllowBrowser: false,
      // Decisions are paid requests: never forward them through an HTTP redirect.
      fetch: (url, init) => transport(url, { ...init, redirect: 'error' }),
    });
  }

  override systemOne<const Q extends Questions>(request: SystemOneRequest<Q>, options?: RequestOptions): APIPromise<SystemOneResult<Q>> {
    return super.systemOne(request, options) as APIPromise<SystemOneResult<Q>>;
  }

  /** Rank 2–10 JSON items in one paid request; scores are relative strengths, not confidence. */
  async rank(request: RankRequest, options?: RequestOptions): Promise<RankResult> {
    const body = prepareRank(request);
    const count = request.items.length;
    const result = await this.systemOne(body, options);
    if (typeof result.model !== 'string' || !result.model.trim() || !result.usage ||
        ![result.usage.input_tokens, result.usage.output_tokens,
          ...(result.usage.billable_input_tokens === undefined ? [] : [result.usage.billable_input_tokens])]
          .every(value => typeof value === 'number' && Number.isInteger(value) && value >= 0)) {
      throw new Error('Ranking response contains invalid model or usage metadata.');
    }
    return {
      model: result.model,
      ranking: rankAnswers(result.answers, count),
      comparisons: Object.keys(body.questions).length,
      usage: result.usage,
    };
  }
}
