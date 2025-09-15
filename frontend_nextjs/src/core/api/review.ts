import { fetchStream } from "../sse";

import { resolveServiceURL } from "./resolve-service-url";
// ⛔️ 不再引入 StreamEvent

// 代表“解析后的后端 payload”
export type ReviewEventPayload =
  | {
      type: 'text';
      text: string;
    }
  | {
      type: 'metadata';
      metadata: {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        references: Record<string, any>;
      };
    }
  | {
      type: 'reference_item';
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      item: any;
    };

// 我们对外返回给 UI 的事件结构
type AppStreamEvent<T> = {
  type: string;
  data: T;
};

// 1. 用于生成大纲
export async function* generateOutlineStream(
  topic: string,
  language: string,
  options: { abortSignal?: AbortSignal } = {},
): AsyncGenerator<AppStreamEvent<ReviewEventPayload>> {
  try {
    const stream = fetchStream(resolveServiceURL("/api/outline"), {
      body: JSON.stringify({ topic, language }),
      signal: options.abortSignal,
    });

    for await (const event of stream) {
      // 后端通常会发 data: string，这里解析并标注为我们期望的 payload
      const parsed = JSON.parse(event.data) as ReviewEventPayload;

      yield {
        type: event.event, // 映射为外层 type，符合你现有 UI 的使用
        data: parsed,
      };
    }
  } catch (e) {
    console.error("Error generating outline:", e);
    throw e;
  }
}

// 2. 用于生成综述
export async function* generateReviewStream(
  topic: string,
  outline: string,
  options: { abortSignal?: AbortSignal } = {},
): AsyncGenerator<AppStreamEvent<ReviewEventPayload>> {
  try {
    const stream = fetchStream(resolveServiceURL("/api/review"), {
      body: JSON.stringify({ topic, outline }),
      signal: options.abortSignal,
    });

    for await (const event of stream) {
      const parsed = JSON.parse(event.data) as ReviewEventPayload;

      yield {
        type: event.event,
        data: parsed,
      };
    }
  } catch (e) {
    console.error("Error generating review:", e);
    throw e;
  }
}
