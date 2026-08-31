/**
 * api.ts — HTTP helpers + SSE stream consumer.
 *
 * ponytail: thin fetch wrappers, no axios. SSE via ReadableStream (supports POST).
 */

import type { PlanStep, BackgroundTask } from './state';

// Base URL for the FastAPI backend. In the Electron desktop client the main
// process spawns uvicorn on 127.0.0.1:8001 and App.tsx calls setBaseUrl() once
// it reports ready. In pure-browser dev (`npm run dev:renderer`) it stays '' and
// the Vite /api proxy forwards to localhost:8000.
let baseUrl = '';
let originKnown = false;
let notifyOrigin: () => void = () => {};
const backendReady = new Promise<void>(resolve => { notifyOrigin = resolve; });

export function setBaseUrl(url: string) {
  baseUrl = url.replace(/\/$/, '');
  if (!originKnown) { originKnown = true; notifyOrigin(); }
}

/** Pure-browser dev (`npm run dev:renderer`): the Vite /api proxy is authoritative. */
export function markProxyReady() {
  if (!originKnown) { originKnown = true; notifyOrigin(); }
}

/**
 * Resolves once the real backend origin is known: the Electron main process
 * reports `ready` on 127.0.0.1:8001, browser dev has no boot phase. Components
 * that fetch server data on mount must `await` this — firing requests with an
 * empty base URL during uvicorn startup proxy-ECONNREFUSEDs and (worse)
 * leaves one-shot fetches in a permanent error state.
 */
export function whenBackendReady(): Promise<void> { return backendReady; }

const withBase = (path: string) => baseUrl + path;

// ---- helpers ----

async function get<T>(path: string): Promise<T> {
  const r = await fetch(withBase(path));
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(withBase(path), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

// ---- SSE ----

export type SSEEvent =
  | { type: 'token'; content: string }
  | { type: 'tool_start'; id: string; name: string; args?: Record<string, unknown>; parent_id?: string; kind?: 'subagent' | 'tool' }
  | { type: 'tool_end'; id: string; name: string; status: string; result?: string; execution_time?: number | null }
  | { type: 'plan'; steps: Array<{ id: string; description: string; target: string; depends_on?: string[] }> }
  | { type: 'done' }
  | { type: 'error'; message: string };

export interface SSECallbacks {
  onToolStart?: (id: string, name: string, args?: Record<string, unknown>, parentId?: string, kind?: 'subagent' | 'tool') => void;
  onToolEnd?: (id: string, status: string, result?: string, executionTime?: number, name?: string) => void;
  onPlan?: (steps: PlanStep[]) => void;
  onToken?: (content: string) => void;
  onDone?: () => void;
  onError?: (message: string) => void;
}

export function streamChat(
  query: string,
  threadId: string,
  callbacks: SSECallbacks,
): AbortController {
  const controller = new AbortController();

  fetch(withBase('/api/agent/chat/stream'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, thread_id: threadId }),
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        callbacks.onError?.(`${response.status} ${response.statusText}`);
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        callbacks.onError?.('No response stream');
        return;
      }

      const decoder = new TextDecoder();
      let buffer = '';

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // SSE messages are separated by \n\n
          const parts = buffer.split('\n\n');
          buffer = parts.pop() ?? '';

          for (const part of parts) {
            const lines = part.split('\n');
            for (const line of lines) {
              if (!line.startsWith('data: ')) continue;
              try {
                const event: SSEEvent = JSON.parse(line.slice(6));
                switch (event.type) {
                  case 'tool_start':
                    callbacks.onToolStart?.(event.id, event.name, event.args, event.parent_id, event.kind);
                    break;
                  case 'tool_end':
                    callbacks.onToolEnd?.(event.id, event.status, event.result, event.execution_time ?? undefined, event.name);
                    break;
                  case 'plan':
                    callbacks.onPlan?.(event.steps.map(s => ({ id: s.id, description: s.description, target: s.target })));
                    break;
                  case 'token':
                    callbacks.onToken?.(event.content);
                    break;
                  case 'done':
                    callbacks.onDone?.();
                    break;
                  case 'error':
                    callbacks.onError?.(event.message);
                    break;
                }
              } catch {
                // skip malformed SSE lines
              }
            }
          }
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        callbacks.onError?.(String(err));
      }
    })
    .catch((err) => {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      callbacks.onError?.(String(err));
    });

  return controller;
}

/**
 * streamNotify — SSE stream for a finished background task. The backend runs a
 * 1-2 sentence notifier LLM turn ("agent informs the user"); we surface it as a
 * streaming assistant message. Mirrors the chat stream's event shape (token/done/error).
 */
export function streamNotify(
  threadId: string,
  task: BackgroundTask,
  callbacks: { onToken?: (content: string) => void; onDone?: () => void; onError?: (message: string) => void },
): AbortController {
  const controller = new AbortController();
  const bodyTask = {
    task_id: task.taskId,
    kind: task.kind,
    paper_name: task.paperName,
    status: task.status,
    progress: task.progress,
    error: task.error,
    result: task.result,
    stage: task.stage,
  };

  fetch(withBase('/api/agent/notify/stream'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ thread_id: threadId, task: bodyTask }),
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        callbacks.onError?.(`${response.status} ${response.statusText}`);
        return;
      }
      const reader = response.body?.getReader();
      if (!reader) {
        callbacks.onError?.('No response stream');
        return;
      }
      const decoder = new TextDecoder();
      let buffer = '';
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split('\n\n');
          buffer = parts.pop() ?? '';
          for (const part of parts) {
            for (const line of part.split('\n')) {
              if (!line.startsWith('data: ')) continue;
              try {
                const event = JSON.parse(line.slice(6)) as { type: string; content?: string; message?: string };
                if (event.type === 'token') callbacks.onToken?.(event.content ?? '');
                else if (event.type === 'done') callbacks.onDone?.();
                else if (event.type === 'error') callbacks.onError?.(event.message ?? 'unknown error');
              } catch {
                // skip malformed SSE lines
              }
            }
          }
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        callbacks.onError?.(String(err));
      }
    })
    .catch((err) => {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      callbacks.onError?.(String(err));
    });

  return controller;
}

/**
 * openTaskStream — live SSE feed of background-task state (replaces fast polling).
 *
 * Browser EventSource reconnects natively: on drop it retries and the server
 * resends the full snapshot, so a connection gap heals without client code.
 * Event shape: { type: 'task_snapshot' | 'task_update', task: {...TaskStatus} }.
 * Returns a dispose function.
 */
export function openTaskStream(onTask: (t: BackgroundTask) => void): () => void {
  const es = new EventSource(withBase('/api/agent/tasks/stream'));
  es.onmessage = (ev: MessageEvent) => {
    try {
      const data = JSON.parse(ev.data as string);
      const t = data?.task;
      if (!t) return;
      onTask({
        taskId: String(t.task_id ?? ''),
        kind: t.kind ?? '',
        paperName: t.paper_name ?? '',
        status: (t.status as BackgroundTask['status']) ?? 'pending',
        progress: t.progress ?? '',
        error: t.error ?? null,
        result: t.result ?? null,
        notify: !!t.notify,
        stage: t.stage ?? '',
        createdAt: String(t.created_at ?? ''),
        updatedAt: String(t.updated_at ?? ''),
      });
    } catch {
      // malformed frame → ignore
    }
  };
  // EventSource reconnects itself after network drops; the server replays the
  // snapshot on each (re)connect, so no manual error handling is required here.
  es.onerror = () => { /* keep default auto-reconnect */ };
  return () => es.close();
}

// ---- API calls ----

export const api = {
  // Health / system
  getAgentHealth: () => get<{ status: string; model: string; tools: number }>('/api/agent/health'),
  getIndexStats: () => get<{ backend: string; collection_name: string; count: number }>('/api/index/stats'),

  // Workspace file explorer
  listWorkspace: (path = '.') =>
    get<{ ok: boolean; data: { path: string; entries: Array<{ name: string; is_dir: boolean; size: number | null }> } }>(
      `/api/workspace/list?path=${encodeURIComponent(path)}`
    ),
  readWorkspaceFile: (path: string) =>
    get<{ ok: boolean; data: { path: string; is_binary: boolean; content: string } }>(
      `/api/workspace/read?path=${encodeURIComponent(path)}`
    ),

  // Upload (multipart — no JSON content-type)
  uploadPDF: async (file: File) => {
    const form = new FormData();
    form.append('file', file);
    const r = await fetch(withBase('/api/pdf/process'), { method: 'POST', body: form });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json() as Promise<{ task_id: string; paper_name: string; status: string }>;
  },

  // Background task stack (agent-driven + upload/index tasks)
  listTasks: async () => {
    const raw = await get<Array<{
      task_id: string; kind?: string; paper_name?: string; status: string;
      progress?: string; error?: string | null;
      result?: Record<string, unknown> | null; notify?: boolean;
      stage?: string; created_at?: string; updated_at?: string;
    }>>('/api/agent/tasks');
    return raw.map(t => ({
      taskId: t.task_id,
      kind: t.kind ?? '',
      paperName: t.paper_name ?? '',
      status: t.status as BackgroundTask['status'],
      progress: t.progress ?? '',
      error: t.error ?? null,
      result: t.result ?? null,
      notify: !!t.notify,
      stage: t.stage ?? '',
      createdAt: String(t.created_at ?? ''),
      updatedAt: String(t.updated_at ?? ''),
    })) as BackgroundTask[];
  },

  // Indexing
  runIndexing: (ragChunksPath: string) =>
    post<{ task_id: string; status: string }>('/api/index/run', { rag_chunks_path: ragChunksPath, config_path: '' }),
};
