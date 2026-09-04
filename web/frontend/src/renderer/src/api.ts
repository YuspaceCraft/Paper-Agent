/**
 * api.ts — HTTP helpers + SSE stream consumer.
 *
 * ponytail: thin fetch wrappers, no axios. SSE via ReadableStream (supports POST).
 */

import type { PlanStep, BackgroundTask, AgentMode } from './state';

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

async function put<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(withBase(path), {
    method: 'PUT',
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
  | { type: 'mode'; mode: 'react' | 'plan'; source: 'user' | 'auto' }
  | { type: 'plan'; steps: Array<{ id: string; description: string; target: string; depends_on?: string[]; status?: string }> }
  | { type: 'plan_step'; id: string; status: string; name?: string; description?: string; output?: string }
  | { type: 'plan_progress'; done: number; total: number }
  | { type: 'plan_verify'; status: string; done: number; total: number; outstanding: Array<{ id: string; description: string; reason: string }> }
  | { type: 'done' }
  | { type: 'error'; message: string };

export interface PlanVerdict {
  status: string;
  done: number;
  total: number;
  outstanding: Array<{ id: string; description: string; reason: string }>;
}

export interface SSECallbacks {
  onToolStart?: (id: string, name: string, args?: Record<string, unknown>, parentId?: string, kind?: 'subagent' | 'tool') => void;
  onToolEnd?: (id: string, status: string, result?: string, executionTime?: number, name?: string) => void;
  onMode?: (mode: 'react' | 'plan', source: 'user' | 'auto') => void;
  onPlan?: (steps: PlanStep[]) => void;
  onPlanStep?: (id: string, status: PlanStep['status'], output?: string) => void;
  onPlanProgress?: (done: number, total: number) => void;
  onPlanVerify?: (verdict: PlanVerdict) => void;
  onToken?: (content: string) => void;
  onDone?: () => void;
  onError?: (message: string) => void;
}

export function streamChat(
  query: string,
  threadId: string,
  mode: AgentMode,
  callbacks: SSECallbacks,
): AbortController {
  const controller = new AbortController();

  fetch(withBase('/api/agent/chat/stream'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, thread_id: threadId, mode }),
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
                  case 'mode':
                    callbacks.onMode?.(event.mode, event.source);
                    break;
                  case 'plan':
                    callbacks.onPlan?.(event.steps.map(s => ({
                      id: s.id,
                      description: s.description,
                      target: s.target,
                      depends_on: s.depends_on,
                      status: (s.status as PlanStep['status'] | undefined) ?? 'pending',
                    })));
                    break;
                  case 'plan_step':
                    callbacks.onPlanStep?.(event.id, (event.status as PlanStep['status']) ?? 'pending', event.output);
                    break;
                  case 'plan_progress':
                    callbacks.onPlanProgress?.(event.done, event.total);
                    break;
                  case 'plan_verify':
                    callbacks.onPlanVerify?.({ status: event.status, done: event.done, total: event.total, outstanding: event.outstanding });
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

  // Workspace file explorer (root 决定基准根: project=文献问答+写作, experiments=实验)
  listWorkspace: (path = '.', root: WorkspaceRoot = 'project') =>
    get<{ ok: boolean; data: { path: string; entries: Array<{ name: string; is_dir: boolean; size: number | null }> } }>(
      `/api/workspace/list?path=${encodeURIComponent(path)}&root=${root}`
    ),
  readWorkspaceFile: (path: string, root: WorkspaceRoot = 'project') =>
    get<{ ok: boolean; data: { path: string; is_binary: boolean; content: string } }>(
      `/api/workspace/read?path=${encodeURIComponent(path)}&root=${root}`
    ),

  // ---- Path settings（可配置项目路径 / 实验根，v10 界面）----
  getSettings: () => get<Settings>('/api/settings'),
  updateSettings: (body: { project_path?: string | null; experiments_path?: string | null }) =>
    put<Settings>('/api/settings', body),
  /** 只读目录浏览（路径选择器用）：空 path → 盘符列表；否则该目录的子目录。 */
  browseDir: (path = '') =>
    get<{ ok: boolean; data: { path: string; entries: Array<{ name: string; is_dir: boolean }> } }>(
      `/api/workspace/browse?path=${encodeURIComponent(path)}`
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

  // ---- Creation (写作工作区, v10 / Phase B) ----
  listCreationDocs: (status = '') =>
    get<{ docs: CreationDocMeta[] }>(`/api/creation/docs${status ? `?status=${encodeURIComponent(status)}` : ''}`),
  getCreationDoc: (docId: string) => get<CreationDoc>(`/api/creation/docs/${encodeURIComponent(docId)}`),
  createCreationDoc: (title: string) =>
    post<{ doc_id: string }>('/api/creation/docs', { title }),
  setCreationOutline: (docId: string, outline: SectionOutline[]) =>
    put<{ outline: SectionOutline[] }>(`/api/creation/docs/${encodeURIComponent(docId)}/outline`, { outline }),
  writeCreationSection: (docId: string, sectionId: string, content: string) =>
    put<{ doc_id: string; section_id: string; status: string; word_count: number }>(
      `/api/creation/docs/${encodeURIComponent(docId)}/sections/${encodeURIComponent(sectionId)}`,
      { content },
    ),
  /** 导出 docx → 触发浏览器/Electron 下载（a[download]）。 */
  downloadDocx: async (docId: string) => {
    const r = await fetch(withBase(`/api/creation/docs/${encodeURIComponent(docId)}/export-docx`));
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${docId}.docx`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  },

  // ---- Experiments（实验工作区, v10 / Phase D）----
  listExperimentProjects: () => get<{ projects: string[] }>('/api/experiments/projects'),
  listExperiments: (project = '') =>
    get<{ experiments: Experiment[] }>(`/api/experiments${project ? `?project=${encodeURIComponent(project)}` : ''}`),
  getExperiment: (expId: string) =>
    get<Experiment>(`/api/experiments/${encodeURIComponent(expId)}`),
  getExperimentMetrics: (expId: string) =>
    get<{ exp_id: string; metrics: Record<string, unknown> }>(`/api/experiments/${encodeURIComponent(expId)}/metrics`),
  getExperimentLogs: (expId: string) =>
    get<{ exp_id: string; log: string }>(`/api/experiments/${encodeURIComponent(expId)}/logs`),
  runExperiment: (project: string, command: string, name = '') =>
    post<{ exp_id: string; status: string; project: string }>('/api/experiments/run', { project, command, name }),
  getProjectGit: (project: string, kind: 'diff' | 'log' | 'status' = 'diff') =>
    get<{ kind: string; output: string }>(`/api/experiments/projects/${encodeURIComponent(project)}/git?kind=${kind}`),
};

// ---- Creation types (v10 / Phase B) ----

export interface CreationDocMeta {
  doc_id: string;
  title: string;
  status: string;
  n_sections: number;
  updated_at: string;
}

export interface SectionOutline {
  section_id: string;
  title: string;
  section_type: string;
  cites: string[];
  status: 'pending' | 'writing' | 'done';
}

export interface CreationDoc {
  doc_id: string;
  title: string;
  status: string;
  outline: SectionOutline[];
  sections: Record<string, { status: string; updated_at: string; word_count: number }>;
  sections_content: Record<string, string>;
  assembled_md: string;
  updated_at: string;
}

// ---- Workspace path settings (v10 / 可配置项目路径) ----

export type WorkspaceRoot = 'project' | 'experiments';

export interface Settings {
  /** None = 未显式设置（文献问答根=代码根，写作目录=web/workspace/docs）。 */
  project_path: string | null;
  /** 文献问答/通用工具实际根（未设置时=代码根）。 */
  project_root: string;
  /** 实验根（独立于文献问答，默认 web/workspace/experiments）。 */
  experiments_path: string;
  /** 写作文档保存目录（project_path 设置时 = {project_path}/writing）。 */
  writing_dir: string;
}

// ---- Experiment types (v10 / Phase D) ----

export interface Experiment {
  exp_id: string;
  project: string;
  name: string;
  command: string;
  status: 'pending' | 'running' | 'done' | 'failed' | 'unknown';
  exit_code: number | null;
  git_sha: string;
  metrics: Record<string, unknown>;
  created_at: string;
  finished_at: string;
  log_tail?: string;
}
