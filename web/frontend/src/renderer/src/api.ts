/**
 * api.ts — HTTP helpers + SSE stream consumer.
 *
 * ponytail: thin fetch wrappers, no axios. SSE via ReadableStream (supports POST).
 */

import type {
  PlanStep,
  BackgroundTask,
  AgentMode,
  CompletionReportData,
} from './state';

// Base URL for the FastAPI backend. In the Electron desktop client the main
// process spawns uvicorn on 127.0.0.1:8001 and App.tsx calls setBaseUrl() once
// it reports ready. In pure-browser dev (`npm run dev:renderer`) it stays '' and
// the Vite /api proxy forwards to localhost:8000.
let baseUrl = '';
let apiToken = '';
let originKnown = false;
let notifyOrigin: () => void = () => {};
const backendReady = new Promise<void>(resolve => { notifyOrigin = resolve; });

export function setBaseUrl(url: string) {
  baseUrl = url.replace(/\/$/, '');
  if (!originKnown) { originKnown = true; notifyOrigin(); }
}

export function setApiToken(token: string) {
  apiToken = token;
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
const authHeaders = (headers: Record<string, string> = {}) =>
  apiToken ? { ...headers, 'X-Demo-Token': apiToken } : headers;
const withAuthQuery = (path: string) => {
  if (!apiToken) return withBase(path);
  const separator = path.includes('?') ? '&' : '?';
  return `${withBase(path)}${separator}token=${encodeURIComponent(apiToken)}`;
};

// ---- helpers ----

async function get<T>(path: string): Promise<T> {
  const r = await fetch(withBase(path), { headers: authHeaders() });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(withBase(path), {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

async function put<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(withBase(path), {
    method: 'PUT',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

async function del<T>(path: string): Promise<T> {
  const r = await fetch(withBase(path), {
    method: 'DELETE',
    headers: authHeaders(),
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

// ---- SSE ----

export type SSEEvent =
  | { type: 'token'; content: string }
  | { type: 'replace_answer'; content: string }
  | { type: 'tool_start'; id: string; name: string; args?: Record<string, unknown>; parent_id?: string; kind?: 'subagent' | 'tool' }
  | { type: 'tool_end'; id: string; name: string; status: string; outcome?: string; code?: string; retryable?: boolean; result?: string; execution_time?: number | null }
  | { type: 'mode'; mode: 'react' | 'plan'; source: 'user' | 'auto' }
  | { type: 'plan'; steps: Array<{ id: string; description: string; target: string; depends_on?: string[]; status?: string }> }
  | { type: 'plan_step'; id: string; status: string; name?: string; description?: string; output?: string }
  | { type: 'plan_progress'; done: number; total: number }
  | { type: 'plan_verify'; status: string; done: number; total: number; outstanding: Array<{ id: string; description: string; reason: string }> }
  | { type: 'task_status'; task_status: CompletionReportData; final_confidence?: CompletionReportData['confidence'] }
  | { type: 'approval_required'; approval: AgentApproval; thread_id?: string }
  // 对话中心化（L2/L4）：写作/实验内联状态事件（带 thread_id 供归因）
  | { type: 'doc_section'; doc_id: string; section_id?: string; title?: string; status: string; word_count?: number; thread_id?: string }
  | { type: 'experiment'; exp_id: string; project?: string; name?: string; command?: string; status: string; exit_code?: number | null; thread_id?: string }
  | { type: 'done' }
  | { type: 'error'; message: string; code?: string };

export interface PlanVerdict {
  status: string;
  done: number;
  total: number;
  outstanding: Array<{ id: string; description: string; reason: string }>;
}

export interface AgentApproval {
  kind: 'tool_approval';
  tool: string;
  tool_version?: string;
  side_effect?: boolean;
  permissions?: string[];
  args?: {
    arg_keys?: string[];
    args_sha256?: string;
    args_bytes?: number;
    arg_preview?: Record<string, unknown> | unknown;
  };
  calls?: Array<{
    tool: string;
    tool_version?: string;
    args?: {
      arg_keys?: string[];
      args_sha256?: string;
      args_bytes?: number;
      arg_preview?: Record<string, unknown> | unknown;
    };
    idempotency_key?: string;
  }>;
  idempotency_key?: string;
  reason?: string;
}

export interface AgentResumeResponse {
  answer: string;
  intent?: string;
  thread_id: string;
  mode?: string;
  error?: string | null;
  requires_approval?: boolean;
  approval?: AgentApproval | null;
  task_status?: CompletionReportData;
  final_confidence?: CompletionReportData['confidence'];
}

export interface SSECallbacks {
  onToolStart?: (id: string, name: string, args?: Record<string, unknown>, parentId?: string, kind?: 'subagent' | 'tool') => void;
  onToolEnd?: (id: string, status: string, result?: string, executionTime?: number, name?: string) => void;
  onMode?: (mode: 'react' | 'plan', source: 'user' | 'auto') => void;
  onPlan?: (steps: PlanStep[]) => void;
  onPlanStep?: (id: string, status: PlanStep['status'], output?: string) => void;
  onPlanProgress?: (done: number, total: number) => void;
  onPlanVerify?: (verdict: PlanVerdict) => void;
  onTaskStatus?: (report: CompletionReportData) => void;
  onApprovalRequired?: (approval: AgentApproval) => void;
  onDocSection?: (docId: string, payload: { section_id?: string; title?: string; status: string; word_count?: number }) => void;
  onExperiment?: (expId: string, payload: { project?: string; name?: string; command?: string; status: string; exit_code?: number | null }) => void;
  onToken?: (content: string) => void;
  onReplaceAnswer?: (content: string) => void;
  onDone?: () => void;
  onError?: (message: string, code?: string) => void;
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
    headers: authHeaders({ 'Content-Type': 'application/json' }),
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
                  case 'task_status':
                    callbacks.onTaskStatus?.(event.task_status);
                    break;
                  case 'approval_required':
                    callbacks.onApprovalRequired?.(event.approval);
                    break;
                  case 'doc_section':
                    callbacks.onDocSection?.(event.doc_id, {
                      section_id: event.section_id,
                      title: event.title,
                      status: event.status,
                      word_count: event.word_count,
                    });
                    break;
                  case 'experiment':
                    callbacks.onExperiment?.(event.exp_id, {
                      project: event.project,
                      name: event.name,
                      command: event.command,
                      status: event.status,
                      exit_code: event.exit_code ?? null,
                    });
                    break;
                  case 'token':
                    callbacks.onToken?.(event.content);
                    break;
                  case 'replace_answer':
                    callbacks.onReplaceAnswer?.(event.content);
                    break;
                  case 'done':
                    callbacks.onDone?.();
                    break;
                  case 'error':
                    callbacks.onError?.(event.message, event.code);
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
    headers: authHeaders({ 'Content-Type': 'application/json' }),
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
  const es = new EventSource(withAuthQuery('/api/agent/tasks/stream'));
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
        threadId: null,
        percent: null,
        startedAt: null,
        finishedAt: null,
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

/**
 * openEvalRunStream — live SSE feed of one batch eval run (evaluation/live.py).
 *
 * Event shape: run_started → query_started / query_finished → aggregate (running
 * metrics: recall/mrr/task/tool/tokens/cost + done/total/eta) → judge_* →
 * run_finished | run_failed; `heartbeat` frames keep the socket warm and are
 * ignored by callers. EventSource reconnects natively and the server replays the
 * buffered history on (re)connect, so a drop heals without client code.
 * Returns a dispose function.
 */
export function openEvalRunStream(
  runId: string,
  onEvent: (ev: EvalLiveEvent) => void,
): () => void {
  const es = new EventSource(withAuthQuery(`/api/eval/runs/${encodeURIComponent(runId)}/stream`));
  es.onmessage = (ev: MessageEvent) => {
    try {
      const data = JSON.parse(ev.data as string) as EvalLiveEvent;
      if (data?.type && data.type !== 'heartbeat') onEvent(data);
    } catch { /* malformed frame → ignore */ }
  };
  es.onerror = () => { /* keep default auto-reconnect */ };
  return () => es.close();
}

/**
 * openEvalSingleStream — live staged view of one query's full turn.
 *
 * Frames: single_started → single_snapshot（每 ~0.8s，带当前 flow 快照）
 * → single_done | single_error。重连只会重读 trace 事件，不会重跑该 query。
 */
export function openEvalSingleStream(
  threadId: string,
  onEvent: (ev: EvalSingleStreamEvent) => void,
): () => void {
  const es = new EventSource(withAuthQuery(`/api/eval/single/${encodeURIComponent(threadId)}/stream`));
  es.onmessage = (ev: MessageEvent) => {
    try {
      const data = JSON.parse(ev.data as string) as EvalSingleStreamEvent;
      if (data?.type) onEvent(data);
    } catch { /* malformed frame → ignore */ }
  };
  es.onerror = () => { /* keep default auto-reconnect */ };
  return () => es.close();
}

// ---- API calls ----

export const api = {
  // Health / system
  getAgentHealth: () => get<AgentHealth>('/api/agent/health'),
  resumeAgent: (threadId: string, approved: boolean, mode: AgentMode = 'auto') =>
    post<AgentResumeResponse>('/api/agent/resume', {
      thread_id: threadId,
      approved,
      mode,
    }),
  getIndexStats: () => get<{ backend: string; collection_name: string; count: number }>('/api/index/stats'),

  // Workspace file explorer (root 决定基准根: project=文献问答+写作, experiments=实验)
  listWorkspace: (path = '.', root: WorkspaceRoot = 'project') =>
    get<{ outcome: string; data: { path: string; entries: Array<{ name: string; is_dir: boolean; size: number | null }> } }>(
      `/api/workspace/list?path=${encodeURIComponent(path)}&root=${root}`
    ),
  readWorkspaceFile: (path: string, root: WorkspaceRoot = 'project') =>
    get<{ outcome: string; data: { path: string; is_binary: boolean; content: string } }>(
      `/api/workspace/read?path=${encodeURIComponent(path)}&root=${root}`
    ),

  // ---- Path settings（可配置项目路径 / 实验根，v10 界面）----
  getSettings: () => get<Settings>('/api/settings'),
  updateSettings: (body: { project_path?: string | null; experiments_path?: string | null }) =>
    put<Settings>('/api/settings', body),
  /** 只读目录浏览（路径选择器用）：空 path → 盘符列表；否则该目录的子目录。 */
  browseDir: (path = '') =>
    get<{ outcome: string; data: { path: string; entries: Array<{ name: string; is_dir: boolean }> } }>(
      `/api/workspace/browse?path=${encodeURIComponent(path)}`
    ),

  // Upload (multipart — no JSON content-type)
  uploadPDF: async (file: File) => {
    const form = new FormData();
    form.append('file', file);
    const r = await fetch(withBase('/api/pdf/process'), {
      method: 'POST',
      headers: authHeaders(),
      body: form,
    });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json() as Promise<{ task_id: string; paper_name: string; status: string }>;
  },

  // Background task stack (agent-driven + upload/index tasks)
  listTasks: async () => {
    const raw = await get<Array<{
      task_id: string; kind?: string; paper_name?: string; status: string;
      progress?: string; error?: string | null;
      error_code?: string; retryable?: boolean;
      outcome?: string;
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
      errorCode: t.error_code ?? '',
      retryable: !!t.retryable,
      outcome: t.outcome ?? '',
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
    const r = await fetch(
      withBase(`/api/creation/docs/${encodeURIComponent(docId)}/export-docx`),
      { headers: authHeaders() },
    );
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
  /** 项目 manifest（project.json 委托契约）+ 近期实验 —— 实验面板/文档引用。 */
  getProjectManifest: (project: string) =>
    get<{ project: string; manifest: ProjectManifest; recent_experiments: Experiment[] }>(
      `/api/experiments/${encodeURIComponent(project)}/manifest`
    ),

  // ---- Config center（配置中心，v12 界面）----
  getConfigTools: () => get<ConfigTools>('/api/config/tools'),
  updateConfigTools: (body: UpdateConfigToolsBody) =>
    put<{ outcome: string; agents: Record<string, ConfigAgentGroup> }>('/api/config/tools', body),
  getExperimentConfig: () => get<ExperimentConfig>('/api/config/experiment'),
  updateExperimentConfig: (body: UpdateExperimentConfigBody) =>
    put<ExperimentConfig>('/api/config/experiment', body),
  getMcpConfig: () => get<McpInfo>('/api/config/mcp'),
  updateMcpConfig: (servers: McpServerInfo[]) =>
    put<{ outcome: string; servers: McpServerInfo[] }>('/api/config/mcp', { servers }),
  testMcpServer: (name: string) =>
    post<{ name: string; outcome: string; tool_count: number; error: string | null }>(
      '/api/config/mcp/test', { name }),
  getSkillsConfig: () => get<SkillsConfigInfo>('/api/config/skills'),
  updateSkillsConfig: (disabled: string[]) =>
    put<{ outcome: string; skills: SkillInfo[] }>('/api/config/skills', { disabled }),
  getPromptsConfig: () => get<PromptsConfigInfo>('/api/config/prompts'),
  updatePromptsConfig: (canary: Record<string, { version: string; percent: number }>) =>
    put<{ outcome: string } & PromptsConfigInfo>('/api/config/prompts', { canary }),
  getMemory: () => get<MemoryPayload>('/api/memory'),
  createMemory: (body: {
    content: string;
    type?: string;
    confidence?: number;
    source_ref?: string;
    expires_at?: string | null;
    consent?: boolean;
  }) => post<{ outcome: string; record: MemoryRecordData }>('/api/memory', body),
  deleteMemory: (memoryId: string) =>
    del<{ outcome: string; deleted: string }>(`/api/memory/${encodeURIComponent(memoryId)}`),
  approveMemory: (memoryId: string) =>
    post<{ outcome: string; record: MemoryRecordData }>(
      `/api/memory/${encodeURIComponent(memoryId)}/approve`, {}),
  updateMemoryPolicy: (body: { disabled_ids?: string[]; disabled_types?: string[] }) =>
    put<{ outcome: string; policy: MemoryPayload['policy'] }>('/api/memory/policy', body),

  // ---- Agent 评测（evaluation 包，/api/eval）----
  listEvalRuns: (limit = 20) => get<EvalRunsResp>(`/api/eval/runs?limit=${limit}`),
  getEvalRun: (runId: string) => get<EvalRunReport>(`/api/eval/runs/${encodeURIComponent(runId)}`),
  getEvalBadcases: (runId: string) =>
    get<{ run_id: string; badcases: EvalBadcase[]; count: number }>(
      `/api/eval/runs/${encodeURIComponent(runId)}/badcases`),
  getEvalTrace: (traceId: string) =>
    get<{ trace_id: string; event_count: number; events: EvalTraceEvent[] }>(
      `/api/eval/traces/${encodeURIComponent(traceId)}`),
  getEvalThread: (threadId: string) =>
    get<{ thread_id: string; turns: EvalTurn[]; events: EvalTraceEvent[] }>(
      `/api/eval/thread/${encodeURIComponent(threadId)}`),
  startEvalRun: (body: { manifest?: string; limit?: number; judge?: boolean }) =>
    post<{ outcome: string; run_id: string; status: string }>('/api/eval/runs', body),
  submitEvalFeedback: (body: {
    trace_id: string;
    key: string;
    score?: number | boolean | null;
    comment?: string;
    run_id?: string;
    project?: string;
  }) => post<{ outcome: string; run_id: string; key: string }>('/api/eval/feedback', body),
  addEvalRegressionCase: (traceId: string) =>
    post<{ outcome: string; written: string; qa_id: string; chunk_ids: number }>(
      '/api/eval/manifest/from-trace', { trace_id: traceId },
    ),
  /** 单样例透明评测：一条 query 的完整链路分阶段可视化。 */
  runSingleEval: (query: string, opts?: { threadId?: string; mode?: string }) =>
    post<EvalSingleFlow>('/api/eval/single', {
      query, ...(opts?.threadId ? { thread_id: opts.threadId } : {}),
      ...(opts?.mode ? { mode: opts.mode } : {}),
    }),
  /** 单样例评测（非阻塞）：起跑后接 GET /api/eval/single/{thread_id}/stream 看实时阶段。 */
  startEvalSingle: (query: string, opts?: {
    threadId?: string;
    mode?: string;
    promptOverrides?: Record<string, string>;
  }) =>
    post<{ outcome: string; thread_id: string; trace_id: string; status: string; reused?: boolean }>(
      '/api/eval/single/start', {
        query, ...(opts?.threadId ? { thread_id: opts.threadId } : {}),
        ...(opts?.mode ? { mode: opts.mode } : {}),
        ...(opts?.promptOverrides && Object.keys(opts.promptOverrides).length
          ? { prompt_overrides: opts.promptOverrides }
          : {}),
      }),
};

// ---- Eval types (Agent 评测体系 / evaluation 包) ----

/** 跑批实时进度事件（evaluation/live.py 的事件契约）。 */
export interface EvalLiveEvent {
  type: string;                 // run_started / query_started / query_finished /
                                // aggregate / judge_* / run_finished / run_failed
  run_id?: string;
  dataset_id?: string;
  status?: string;
  done?: number;
  total?: number;
  index?: number;
  qid?: string;
  query?: string;
  category?: string;
  success?: boolean;
  'recall@5'?: number | null;
  'ndcg@10'?: number | null;
  mrr?: number | null;
  tool_errors?: number;
  duration_s?: number | null;
  tokens?: number | null;
  cost_usd?: number | null;
  trace_id?: string;
  run_error?: string;
  elapsed_s?: number;
  eta_s?: number | null;
  overall?: Record<string, unknown>;
  last?: EvalLiveQueryRow;
  recent?: EvalLiveQueryRow[];
  gate?: string | null;
  badcase_count?: number;
  error?: string;
  source?: string;
}

/** 单条 QA 的实时指标行（aggregate.recent / query_finished）。 */
export interface EvalLiveQueryRow {
  index: number;
  total: number;
  qid: string;
  query: string;
  category?: string;
  success?: boolean;
  'recall@5'?: number | null;
  'ndcg@10'?: number | null;
  mrr?: number | null;
  tool_errors?: number;
  duration_s?: number | null;
  tokens?: number | null;
  cost_usd?: number | null;
  trace_id?: string;
  run_error?: string;
}

/** 单样例流式事件（flow 为当前已完成阶段的分阶段视图）。 */
export interface EvalSingleStreamEvent {
  type: string;                 // single_started / single_snapshot / single_done / single_error
  thread_id?: string;
  trace_id?: string;
  status?: string;
  elapsed_s?: number;
  flow?: EvalSingleFlow;
  error?: string;
}

export interface EvalRunsResp {
  runs: Array<{
    run_id: string; dataset_id: string; status: string; query_count: number;
    started_at: string; overall: Record<string, unknown>;
    finished_at?: string | null;
    /** 完整用时（running 时是「已用」秒数，由 started_at/finished_at 算出）。 */
    duration_s?: number | null;
    badcase_count?: number;
  }>;
  count: number;
}

export interface EvalRunReport {
  run_id: string;
  dataset_id: string;
  started_at: string;
  finished_at: string;
  /** 完整用时（跑批开始 → 结束的墙钟秒数）。 */
  duration_s?: number | null;
  status: string;
  query_count: number;
  metadata: Record<string, unknown>;
  overall: Record<string, unknown>;
  dimension: Array<Record<string, unknown>>;
  badcases: EvalBadcase[];
  judge: Record<string, unknown>;
  tool_metrics: Record<string, unknown>;
  task_metrics: Record<string, unknown>;
  baseline_delta: { gate: string; deltas: Record<string, unknown> };
  cost_estimate: Record<string, unknown>;
  notes: string[];
}

export interface EvalBadcase {
  query_id?: string;
  query: string;
  category: string;
  mrr?: number;
  recall?: number;
  tool_errors?: number;
  /** 任务级失败原因（task_error / run_error）。 */
  task_error?: string;
  status?: string;
  success?: boolean;
  duration_s?: number | null;
  answer?: string;
  trace_id?: string;
  run_error?: string;
}

export interface EvalTraceEvent {
  seq: number;
  ts: string;
  event_type: string;
  node: string;
  duration_ms?: number | null;
  model?: string | null;
  intent?: string | null;
  error?: string | null;
  tool?: string | null;
  outcome?: string | null;
  parent_id?: string | null;
  payload: Record<string, unknown>;
}

export interface EvalTurn {
  trace_id: string;
  turn_seq: number;
  started_at: string;
  event_count: number;
}

// ---- 单样例透明评测（evaluation/flow.build_flow） ----

export interface EvalFlowStage {
  key: string;
  label: string;
  duration_ms: number;
  tokens: { prompt: number; completion: number; total: number; estimated: number };
  items: EvalFlowItem[];
}

export interface EvalFlowItem {
  type: 'query' | 'intent' | 'llm' | 'tool' | 'retrieval' | 'plan' | 'plan_step'
    | 'plan_verify' | 'answer' | 'turn_end';
  query?: string;
  intent?: string;
  confidence?: number;
  entities?: string[];
  focus_papers?: string[];
  needs_planning?: boolean | null;
  domain?: string;
  node?: string;
  model?: string;
  mode?: string;
  tokens?: Record<string, unknown>;
  duration_ms?: number | null;
  error?: string | null;
  tool?: string;
  args?: Record<string, unknown> | string;
  result?: unknown;
  outcome?: string;
  operation?: Record<string, unknown>;
  parsed?: { is_envelope?: boolean; outcome?: string; error_type?: string; error?: string };
  chunk_ids?: string[];
  snapshot?: string;
  steps?: unknown[];
  status?: string;
  detail?: string;
  step_id?: string;
  answer?: string;
  verification?: Record<string, unknown> | null;
}

export interface EvalSingleFlow {
  query: string;
  thread_id: string;
  stages: EvalFlowStage[];
  metrics: {
    total_duration_ms: number;
    llm_calls: number;
    tokens: { prompt: number; completion: number; total: number };
    estimated_calls: number;
    tool_calls: number;
    tool_failures: number;
    task_success: boolean;
    task_status?: string;
    intent: string;
    mode: string;
    turn_status: string;
    cost_usd?: number;
    trace_id?: string;
    event_count?: number;
    elapsed_s?: number | null;
    config_revision?: string;
    config_hash?: string;
    prompt_versions?: Record<string, string>;
  };
  context: Record<string, unknown>;
  context_snapshot?: string;
  status?: string;
  error?: string;
  events?: EvalTraceEvent[];
}

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

// ---- Project manifest（project.json 委托契约, 对话中心化 L4）----

export interface ProjectManifest {
  project: string;
  paper: string;
  description: string;
  entry: { run: string; data: string; config: string };
  key_files: string[];
  metrics_schema: Record<string, unknown>;
  baseline: Record<string, unknown>;
  status: string;
  last_run: string;
  last_delegate: string;
  changed_files: string[];
  changelog: Array<{ kind: string; summary: string; at: string }>;
  last_commit_sha: string;
}

// ---- Config center types (v12 / 配置中心) ----

export interface AgentHealth {
  status: string;
  model: string;
  tools: number;
}

export interface MemoryRecordData {
  memory_id: string;
  type: string;
  content: string;
  source_ref: string;
  confidence: number;
  created_at: string;
  expires_at?: string | null;
  consent: boolean;
  revision: string;
  semantic_key?: string;
  status?: string;
  supersedes?: string[];
  conflicts_with?: string[];
  enabled: boolean;
}

export interface MemoryPayload {
  path: string;
  records: MemoryRecordData[];
  policy: {
    min_confidence: number;
    max_age_days: number | null;
    require_consent: boolean;
    disabled_ids: string[];
    disabled_types: string[];
  };
}

export interface ConfigToolInfo {
  name: string;
  description: string;
  source: string;
  loaded: boolean;
  enabled: boolean;
}

export interface ConfigAgentGroup {
  label: string;
  max_steps: number;
  tools: ConfigToolInfo[];
}

/** 按 agent 分组的工具清单：parent + arxiv/ingest/creator/coder。 */
export interface ConfigTools {
  agents: Record<string, ConfigAgentGroup>;
}

export interface UpdateConfigToolsBody {
  disabled: Record<string, string[]>;
  max_steps?: Record<string, number>;
}

export interface ExperimentConfig {
  paths: {
    project_path: string | null;
    project_root: string;
    experiments_path: string;
    writing_dir: string;
  };
  delegate_prefer: 'mcp' | 'cli';
  delegate_timeout: number;
  auto_git_commit: boolean;
  manifest_auto_update: boolean;
}

export interface UpdateExperimentConfigBody {
  delegate_prefer?: 'mcp' | 'cli';
  delegate_timeout?: number;
  auto_git_commit?: boolean;
  manifest_auto_update?: boolean;
}

/** MCP server 行（除展示字段外透传 .mcp.json 原始字段）。 */
export interface McpServerInfo {
  name: string;
  transport?: string;
  command?: string;
  args?: string[];
  url?: string;
  headers?: Record<string, string>;
  env?: Record<string, string>;
  disabled?: boolean;
  status: 'ok' | 'unknown';
  tools: number;
  [key: string]: unknown;
}

export interface McpInfo {
  exists: boolean;
  path: string;
  servers: McpServerInfo[];
}

export interface SkillInfo {
  name: string;
  description: string;
  path: string;
  resources: string[];
  enabled: boolean;
}

export interface SkillsConfigInfo {
  skills: SkillInfo[];
}

export interface PromptVersionInfo {
  version: string;
  status: string;
  checksum: string;
  evaluation_suite: string;
}

export interface PromptConfigInfo {
  id: string;
  active: {
    version: string;
    binding: string;
    evaluation_suite: string;
  } | null;
  versions: PromptVersionInfo[];
  canary?: { version: string; percent: number } | null;
}

export interface PromptsConfigInfo {
  prompts: PromptConfigInfo[];
}
