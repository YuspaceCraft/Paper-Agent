/**
 * state.ts — AppState types, reducer, and thread persistence helpers.
 *
 * ponytail: single Context + useReducer, no Redux. localStorage for thread persistence.
 */

// ---- types ----

/** 客户端执行模式切换：auto = 后端启发式 | react | plan（显式指定）。 */
export type AgentMode = 'auto' | 'react' | 'plan';

/** plan_step / plan_verify 检查出的一步未完成项。 */
export interface PlanOutstanding {
  id: string;
  description: string;
  reason: string;
}

/** verify_node 的报告式验证结论。 */
export interface PlanVerdictData {
  status: 'satisfied' | 'partial' | 'failed' | 'no_evidence' | string;
  done: number;
  total: number;
  outstanding: PlanOutstanding[];
}

/** One tool call the agent made this turn, rendered as a collapsible step. */
export interface ToolStep {
  id: string;
  name: string;
  args?: Record<string, unknown>;
  result?: string;
  status: 'running' | 'success' | 'error';
  executionTime?: number;
  /** Set on leaf tools called inside a subagent — the parent step's id. */
  parentId?: string;
  /** Nested leaf tools of a subagent step. */
  children?: ToolStep[];
  /** 'subagent' = a parent step (arxiv/ingest); undefined = leaf tool. */
  kind?: 'subagent' | 'tool';
}

/** One step of a plan (plan-and-execute mode), emitted by the plan node.
    status 由 plan_step / tool 事件驱动 TODO 列表的实时勾选；output 供
    skipped（守卫）步骤展示跳过原因。 */
export interface PlanStep {
  id: string;
  description: string;
  target: string;
  depends_on?: string[];
  status?: 'pending' | 'running' | 'done' | 'failed' | 'skipped';
  output?: string;
}

export interface Message {
  id: string;
  role: 'user' | 'system';
  content: string;
  steps?: ToolStep[];
  plan?: PlanStep[];
  /** 计划完成度 {done, total}，后端 plan_progress 事件。 */
  planProgress?: { done: number; total: number } | null;
  /** verify_node 的报告式验证结论（plan 模式收尾）。 */
  verify?: PlanVerdictData | null;
  /** 本回合实际执行模式（mode SSE 事件）。 */
  mode?: 'react' | 'plan';
  status: 'complete' | 'streaming' | 'aborted';
  timestamp: string;
}

export interface ThreadMeta {
  title: string;
  messageCount: number;
  createdAt: string;
  updatedAt: string;
  /** 该对话的客户端模式偏好（按对话记忆，localStorage 持久化）。 */
  mode: AgentMode;
}

/**
 * BackgroundTask — one entry in the background task stack
 * (POST /api/agent/ingest / /api/pdf/process / /api/index/run → GET
 * /api/agent/tasks). Rendered in the TaskCenter panel above the chat.
 *
 * 展示分 3 态：运行中(pending|running) / 完成(done) / 失败(failed)；内部保留
 * 4 态以便区分「排队中」。`threadId` 为所属对话（前端归因，null=系统级/未归属）、
 * `startedAt/finishedAt` 由前端在状态迁移时本地补记（后端不提供）。
 */
export interface BackgroundTask {
  taskId: string;
  kind: string;
  paperName: string;
  /** 所属对话 threadId；null = 系统级任务（未归属/删除对话后解绑） */
  threadId: string | null;
  status: 'pending' | 'running' | 'done' | 'failed';
  progress: string;
  /** 0-100 可选；后端暂不下发 → 恒 null，UI 用 indeterminate 动画条 */
  percent: number | null;
  error: string | null;
  result: Record<string, unknown> | null;
  notify: boolean;
  /** ingest 任务阶段: 'parse' 解析中 | 'index' 向量化入库中 | '' 其他 */
  stage: string;
  /** server timestamp: epoch seconds (float) or ISO string */
  createdAt: string;
  /** server timestamp（同 createdAt 格式），前端每次 TASK_UPDATE 刷新 */
  updatedAt: string;
  /** ISO；pending→running 的时刻（前端补记） */
  startedAt: string | null;
  /** ISO；进入 done/failed 的时刻（前端补记） */
  finishedAt: string | null;
}

export interface AppState {
  threads: Record<string, ThreadMeta>;
  threadOrder: string[];
  activeThreadId: string | null;
  messages: Message[];
  isStreaming: boolean;

  leftPanelOpen: boolean;
  rightPanelOpen: boolean;
  rightPanelWidth: number;

  bgTasks: BackgroundTask[];

  indexStats: { backend: string; collection_name: string; count: number } | null;
  agentHealth: { status: string; model: string; tools: number } | null;
}

export type Action =
  | { type: 'INIT_STATE'; state: Partial<AppState> }
  | { type: 'CREATE_THREAD'; threadId: string }
  | { type: 'SWITCH_THREAD'; threadId: string }
  | { type: 'DELETE_THREAD'; threadId: string }
  | { type: 'UPDATE_THREAD_TITLE'; threadId: string; title: string }
  | { type: 'ADD_MESSAGE'; message: Message }
  | { type: 'APPEND_TOKEN'; messageId: string; token: string }
  | { type: 'FINISH_MESSAGE'; messageId: string }
  | { type: 'ADD_TOOL_STEP'; messageId: string; step: ToolStep }
  | { type: 'UPDATE_TOOL_STEP'; messageId: string; toolCallId: string; patch: Partial<ToolStep> }
  | { type: 'SET_PLAN'; messageId: string; plan: PlanStep[] }
  | { type: 'UPDATE_PLAN_STEP'; messageId: string; stepId: string; patch: Partial<PlanStep> }
  | { type: 'SET_PLAN_PROGRESS'; messageId: string; done: number; total: number }
  | { type: 'SET_PLAN_VERIFY'; messageId: string; verdict: PlanVerdictData }
  | { type: 'SET_MODE'; messageId: string; mode: 'react' | 'plan' }
  | { type: 'SET_THREAD_MODE'; threadId: string; mode: AgentMode }
  | { type: 'ABORT_MESSAGE'; messageId: string }
  | { type: 'SET_STREAMING'; isStreaming: boolean }
  | { type: 'SET_RIGHT_PANEL_WIDTH'; width: number }
  | { type: 'TOGGLE_LEFT_PANEL' }
  | { type: 'TOGGLE_RIGHT_PANEL' }
  | { type: 'SET_INDEX_STATS'; stats: AppState['indexStats'] }
  | { type: 'SET_AGENT_HEALTH'; health: AppState['agentHealth'] }
  | { type: 'TASK_SYNC'; tasks: BackgroundTask[] }
  | { type: 'TASK_UPDATE'; task: BackgroundTask }
  | { type: 'TASK_ATTRIBUTE'; taskId: string; threadId: string | null }
  | { type: 'TASK_DISMISS'; taskId: string };

// ---- initial state ----

export const initialAppState: AppState = {
  threads: {},
  threadOrder: [],
  activeThreadId: null,
  messages: [],
  isStreaming: false,
  leftPanelOpen: true,
  rightPanelOpen: true,
  rightPanelWidth: 420,
  bgTasks: [],
  indexStats: null,
  agentHealth: null,
};

// ---- background task helpers ----

/** 服务端任务时间戳（epoch 秒浮点或 ISO 字符串）→ 毫秒；解析失败返回 NaN。 */
export function parseTaskTs(ts: string): number {
  const n = Number(ts);
  if (Number.isFinite(n)) return n > 1e12 ? n : n * 1000; // 秒 → 毫秒
  const d = new Date(ts).getTime();
  return Number.isFinite(d) ? d : NaN;
}

/**
 * 合并一条后台任务的新状态到现有记录上。
 * - 保留前端归因（threadId）与 percent（后端不下发）。
 * - 状态迁移时补记 startedAt（pending→running）/ finishedAt（→done/failed），
 *   反向迁移防御性地清掉 finishedAt。
 */
function mergeTask(existing: BackgroundTask | undefined, incoming: BackgroundTask): BackgroundTask {
  const now = new Date().toISOString();
  const status = incoming.status;
  const terminal = status === 'done' || status === 'failed';
  const running = status === 'running' || status === 'pending';

  let startedAt = existing?.startedAt ?? null;
  let finishedAt = existing?.finishedAt ?? null;
  if (running) {
    // 反向更新（服务端不应发生）防御：不 keep finishedAt
    finishedAt = null;
  }
  if (status === 'running' && !existing?.startedAt) startedAt = now;
  if (terminal) {
    startedAt = startedAt ?? existing?.createdAt ?? now;
    if (!existing?.finishedAt) finishedAt = now;
  }

  return {
    ...incoming,
    threadId: incoming.threadId ?? existing?.threadId ?? null,
    percent: incoming.percent ?? existing?.percent ?? null,
    startedAt,
    finishedAt,
    updatedAt: now,
  };
}

// ---- step tree helpers ----
// `steps` is a tree: subagent steps have `children`, leaf tools have `parentId`.
// v15: 树维护幂等化 —— ①同一 id 去重(后端 executor 重试/重复 emitter 会重发
// 相同 id 的 tool_start); ②乱序重挂 —— 子步骤在父卡未到时先挂顶层,父卡插入
// 时统一移回其 children,摆脱对 SSE 到达顺序的隐式依赖。

/** Recursively find whether a step with `id` already exists (dedup key). */
function findStep(steps: ToolStep[], id: string): boolean {
  for (const s of steps) {
    if (s.id === id) return true;
    if (s.children && findStep(s.children, id)) return true;
  }
  return false;
}

/** Attach `step` under the node with `parentId`; `ok=false` if parent missing. */
function attachUnder(steps: ToolStep[], parentId: string, step: ToolStep):
  { steps: ToolStep[]; ok: boolean } {
  let ok = false;
  const next = steps.map(s => {
    if (ok) return s;
    if (s.id === parentId) {
      ok = true;
      return { ...s, children: [...(s.children ?? []), step] };
    }
    if (s.children) {
      const kid = attachUnder(s.children, parentId, step);
      if (kid.ok) {
        ok = true;
        return { ...s, children: kid.steps };
      }
    }
    return s;
  });
  return { steps: next, ok };
}

function insertStep(steps: ToolStep[], step: ToolStep): ToolStep[] {
  if (findStep(steps, step.id)) return steps;  // 幂等去重

  if (step.parentId) {
    const attached = attachUnder(steps, step.parentId, step);
    if (attached.ok) return attached.steps;
    // 父卡还没到 → 暂挂顶层,等父卡插入时被吸收重挂
    return [...steps, step];
  }

  // 顶层新父节点:吸收顶层中 parentId 指向它的待定子步骤(乱序重挂)
  const pending = steps.filter(s => s.parentId === step.id);
  if (pending.length) {
    const rest = steps.filter(s => s.parentId !== step.id);
    return [...rest, { ...step, children: [...(step.children ?? []), ...pending] }];
  }
  return [...steps, step];
}

function patchStep(steps: ToolStep[], id: string, patch: Partial<ToolStep>): ToolStep[] {
  return steps.map(s => {
    if (s.id === id) return { ...s, ...patch };
    if (s.children) return { ...s, children: patchStep(s.children, id, patch) };
    return s;
  });
}

function patchPlanStep(plan: PlanStep[] | undefined, stepId: string, patch: Partial<PlanStep>): PlanStep[] {
  return (plan ?? []).map(s => (s.id === stepId ? { ...s, ...patch } : s));
}

// ---- reducer ----

export function appReducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'INIT_STATE':
      return { ...state, ...action.state };

    case 'CREATE_THREAD': {
      const now = new Date().toISOString();
      const meta: ThreadMeta = { title: '', messageCount: 0, createdAt: now, updatedAt: now, mode: 'auto' };
      return {
        ...state,
        threads: { ...state.threads, [action.threadId]: meta },
        threadOrder: [action.threadId, ...state.threadOrder],
        activeThreadId: action.threadId,
        messages: [],
      };
    }

    case 'SWITCH_THREAD':
      return { ...state, activeThreadId: action.threadId };

    case 'DELETE_THREAD': {
      const deleted = action.threadId;
      const { [deleted]: _, ...rest } = state.threads;
      const newOrder = state.threadOrder.filter(id => id !== deleted);
      const fall = deleted === state.activeThreadId;
      return {
        ...state,
        threads: rest,
        threadOrder: newOrder,
        activeThreadId: fall ? (newOrder[0] ?? null) : state.activeThreadId,
        messages: fall ? [] : state.messages,
        // 任务跟随对话：该对话的已结束任务随之清除；运行中任务解绑为系统级继续展示
        bgTasks: state.bgTasks
          .filter(t => !(t.threadId === deleted && (t.status === 'done' || t.status === 'failed')))
          .map(t => t.threadId === deleted ? { ...t, threadId: null } : t),
      };
    }

    case 'UPDATE_THREAD_TITLE':
      if (!state.threads[action.threadId]) return state;
      return {
        ...state,
        threads: {
          ...state.threads,
          [action.threadId]: {
            ...state.threads[action.threadId],
            title: action.title,
            updatedAt: new Date().toISOString(),
          },
        },
      };

    case 'ADD_MESSAGE': {
      const newMsgs = [...state.messages, action.message];
      const tid = state.activeThreadId;
      if (tid && state.threads[tid]) {
        return {
          ...state,
          messages: newMsgs,
          threads: {
            ...state.threads,
            [tid]: {
              ...state.threads[tid],
              messageCount: newMsgs.filter(m => m.role === 'user').length,
              updatedAt: new Date().toISOString(),
              title: state.threads[tid].title || (action.message.role === 'user'
                ? action.message.content.slice(0, 30)
                : state.threads[tid].title),
            },
          },
        };
      }
      return { ...state, messages: newMsgs };
    }

    case 'APPEND_TOKEN':
      return {
        ...state,
        messages: state.messages.map(m =>
          m.id === action.messageId
            ? { ...m, content: m.content + action.token }
            : m
        ),
      };

    case 'FINISH_MESSAGE':
      return {
        ...state,
        isStreaming: false,
        messages: state.messages.map(m =>
          m.id === action.messageId
            ? { ...m, status: 'complete' as const }
            : m
        ),
      };

    case 'ADD_TOOL_STEP':
      return {
        ...state,
        messages: state.messages.map(m =>
          m.id === action.messageId
            ? { ...m, steps: insertStep(m.steps ?? [], action.step) }
            : m
        ),
      };

    case 'UPDATE_TOOL_STEP':
      return {
        ...state,
        messages: state.messages.map(m =>
          m.id === action.messageId
            ? { ...m, steps: patchStep(m.steps ?? [], action.toolCallId, action.patch) }
            : m
        ),
      };

    case 'SET_PLAN':
      return {
        ...state,
        messages: state.messages.map(m =>
          m.id === action.messageId ? { ...m, plan: action.plan } : m
        ),
      };

    case 'UPDATE_PLAN_STEP':
      return {
        ...state,
        messages: state.messages.map(m =>
          m.id === action.messageId
            ? { ...m, plan: patchPlanStep(m.plan, action.stepId, action.patch) }
            : m
        ),
      };

    case 'SET_PLAN_PROGRESS':
      return {
        ...state,
        messages: state.messages.map(m =>
          m.id === action.messageId
            ? { ...m, planProgress: { done: action.done, total: action.total } }
            : m
        ),
      };

    case 'SET_PLAN_VERIFY':
      return {
        ...state,
        messages: state.messages.map(m =>
          m.id === action.messageId ? { ...m, verify: action.verdict } : m
        ),
      };

    case 'SET_MODE':
      return {
        ...state,
        messages: state.messages.map(m =>
          m.id === action.messageId ? { ...m, mode: action.mode } : m
        ),
      };

    case 'SET_THREAD_MODE': {
      const th = state.threads[action.threadId];
      if (!th) return state;
      return {
        ...state,
        threads: {
          ...state.threads,
          [action.threadId]: { ...th, mode: action.mode },
        },
      };
    }

    case 'ABORT_MESSAGE':
      return {
        ...state,
        isStreaming: false,
        messages: state.messages.map(m =>
          m.id === action.messageId
            ? { ...m, status: 'aborted' as const }
            : m
        ),
      };

    case 'SET_STREAMING':
      return { ...state, isStreaming: action.isStreaming };

    case 'SET_RIGHT_PANEL_WIDTH':
      return { ...state, rightPanelWidth: action.width };

    case 'TOGGLE_LEFT_PANEL':
      return { ...state, leftPanelOpen: !state.leftPanelOpen };
    case 'TOGGLE_RIGHT_PANEL':
      return { ...state, rightPanelOpen: !state.rightPanelOpen };

    case 'SET_INDEX_STATS':
      return { ...state, indexStats: action.stats };
    case 'SET_AGENT_HEALTH':
      return { ...state, agentHealth: action.health };

    case 'TASK_SYNC':
      // Fallback-poll result of /api/agent/tasks — merge rather than replace:
      // a poll response can be stale or racing (e.g. fired against the pre-ready
      // origin) and must NEVER *remove* a task the panel already shows — only the
      // dismiss button removes. Existing tasks update in place, new ones append.
      {
        const byId = new Map(action.tasks.map(t => [t.taskId, t]));
        const merged = state.bgTasks.map(t => {
          const n = byId.get(t.taskId);
          return n ? mergeTask(t, n) : t;
        });
        const known = new Set(state.bgTasks.map(t => t.taskId));
        for (const t of action.tasks) {
          if (!known.has(t.taskId)) merged.push(mergeTask(undefined, t));
        }
        return { ...state, bgTasks: merged };
      }

    case 'TASK_UPDATE': {
      // SSE push: upsert a single task (new → front, existing → in place)，
      // 合并保留归因/时间戳并在状态迁移时补记。
      const { task } = action;
      const idx = state.bgTasks.findIndex(t => t.taskId === task.taskId);
      const merged = mergeTask(idx >= 0 ? state.bgTasks[idx] : undefined, task);
      return {
        ...state,
        bgTasks: idx >= 0
          ? state.bgTasks.map(t => t.taskId === task.taskId ? merged : t)
          : [merged, ...state.bgTasks],
      };
    }

    case 'TASK_ATTRIBUTE': {
      // 前端归因：补记所属对话（任务可能尚未到达 → 未命中则等 TASK_SYNC/UPDATE 落地）。
      const { taskId, threadId } = action;
      return {
        ...state,
        bgTasks: state.bgTasks.map(t =>
          t.taskId === taskId ? { ...t, threadId } : t
        ),
      };
    }

    case 'TASK_DISMISS':
      return { ...state, bgTasks: state.bgTasks.filter(t => t.taskId !== action.taskId) };

    default:
      return state;
  }
}

// ---- localStorage persistence ----

const THREADS_KEY = 'demo_threads';
const MSG_PREFIX = 'demo_msgs_';

export function loadThreads(): { threads: Record<string, ThreadMeta>; threadOrder: string[] } {
  try {
    const raw = localStorage.getItem(THREADS_KEY);
    if (raw) {
      const { threads, threadOrder } = JSON.parse(raw);
      // 旧版本持久化数据缺 mode → 兜底 'auto'，避免 undefined 破坏切换按键
      for (const t of Object.values(threads ?? {})) {
        (t as ThreadMeta).mode = ((t as ThreadMeta).mode ?? 'auto') as AgentMode;
      }
      return { threads: threads ?? {}, threadOrder: threadOrder ?? [] };
    }
  } catch { /* ignore */ }
  return { threads: {}, threadOrder: [] };
}

export function saveThreads(threads: Record<string, ThreadMeta>, order: string[]) {
  try {
    localStorage.setItem(THREADS_KEY, JSON.stringify({ threads, threadOrder: order }));
  } catch { /* ignore */ }
}

export function loadMessages(threadId: string): Message[] {
  try {
    const raw = localStorage.getItem(MSG_PREFIX + threadId);
    if (raw) return JSON.parse(raw);
  } catch { /* ignore */ }
  return [];
}

export function saveMessages(threadId: string, messages: Message[]) {
  try {
    localStorage.setItem(MSG_PREFIX + threadId, JSON.stringify(messages));
  } catch (e) {
    if (e instanceof DOMException && e.name === 'QuotaExceededError') {
      console.warn('localStorage full, cannot save messages for', threadId);
    }
  }
}

// ---- background task persistence ----
// 目的：跨刷新/重启保住「归因 + 历史」；上限截断防 localStorage 膨胀。
// 注意：服务端 Redis 任务 1h TTL，这里保留的是前端自己的展示副本，二者独立。

const BG_KEY = 'demo_bg_tasks';
const BG_CAP = 200;

export function loadBgTasks(): BackgroundTask[] {
  try {
    const raw = localStorage.getItem(BG_KEY);
    if (raw) {
      const list = JSON.parse(raw);
      if (Array.isArray(list)) return normalizeBgTasks(list);
    }
  } catch { /* ignore */ }
  return [];
}

export function saveBgTasks(tasks: BackgroundTask[]) {
  try {
    if (tasks.length === 0) {
      localStorage.removeItem(BG_KEY);
      return;
    }
    // 保序（新→旧）截断；历史旧持久化的字段缺失按默认值补，避免旧版本数据崩渲染
    localStorage.setItem(BG_KEY, JSON.stringify(tasks.slice(0, BG_CAP)));
  } catch (e) {
    if (e instanceof DOMException && e.name === 'QuotaExceededError') {
      console.warn('localStorage full, cannot save background tasks');
    }
  }
}

/** 旧版本持久化数据字段兜底，保证类型安全。 */
function normalizeBgTasks(list: unknown[]): BackgroundTask[] {
  return list.filter((x): x is BackgroundTask => !!x && typeof (x as BackgroundTask).taskId === 'string')
    .map(t => ({
      threadId: t.threadId ?? null,
      percent: t.percent ?? null,
      startedAt: t.startedAt ?? null,
      finishedAt: t.finishedAt ?? null,
      updatedAt: t.updatedAt ?? t.createdAt ?? new Date(0).toISOString(),
      kind: t.kind ?? '',
      paperName: t.paperName ?? '',
      progress: t.progress ?? '',
      stage: t.stage ?? '',
      error: t.error ?? null,
      result: t.result ?? null,
      notify: !!t.notify,
      taskId: t.taskId,
      status: t.status ?? 'pending',
      createdAt: t.createdAt ?? '',
    }));
}
