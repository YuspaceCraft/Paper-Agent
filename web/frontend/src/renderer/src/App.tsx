/**
 * App.tsx — root component: three-column layout + context provider.
 *
 * ponytail: useReducer + Context, no external state lib.
 */

import { type FC, createContext, useContext, useReducer, useEffect, useCallback, useRef, useState } from 'react';
import {
  type AppState, type Action, type Message, type BackgroundTask,
  initialAppState, appReducer,
  loadThreads, saveThreads, loadMessages, saveMessages,
  loadBgTasks, saveBgTasks, parseTaskTs,
} from './state';
import { api, setBaseUrl, markProxyReady, streamNotify, openTaskStream, whenBackendReady } from './api';

import { TopBar } from './components/TopBar';
import { LeftPanel } from './components/LeftPanel';
import { ChatView } from './components/ChatView';
import { TaskCenter } from './components/TaskCenter';
import { FileExplorer } from './components/FileExplorer';
import { StatusBar } from './components/StatusBar';

interface AppContextType {
  state: AppState;
  dispatch: React.Dispatch<Action>;
}
const AppCtx = createContext<AppContextType>({ state: initialAppState, dispatch: () => {} });
export const useApp = () => useContext(AppCtx);

/** 从 ingest_paper 工具的返回 JSON 中提取 task_id（tool_end 归因用）。 */
function extractIngestTaskId(result?: string): string | null {
  if (!result) return null;
  try {
    const obj = JSON.parse(result);
    const tid = obj?.data?.task_id ?? obj?.task_id;
    return typeof tid === 'string' && tid ? tid : null;
  } catch {
    return null;
  }
}

const App: FC = () => {
  const [state, dispatch] = useReducer(appReducer, initialAppState);
  const [apiOnline, setApiOnline] = useState(true);
  const [backendError, setBackendError] = useState('');

  // Fetch server info (papers / index / health). Called on mount and again when
  // the Electron backend reports ready (its port is only known at that point).
  const refreshServerInfo = useCallback(() => {
    api.getAgentHealth().then(h => {
      dispatch({ type: 'SET_AGENT_HEALTH', health: h });
      setApiOnline(true);
    }).catch(() => setApiOnline(false));
    api.getIndexStats().then(s => dispatch({ type: 'SET_INDEX_STATS', stats: s })).catch(() => {});
  }, []);

  // ---- init: load persisted data + fetch server info ----
  useEffect(() => {
    const stored = loadThreads();
    let msgs = state.messages;
    const patch: Partial<AppState> = {};
    if (stored.threadOrder.length > 0) {
      const firstId = stored.threadOrder[0];
      msgs = loadMessages(firstId);
      patch.threads = stored.threads;
      patch.threadOrder = stored.threadOrder;
      patch.activeThreadId = firstId;
      patch.messages = msgs;
    }

    // 任务归因 + 历史跨刷新恢复；本地记录状态基线，避免 reload 后把历史里已完成的
    // parse 任务当「新完成」重复自动触发索引。
    const savedTasks = loadBgTasks();
    for (const t of savedTasks) lastTaskStatus.current[t.taskId] = 'baseline';
    patch.bgTasks = savedTasks;
    dispatch({ type: 'INIT_STATE', state: patch });

    // Load saved panel width
    try {
      const saved = localStorage.getItem('demo_right_w');
      if (saved) {
        const w = parseInt(saved, 10);
        if (w >= 280 && w <= Math.floor(window.innerWidth * 0.55)) {
          dispatch({ type: 'SET_RIGHT_PANEL_WIDTH', width: w });
          document.documentElement.style.setProperty('--right-w', `${w}px`);
        }
      }
    } catch { /* ignore */ }

    // Fetch server info. Electron: only after the main process reports the
    // backend ready (origin is 8001 and uvicorn may take seconds to boot);
    // firing here would proxy-ECONNREFUSED and flip apiOnline off pointlessly.
    // Pure-browser dev: the Vite /api proxy is authoritative from the start.
    if (!window.electronAPI) {
      markProxyReady();
      refreshServerInfo();
    }
  }, [refreshServerInfo]);

  // ---- Electron backend readiness: switch to the real origin + re-fetch ----
  useEffect(() => {
    const bridge = window.electronAPI;
    if (!bridge) return;

    const onReady = (port: number) => {
      setBaseUrl(`http://127.0.0.1:${port}`);
      setBackendError('');
      refreshServerInfo();
    };
    const onError = (msg: string) => {
      setApiOnline(false);
      setBackendError(msg);
    };

    // Replay the already-ready state (race: backend became ready before React
    // mounted and subscribed). Pull-based, so nothing is missed.
    bridge.getBackendStatus().then(status => {
      if (status === 'ready') {
        bridge.getBackendPort().then(port => { if (port) onReady(port); });
      } else if (status === 'error') {
        onError('后端启动失败，常见原因：8001 端口被残留进程占用');
      }
    });

    return bridge.onBackendStatus((data) => {
      if (data.status === 'ready' && data.port) onReady(data.port);
      else if (data.status === 'error') onError(data.error ?? '后端启动失败');
    });
  }, [refreshServerInfo]);

  // ---- persist threads + messages on change ----
  useEffect(() => {
    saveThreads(state.threads, state.threadOrder);
  }, [state.threads, state.threadOrder]);

  useEffect(() => {
    if (state.activeThreadId) {
      saveMessages(state.activeThreadId, state.messages);
    }
  }, [state.messages, state.activeThreadId]);

  // ---- persist background tasks (归因 + 历史, 跨刷新/重启) ----
  useEffect(() => {
    saveBgTasks(state.bgTasks);
  }, [state.bgTasks]);

  // ---- window resize: clamp panel width ----
  useEffect(() => {
    const onResize = () => {
      const maxW = Math.floor(window.innerWidth * 0.55);
      if (state.rightPanelWidth > maxW) {
        const clamped = Math.max(280, maxW);
        dispatch({ type: 'SET_RIGHT_PANEL_WIDTH', width: clamped });
        document.documentElement.style.setProperty('--right-w', `${clamped}px`);
      }
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [state.rightPanelWidth]);

  // ---- thread actions ----
  const handleCreateThread = useCallback(() => {
    dispatch({ type: 'CREATE_THREAD', threadId: crypto.randomUUID() });
  }, []);

  const handleSwitchThread = useCallback((id: string) => {
    const msgs = loadMessages(id);
    dispatch({ type: 'SWITCH_THREAD', threadId: id });
    // After switching, we need to set messages — handled via a hack: re-dispatch
    // ponytail: batch load in a microtask
    setTimeout(() => {
      dispatch({ type: 'INIT_STATE', state: { messages: msgs } });
    }, 0);
  }, []);

  const handleDeleteThread = useCallback((id: string) => {
    dispatch({ type: 'DELETE_THREAD', threadId: id });
    // 该对话运行中的任务解绑为系统级（reducer 已做）；标记 detached 防 first-seen 兜底归回别的对话
    for (const t of bgTasksRef.current) {
      if (t.threadId === id && (t.status === 'running' || t.status === 'pending')) {
        detached.current.add(t.taskId);
      }
    }
    // 清理该对话的权威归因与消息存储
    for (const [tid, th] of forcedThread.current) {
      if (th === id) forcedThread.current.delete(tid);
    }
    try { localStorage.removeItem('demo_msgs_' + id); } catch {}
  }, []);

  // ---- background task stack: poll + announce completions ----
  // The agent-driven ingest runs asynchronously; we poll the stack and
  // surface a completion notice ("agent informs the user") for tasks that
  // finished and opted into notification.
  const bgDismissed = useRef<Set<string>>(new Set());
  const bgKnown = useRef<Record<string, string>>({});   // taskId → last status
  const bgNotified = useRef<Set<string>>(new Set());
  const bgLastJson = useRef('');
  const activeThreadRef = useRef(state.activeThreadId);
  useEffect(() => { activeThreadRef.current = state.activeThreadId; });
  const bgAbort = useRef<AbortController | null>(null);

  // ---- 任务 × 对话 归因 ----
  // 权威归因（tool_end / 上传发起）：taskId → threadId，优先于 first-seen 兜底。
  const forcedThread = useRef<Map<string, string>>(new Map());
  // 删除对话后解绑的运行中任务：标记为系统级，防止 first-seen 兜底又归到别的对话。
  const detached = useRef<Set<string>>(new Set());
  const bgTasksRef = useRef(state.bgTasks);
  useEffect(() => { bgTasksRef.current = state.bgTasks; });
  const threadsRef = useRef(state.threads);
  useEffect(() => { threadsRef.current = state.threads; });
  // pdf 任务状态基线/过渡检测（Web 上传自动触发索引用）。
  const lastTaskStatus = useRef<Record<string, string>>({});
  const autoIndexed = useRef<Set<string>>(new Set());

  const handleDismissBgTask = useCallback((taskId: string) => {
    bgDismissed.current.add(taskId);
    dispatch({ type: 'TASK_DISMISS', taskId });
  }, []);

  const fireNotify = useCallback((task: BackgroundTask) => {
    // 通知落到任务的所属对话（不是完成时激活的对话）；归属对话已删除则跳过
    // （运行中任务被删除对话解绑为系统级后 threadId 为 null，命中 same check）。
    const threads = threadsRef.current;
    const threadId = task.threadId && threads[task.threadId] ? task.threadId : null;
    if (!threadId) return;
    const active = activeThreadRef.current;
    const sysId = crypto.randomUUID();
    const msg: Message = {
      id: sysId,
      role: 'system',
      content: '',
      status: 'streaming',
      timestamp: new Date().toISOString(),
    };

    // 通知落在非激活对话：不能改 state.messages（那是当前对话的数组，会被持久化时
    // 串话）。改为流式消费通知文案后，把完成消息直接写进目标对话的 localStorage，
    // 用户切回该对话时可见。
    if (active !== threadId) {
      let buf = '';
      bgAbort.current?.abort();
      bgAbort.current = streamNotify(threadId, task, {
        onToken: c => { buf += c; },
        onDone: () => {
          const saved = loadMessages(threadId);
          saved.push({ ...msg, content: buf || '完成。', status: 'complete' as const });
          saveMessages(threadId, saved);
          api.getIndexStats().then(st => dispatch({ type: 'SET_INDEX_STATS', stats: st })).catch(() => {});
        },
        onError: message => {
          const saved = loadMessages(threadId);
          saved.push({ ...msg, content: `⚠️ ${message}`, status: 'complete' as const });
          saveMessages(threadId, saved);
        },
      });
      return;
    }

    // 激活对话：流式渲染成系统消息（原有行为）。
    dispatch({ type: 'ADD_MESSAGE', message: msg });
    dispatch({ type: 'SET_STREAMING', isStreaming: true });
    bgAbort.current?.abort();
    bgAbort.current = streamNotify(threadId, task, {
      onToken: c => dispatch({ type: 'APPEND_TOKEN', messageId: sysId, token: c }),
      onDone: () => {
        dispatch({ type: 'FINISH_MESSAGE', messageId: sysId });
        dispatch({ type: 'SET_STREAMING', isStreaming: false });
        api.getIndexStats().then(st => dispatch({ type: 'SET_INDEX_STATS', stats: st })).catch(() => {});
      },
      onError: message => {
        dispatch({ type: 'APPEND_TOKEN', messageId: sysId, token: `\n\n⚠️ ${message}` });
        dispatch({ type: 'FINISH_MESSAGE', messageId: sysId });
        dispatch({ type: 'SET_STREAMING', isStreaming: false });
      },
    });
  }, [dispatch]);

  /** Completion detection → announce the assistant once per task. Shared by the
      SSE push path and the fallback poll. `prev` transition running→done/failed
      is the common case; a task that starts AND finishes before it was first
      seen is covered by "fresh first-seen terminal" (created within 45s). */
  const checkNotify = useCallback((t: BackgroundTask) => {
    if (!t.notify) return;
    const prev = bgKnown.current[t.taskId];
    const terminal = t.status === 'done' || t.status === 'failed';
    const ms = parseTaskTs(t.createdAt);
    const fresh = Number.isFinite(ms) && Date.now() - ms < 45_000;
    const justFinished = (prev && prev !== t.status) || (!prev && fresh);
    if (terminal && justFinished && !bgNotified.current.has(t.taskId)) {
      bgNotified.current.add(t.taskId);
      fireNotify(t);
    }
    bgKnown.current[t.taskId] = t.status;
  }, [fireNotify]);

  /**
   * 给一批任务补归因（仅供还没有 threadId 的任务）：
   *   detached（删除对话解绑的运行中任务）→ 保持系统级；
   *   forced（tool_end/上传发起时记录的权威归因）→ 用它；
   *   否则 first-seen 兜底归到当前活跃对话。
   * 归因结果持久化在 bgTasks（saveBgTasks），刷新/重启不丢。
   */
  const attributeTasks = useCallback((tasks: BackgroundTask[]) => {
    const fallback = activeThreadRef.current;
    for (const t of tasks) {
      if (t.threadId) continue;
      if (detached.current.has(t.taskId)) continue;
      const forced = forcedThread.current.get(t.taskId);
      if (forced) {
        dispatch({ type: 'TASK_ATTRIBUTE', taskId: t.taskId, threadId: forced });
        continue;
      }
      // first-seen 兜底只作用于运行中的任务：terminal 且无归属的任务（如旧版本
      // 迁移数据、被删除对话解绑后完成的系统级任务）不归因，保持不可见，免污染历史。
      if (t.status === 'done' || t.status === 'failed') continue;
      if (fallback) dispatch({ type: 'TASK_ATTRIBUTE', taskId: t.taskId, threadId: fallback });
    }
  }, [dispatch]);

  /** SSE push: one task created/updated in real-time → upsert the strip. */
  const handleTaskEvent = useCallback((t: BackgroundTask) => {
    if (bgDismissed.current.has(t.taskId)) return;
    checkNotify(t);
    dispatch({ type: 'TASK_UPDATE', task: t });
    attributeTasks([t]);
  }, [checkNotify, dispatch, attributeTasks]);

  /** Fallback full sync (slow; SSE is primary) — heals gaps missed during a
      disconnected window. Also used as the immediate pop when a tool enqueues. */
  const syncTasks = useCallback(async () => {
    try {
      const tasks = await api.listTasks();
      const visible = tasks.filter(t => !bgDismissed.current.has(t.taskId));

      // minimal re-render: dispatch only when the visible list actually changed
      // (stage included so parse ✓ → index ⏳ flips render even if progress text is static)
      const sig = JSON.stringify(visible.map(t => [t.taskId, t.status, t.progress, t.stage]));
      if (sig !== bgLastJson.current) {
        bgLastJson.current = sig;
        dispatch({ type: 'TASK_SYNC', tasks: visible });
      }
      visible.forEach(checkNotify);
      attributeTasks(visible);
    } catch {
      // backend offline mid-poll — next call retries
    }
  }, [checkNotify, dispatch, attributeTasks]);

  // ---- upload ----
  // 上传 → SSE 会把解析任务推送进任务中心并由 attributeTasks 归因到当前对话；
  // 解析完成后由下方 watcher 自动触发索引（无需轮询 /pdf/status）。
  const handleUpload = useCallback(async (file: File) => {
    try {
      const result = await api.uploadPDF(file);
      if (result.status === 'duplicate') {
        alert('该论文已存在');
        return;
      }
      const activeId = activeThreadRef.current;
      if (result.task_id && activeId) {
        forcedThread.current.set(result.task_id, activeId);
        void syncTasks(); // 立刻落地任务条，不用等下个 SSE
      }
    } catch (e) {
      alert('上传失败: ' + String(e));
    }
  }, [syncTasks]);

  // ---- Web 上传自动触发索引 ----
  // 上传只是「解析」一个任务；解析 done 后自动补跑一次索引。用前一个观察到的状态
  // 做过渡检测（reload 恢复的历史 done 任务被 'baseline' 挡住，不会重复触发）。
  useEffect(() => {
    for (const t of state.bgTasks) {
      if (t.kind !== 'pdf' || t.status !== 'done' || autoIndexed.current.has(t.taskId)) continue;
      if (lastTaskStatus.current[t.taskId] === 'baseline') {
        lastTaskStatus.current[t.taskId] = t.status;
        continue;
      }
      autoIndexed.current.add(t.taskId);
      lastTaskStatus.current[t.taskId] = t.status;
      const threadId = t.threadId;
      void api.runIndexing(`pdf_pipeline/output/${t.paperName}/rag_chunks.json`)
        .then(idx => {
          if (idx.task_id && threadId) forcedThread.current.set(idx.task_id, threadId);
          void syncTasks();
          api.getIndexStats().then(st => dispatch({ type: 'SET_INDEX_STATS', stats: st })).catch(() => {});
        })
        .catch(() => { /* 索引任务启动失败（如 rag 缺失）——留在解析结果即可 */ });
    }
  }, [state.bgTasks, syncTasks]);

  // Real-time background-task feed: SSE (/api/agent/tasks/stream) is the primary
  // channel — the strip updates the instant a task is created or crosses a stage.
  // A 30s poll remains as a safety net (disconnected window, server restart, …).
  //
  // The stream + first sync are gated on whenBackendReady(): in Electron dev the
  // backend boots after the renderer, so firing the EventSource / initial fetch
  // at mount would bind them to the pre-ready origin '' (Vite proxy). Their stale
  // responses could then race and *clear* the strip right after it appears.
  useEffect(() => {
    let disposed = false;
    let off = () => {};
    void whenBackendReady().then(() => {
      if (disposed) return;
      off = openTaskStream(handleTaskEvent);
      void syncTasks();               // instant fill for tasks running before connect
    });
    const timer = setInterval(() => void syncTasks(), 30_000);
    return () => {
      disposed = true;
      off();
      clearInterval(timer);
      bgAbort.current?.abort();
    };
  }, [handleTaskEvent, syncTasks]);

  // ---- panel drag resize ----
  const handleDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = state.rightPanelWidth;

    const onMove = (ev: MouseEvent) => {
      const maxW = Math.floor(window.innerWidth * 0.55);
      const newW = Math.min(maxW, Math.max(280, startW + startX - ev.clientX));
      document.documentElement.style.setProperty('--right-w', `${newW}px`);
    };

    const onUp = (ev: MouseEvent) => {
      const maxW = Math.floor(window.innerWidth * 0.55);
      const finalW = Math.min(maxW, Math.max(280, startW + startX - ev.clientX));
      document.documentElement.style.setProperty('--right-w', `${finalW}px`);
      dispatch({ type: 'SET_RIGHT_PANEL_WIDTH', width: finalW });
      try { localStorage.setItem('demo_right_w', String(finalW)); } catch { /* ignore */ }
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }, [state.rightPanelWidth]);

  return (
    <AppCtx.Provider value={{ state, dispatch }}>
      <div style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
      }}>
        <TopBar
          leftPanelOpen={state.leftPanelOpen}
          rightPanelOpen={state.rightPanelOpen}
          onToggleLeft={() => dispatch({ type: 'TOGGLE_LEFT_PANEL' })}
          onToggleRight={() => dispatch({ type: 'TOGGLE_RIGHT_PANEL' })}
          onUpload={handleUpload}
          apiOnline={apiOnline}
        />

        {/* Backend startup failure (e.g. :8001 port conflict) — surface it,
            don't let the UI sit silently offline. Retry re-invokes backend.start(). */}
        {backendError && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 10,
            padding: '6px 12px', flexShrink: 0,
            background: '#fdecea', color: '#b3261e',
            borderBottom: '1px solid #f5c6c2', fontSize: 12,
          }}>
            <span>⚠️ {backendError}</span>
            <button
              onClick={() => {
                window.electronAPI?.restartBackend();
                setBackendError('正在重启后端…');
                setApiOnline(true);
              }}
              style={{
                marginLeft: 'auto', padding: '2px 10px', borderRadius: 4,
                border: '1px solid #b3261e', background: '#fff',
                color: '#b3261e', fontSize: 12, cursor: 'pointer',
              }}
            >重试启动后端</button>
            <button
              onClick={() => setBackendError('')}
              title="关闭"
              style={{
                padding: '2px 8px', borderRadius: 4, border: 'none',
                background: 'transparent', color: '#b3261e',
                fontSize: 12, cursor: 'pointer',
              }}
            >✕</button>
          </div>
        )}

        <div style={{
          flex: 1,
          display: 'flex',
          overflow: 'hidden',
        }}>
          <LeftPanel
            open={state.leftPanelOpen}
            threads={state.threads}
            threadOrder={state.threadOrder}
            activeThreadId={state.activeThreadId}
            onCreate={handleCreateThread}
            onSwitch={handleSwitchThread}
            onDelete={handleDeleteThread}
          />

          <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <TaskCenter
              tasks={state.bgTasks}
              activeThreadId={state.activeThreadId}
              onDismiss={handleDismissBgTask}
            />
            {state.activeThreadId ? (
              <ChatView
                key={state.activeThreadId}
                threadId={state.activeThreadId}
                messages={state.messages}
                isStreaming={state.isStreaming}
                onAddMessage={msg => dispatch({ type: 'ADD_MESSAGE', message: msg })}
                onAppendToken={(msgId, token) => dispatch({ type: 'APPEND_TOKEN', messageId: msgId, token })}
                onFinishMessage={msgId => {
                  dispatch({ type: 'FINISH_MESSAGE', messageId: msgId });
                  // Refresh after agent may have downloaded/indexed papers
                  api.getIndexStats().then(st => dispatch({ type: 'SET_INDEX_STATS', stats: st })).catch(() => {});
                }}
                onAbortMessage={msgId => dispatch({ type: 'ABORT_MESSAGE', messageId: msgId })}
                onToolStart={(msgId, step) => dispatch({ type: 'ADD_TOOL_STEP', messageId: msgId, step })}
                onToolEnd={(msgId, toolCallId, patch, name, threadId) => {
                  dispatch({ type: 'UPDATE_TOOL_STEP', messageId: msgId, toolCallId, patch });
                  // Agent 刚触发入库 —— 从返回 JSON 提取 task_id 记录权威归因
                  //（归属到发起该对话而非完成时激活的对话），并立刻同步任务条。
                  if (name === 'ingest_paper') {
                    const taskId = extractIngestTaskId(patch.result);
                    if (taskId && threadId) forcedThread.current.set(taskId, threadId);
                    void syncTasks();
                  }
                }}
                onPlan={(msgId, plan) => dispatch({ type: 'SET_PLAN', messageId: msgId, plan })}
                onSetStreaming={v => dispatch({ type: 'SET_STREAMING', isStreaming: v })}
              />
            ) : (
              <div style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: 'var(--color-text-secondary)', fontSize: 15,
              }}>
                👈 新建或选择一个对话开始
              </div>
            )}
          </main>

          {/* Drag handle: resize right panel */}
          <div
            onMouseDown={handleDragStart}
            style={{
              width: 4, cursor: 'col-resize', flexShrink: 0,
              background: 'transparent',
              transition: 'background 0.15s',
            }}
            onMouseEnter={e => (e.currentTarget as HTMLDivElement).style.background = 'var(--color-primary)'}
            onMouseLeave={e => (e.currentTarget as HTMLDivElement).style.background = 'transparent'}
          />

          <FileExplorer open={state.rightPanelOpen} />
        </div>

        <StatusBar
          indexStats={state.indexStats}
          agentHealth={state.agentHealth}
        />
      </div>
    </AppCtx.Provider>
  );
};

export default App;
