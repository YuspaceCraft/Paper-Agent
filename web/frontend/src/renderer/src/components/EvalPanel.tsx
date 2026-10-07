/**
 * EvalPanel.tsx - Agent evaluation workspace (evaluation package).
 *
 * Mounted inside the system-level EvaluationCenter, not beside business
 * workspace tabs.  Single-run is the primary workflow; dataset reports and
 * badcase traces are the next layer.
 *
 * 实时（2026-09-15）：跑批/单样例都走 SSE —— 跑批每条 QA 完成即刷新进度条 +
 * 累计指标（recall/mrr/task/工具/token/成本/ETA）与逐条明细；单样例每个阶段
 * 出现即刷新分阶段视图。SSE 断线由 EventSource 自动重连（服务端回放历史），
 * 不支持 SSE 时单样例退回一次性 POST，跑批退回 5s 轮询列表（服务端写进度行）。
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  api, whenBackendReady, openEvalRunStream, openEvalSingleStream,
  type EvalBadcase, type EvalLiveEvent, type EvalLiveQueryRow,
  type EvalRunReport, type EvalTraceEvent, type EvalRunsResp,
  type PromptsConfigInfo,
} from '../api';
import { MessageSteps } from './MessageSteps';
import { SingleFlowView } from './SingleFlowView';
import type { EvalSingleFlow } from '../api';
import type { ToolStep } from '../state';

const card: React.CSSProperties = {
  border: '1px solid var(--color-border)', borderRadius: 8,
  padding: '8px 10px', marginBottom: 6, background: 'var(--color-surface)',
};
const rowBtn: React.CSSProperties = {
  display: 'flex', justifyContent: 'space-between', gap: 8, width: '100%',
  border: 'none', background: 'transparent', cursor: 'pointer', padding: 0,
  textAlign: 'left', fontSize: 13, fontFamily: 'inherit',
};
const pill: React.CSSProperties = {
  padding: '1px 8px', borderRadius: 10, fontSize: 11,
  background: 'var(--color-primary-light)', color: 'var(--color-primary)',
};
const metricGrid: React.CSSProperties = {
  display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 4, margin: '6px 0',
};
const metricCell: React.CSSProperties = {
  background: 'var(--color-inset)', borderRadius: 6, padding: '4px 6px', fontSize: 12,
};
const mono: React.CSSProperties = {
  fontFamily: 'var(--font-mono)', fontSize: 11, whiteSpace: 'pre-wrap',
  wordBreak: 'break-all', color: 'var(--color-text-secondary)',
};
const inputStyle: React.CSSProperties = {
  width: '100%', padding: '5px 8px', borderRadius: 6, fontSize: 12,
  border: '1px solid var(--color-border)', background: 'var(--color-surface)',
  boxSizing: 'border-box', color: 'inherit',
};

type EvalRunView = 'overview' | 'cases' | 'trace' | 'compare' | 'config';

// trace 事件 → ToolStep 树（MessageSteps 直接消费）
function eventsToSteps(events: EvalTraceEvent[]): ToolStep[] {
  const steps: ToolStep[] = [];
  for (const e of events) {
    const p = e.payload ?? {};
    const t = e.event_type;
    const id = `ev-${e.seq}`;
    if (t === 'tool_call') {
      steps.push({
        id, name: e.tool ?? 'tool',
        args: (p.args as Record<string, unknown>) ?? undefined,
        result: p.operation
          ? JSON.stringify(p.operation, null, 2)
          : String(p.result_summary ?? p.error ?? e.error ?? ''),
        status: e.outcome === 'succeeded' ? 'success' : 'error',
        executionTime: e.duration_ms ?? undefined,
        kind: 'tool',
      });
    } else if (t === 'retrieved_context') {
      const cids = (p.chunk_ids as string[] | undefined) ?? [];
      steps.push({
        id, name: '检索上下文',
        args: p.query ? { query: p.query } : undefined,
        result: `chunks(${cids.length}): ${cids.join(', ')}\n${String(p.snapshot ?? '').slice(0, 600)}`,
        status: 'success', kind: 'tool',
      });
    } else if (t === 'llm_call') {
      const tok = (p.tokens ?? {}) as Record<string, number>;
      steps.push({
        id, name: `LLM · ${e.model ?? ''}`,
        result: `p/c/t tokens = ${tok.prompt_tokens ?? ''}/${tok.completion_tokens ?? ''}/${tok.total_tokens ?? ''} · ${(e.duration_ms ?? 0).toFixed(1)}ms`,
        status: e.error || p.error ? 'error' : 'success',
        kind: 'llm', executionTime: e.duration_ms ?? undefined,
      });
    } else if (t === 'intent') {
      steps.push({
        id, name: `意图 · ${e.intent ?? ''}`,
        result: JSON.stringify({
          confidence: p.confidence, entities: p.entities,
          needs_planning: p.needs_planning, domain: p.domain,
        }),
        status: 'success', kind: 'intent',
      });
    } else if (t === 'plan') {
      steps.push({
        id, name: `计划（${(p.steps as unknown[] | undefined)?.length ?? 0} 步）`,
        result: JSON.stringify(p.steps ?? [], null, 1).slice(0, 800),
        status: 'success', kind: 'plan',
      });
    } else if (t === 'plan_step' || t === 'plan_verify') {
      const st = String(p.status ?? '');
      steps.push({
        id, name: `${t === 'plan_verify' ? '验证' : '计划步'} · ${st}`,
        result: String(p.detail ?? p.verification_status ?? ''),
        status: st === 'failed' || st === 'skipped' ? 'error' : 'success',
        kind: 'plan', executionTime: e.duration_ms ?? undefined,
      });
    } else if (t === 'final_answer') {
      steps.push({
        id, name: '最终回答', result: String(p.answer ?? ''), status: 'success', kind: 'answer',
      });
    }
    // turn_start / node_start / node_end / node_error / turn_end 为链路上下文，不单列
  }
  return steps;
}

function fmt(x: unknown, unit = ''): string {
  if (x === null || x === undefined || x === '') return '—';
  const v = Number(x);
  return Number.isFinite(v) ? `${Math.round(v * 100) / 100}${unit}` : String(x).slice(0, 12);
}

/** 跑批实时进度卡：进度条 + 累计指标 + 最近逐条明细（随 SSE 事件刷新）。 */
function LiveRunCard({ live, rows }: { live: EvalLiveEvent; rows: EvalLiveQueryRow[] }) {
  const ov = (live.overall ?? {}) as Record<string, unknown>;
  const done = Number(live.done ?? 0);
  const total = Number(live.total ?? 0);
  const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
  const failed = live.type === 'run_failed';
  const elapsed = live.elapsed_s;
  const running = !failed && live.type !== 'run_finished';
  // 终态一律显示状态词（done / failed / interrupted）+ 完整用时
  const stateLabel = running ? 'running' : (live.status || (failed ? 'failed' : 'done'));
  const shownSeconds = running ? elapsed : (live.duration_s ?? elapsed);

  return (
    <div style={{ ...card, borderColor: running ? 'var(--color-primary)' : 'var(--color-border)' }}>
      <div style={{ fontSize: 12, fontWeight: 600, display: 'flex', justifyContent: 'space-between', gap: 8 }}>
        <span style={{ fontFamily: 'var(--font-mono)' }}>实时进度 · {live.run_id}</span>
        <span style={pill}>{stateLabel}</span>
      </div>
      <div style={{ height: 6, borderRadius: 3, background: 'var(--color-inset)', margin: '6px 0' }}>
        <div style={{ width: `${pct}%`, height: '100%', borderRadius: 3,
                      background: 'var(--color-primary)', transition: 'width .3s' }} />
      </div>
      <div style={mono}>
        {done}/{total || '?'}（{pct}%）· {running ? '已用' : '用时'} {fmt(shownSeconds)}s
        {live.eta_s ? ` · ETA ${fmt(live.eta_s)}s` : ''}
        {live.qid ? ` · 当前 ${live.qid}` : ''}
        {live.gate ? ` · gate ${live.gate}` : ''}
        {live.badcase_count !== undefined ? ` · badcase ${live.badcase_count}` : ''}
      </div>
      <div style={metricGrid}>
        <div style={metricCell}>recall@5 <b>{fmt(ov['recall@5'])}</b></div>
        <div style={metricCell}>mrr <b>{fmt(ov.mrr)}</b></div>
        <div style={metricCell}>ndcg@10 <b>{fmt(ov['ndcg@10'])}</b></div>
        <div style={metricCell}>任务成功率 <b>{fmt(ov.task_success_rate)}</b></div>
        <div style={metricCell}>
          工具成功率 <b>{fmt(ov.tool_success_rate)}</b>
          {ov.tool_calls !== undefined
            ? <span style={{ color: 'var(--color-text-secondary)' }}>
                （{fmt(ov.tool_calls)} 调用 / {fmt(ov.tool_failures)} 失败）
              </span>
            : null}
        </div>
        <div style={metricCell}>tokens <b>{fmt(ov.tokens_total)}</b></div>
      </div>
      {failed && live.error && (
        <div style={{ color: 'var(--color-warning)', fontSize: 12 }}>{live.error}</div>
      )}
      {rows.length > 0 && (
        <div style={{ borderTop: '1px solid var(--color-border)', paddingTop: 5, marginTop: 4 }}>
          {rows.slice(-10).map(r => (
            <div key={`${r.index}-${r.qid}`}
              style={{ display: 'flex', gap: 6, fontSize: 11, fontFamily: 'var(--font-mono)' }}>
              <span style={{ width: 46 }}>{r.index}/{r.total}</span>
              <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {String(r.query ?? '')}
              </span>
              <span style={{ color: r.category && r.category !== 'ok'
                ? 'var(--color-warning)' : 'var(--color-text-secondary)' }}>
                {r.category || 'ok'}
              </span>
              <span>r@5={fmt(r['recall@5'])}</span>
              <span>{fmt(r.duration_s)}s</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

interface EvalPanelProps {
  section?: 'single' | 'runs';
  onSectionChange?: (section: 'single' | 'runs') => void;
  hideSectionNav?: boolean;
}

export function EvalPanel({
  section,
  onSectionChange,
  hideSectionNav = false,
}: EvalPanelProps = {}) {
  const [internalView, setInternalView] = useState<'single' | 'runs'>('single');
  const view = section ?? internalView;
  const setView = useCallback((next: 'single' | 'runs') => {
    if (section === undefined) setInternalView(next);
    onSectionChange?.(next);
  }, [onSectionChange, section]);
  const [resp, setResp] = useState<EvalRunsResp | null>(null);
  const [report, setReport] = useState<EvalRunReport | null>(null);
  const [runView, setRunView] = useState<EvalRunView>('overview');
  const [traceEvents, setTraceEvents] = useState<EvalTraceEvent[] | null>(null);
  const [manifest, setManifest] = useState('eval_output/datasets/manifest_v2_current_kb_50qa.jsonl');
  const [limitStr, setLimitStr] = useState('20');
  const [starting, setStarting] = useState(false);
  const [msg, setMsg] = useState('');
  // 单样例评测
  const [sqQuery, setSqQuery] = useState('');
  const [sqMode, setSqMode] = useState('auto');
  const [sqRunning, setSqRunning] = useState(false);
  const [flow, setFlow] = useState<EvalSingleFlow | null>(null);
  const [flowB, setFlowB] = useState<EvalSingleFlow | null>(null);
  const [sqElapsed, setSqElapsed] = useState(0);
  const [sqElapsedB, setSqElapsedB] = useState(0);
  const [abEnabled, setAbEnabled] = useState(false);
  const [promptConfig, setPromptConfig] = useState<PromptsConfigInfo | null>(null);
  const [abPromptId, setAbPromptId] = useState('');
  const [abVersionA, setAbVersionA] = useState('');
  const [abVersionB, setAbVersionB] = useState('');
  // 实时进度（SSE）：跑批逐条刷新 liveRows + 累计指标；单样例逐阶段刷新 flow
  const [live, setLive] = useState<EvalLiveEvent | null>(null);
  const [liveRows, setLiveRows] = useState<EvalLiveQueryRow[]>([]);
  const liveClose = useRef<(() => void) | null>(null);
  const sqClose = useRef<(() => void) | null>(null);
  const sqCloseB = useRef<(() => void) | null>(null);
  const followRun = useRef<string | null>(null);

  const refreshRuns = useCallback(async () => {
    await whenBackendReady();
    try {
      setResp(await api.listEvalRuns(20));
    } catch { /* 后端未起/未评测过都静默 */ }
  }, []);

  useEffect(() => { void refreshRuns(); }, [refreshRuns]);
  useEffect(() => {
    let alive = true;
    void whenBackendReady().then(() => api.getPromptsConfig()).then(data => {
      if (!alive) return;
      setPromptConfig(data);
      const first = data.prompts.find(item => item.versions.length > 1);
      if (!first) return;
      const active = first.active?.version ?? first.versions[0]?.version ?? '';
      const candidate = first.versions.find(item => item.status === 'canary')?.version
        ?? first.versions.find(item => item.version !== active)?.version
        ?? active;
      setAbPromptId(first.id);
      setAbVersionA(active);
      setAbVersionB(candidate);
    }).catch(() => { /* prompt A/B is optional */ });
    return () => { alive = false; };
  }, []);
  useEffect(() => {
    if (!resp) return;
    const timer = setInterval(() => void refreshRuns(), 5000);
    return () => clearInterval(timer);
  }, [refreshRuns, resp]);
  useEffect(() => () => {
    liveClose.current?.();
    sqClose.current?.();
    sqCloseB.current?.();
  }, []);

  /** 订阅一个正在跑的 run：逐条指标 + 累计聚合实时刷新（断线自动重连）。 */
  const watchRun = useCallback((runId: string) => {
    liveClose.current?.();
    setLive({ type: 'run_started', run_id: runId, status: 'running', done: 0, total: 0 });
    setLiveRows([]);
    liveClose.current = openEvalRunStream(runId, ev => {
      // 事件是增量语义（query_started 只带当前位置、aggregate 才带累计指标）：
      // 合并而不是覆盖，否则中间帧会把已经算出来的指标闪成 undefined。
      setLive(prev => {
        const base: EvalLiveEvent = prev ?? {
          type: 'run_started', run_id: runId, status: 'running',
        };
        const merged: EvalLiveEvent = { ...base, ...ev };
        if (ev.type === 'query_finished' && ev.index) {
          merged.done = Math.max(Number(merged.done ?? 0), ev.index);
        }
        return merged;
      });
      if (ev.recent?.length) setLiveRows(ev.recent);
      else if (ev.type === 'query_finished') {
        setLiveRows(prev => (prev.some(r => r.qid === ev.qid)
          ? prev : [...prev, ev as EvalLiveQueryRow].slice(-20)));
      }
      if (ev.type === 'run_finished' || ev.type === 'run_failed') {
        liveClose.current?.();
        liveClose.current = null;
        void refreshRuns();
        if (ev.type === 'run_finished' && followRun.current === runId) {
          void api.getEvalRun(runId).then(setReport).catch(() => { /* 稍后手动刷新 */ });
        }
      }
    });
  }, [refreshRuns]);

  const openRun = useCallback(async (runId: string) => {
    setTraceEvents(null);
    setMsg('');
    followRun.current = null;
    try {
      const r = await api.getEvalRun(runId);
      setReport(r);
      setRunView('overview');
      if (r.status === 'running') watchRun(runId);
    } catch (e) { setMsg(`加载报告失败: ${String(e)}`); }
  }, [watchRun]);

  const openBadcase = useCallback(async (b: EvalBadcase) => {
    setMsg('');
    if (!b.trace_id) { setMsg('该 badcase 无 trace_id'); return; }
    try {
      const r = await api.getEvalTrace(b.trace_id);
      setTraceEvents(r.events);
      setRunView('cases');
    } catch (e) { setMsg(`加载 trace 失败: ${String(e)}`); }
  }, []);

  const startRun = useCallback(async () => {
    if (!manifest.trim()) return;
    setStarting(true); setMsg('');
    try {
      const r = await api.startEvalRun({
        manifest: manifest.trim(), limit: Number(limitStr) || 0, judge: true,
      });
      setReport(null);
      setMsg(`评测已启动: ${r.run_id}（实时进度见下，跑完自动切到完整报告）`);
      followRun.current = r.run_id;
      watchRun(r.run_id);
    } catch (e) { setMsg(`启动失败: ${String(e)}`); }
    finally { setStarting(false); }
  }, [manifest, limitStr, watchRun]);

  const runSingle = useCallback(async () => {
    const q = sqQuery.trim();
    if (!q) return;
    setSqRunning(true);
    setMsg('');
    setFlow(null);
    setFlowB(null);
    setSqElapsed(0);
    setSqElapsedB(0);
    sqClose.current?.();
    sqCloseB.current?.();
    sqClose.current = null;
    sqCloseB.current = null;
    try {
      if (abEnabled && abPromptId && abVersionA && abVersionB) {
        const base = `eval-single-ab-${Date.now().toString(36)}`;
        const [startedA, startedB] = await Promise.all([
          api.startEvalSingle(q, {
            threadId: `${base}-a`,
            mode: sqMode === 'auto' ? undefined : sqMode,
            promptOverrides: { [abPromptId]: abVersionA },
          }),
          api.startEvalSingle(q, {
            threadId: `${base}-b`,
            mode: sqMode === 'auto' ? undefined : sqMode,
            promptOverrides: { [abPromptId]: abVersionB },
          }),
        ]);
        let doneA = false;
        let doneB = false;
        const finishVariant = () => {
          if (doneA && doneB) setSqRunning(false);
        };
        sqClose.current = openEvalSingleStream(startedA.thread_id, ev => {
          if (ev.flow) setFlow(ev.flow);
          if (ev.elapsed_s !== undefined) setSqElapsed(ev.elapsed_s);
          if (ev.error) setMsg(`A 组中断: ${ev.error}`);
          if (ev.type === 'single_done' || ev.type === 'single_error') {
            doneA = true;
            sqClose.current?.();
            sqClose.current = null;
            finishVariant();
          }
        });
        sqCloseB.current = openEvalSingleStream(startedB.thread_id, ev => {
          if (ev.flow) setFlowB(ev.flow);
          if (ev.elapsed_s !== undefined) setSqElapsedB(ev.elapsed_s);
          if (ev.error) setMsg(`B 组中断: ${ev.error}`);
          if (ev.type === 'single_done' || ev.type === 'single_error') {
            doneB = true;
            sqCloseB.current?.();
            sqCloseB.current = null;
            finishVariant();
          }
        });
        return;
      }
      const started = await api.startEvalSingle(
        q, { mode: sqMode === 'auto' ? undefined : sqMode });
      sqClose.current = openEvalSingleStream(started.thread_id, ev => {
        if (ev.flow) setFlow(ev.flow);
        if (ev.elapsed_s !== undefined) setSqElapsed(ev.elapsed_s);
        if (ev.error) setMsg(`单样例流中断: ${ev.error}`);
        if (ev.type === 'single_done' || ev.type === 'single_error') {
          sqClose.current?.();
          sqClose.current = null;
          setSqRunning(false);
        }
      });
    } catch (error) {
      if (abEnabled) {
        setMsg(`A/B 单样例评测失败: ${String(error)}`);
        setSqRunning(false);
        return;
      }
      // 后端不支持流式（旧版本）→ 退回一次性 POST
      try {
        setFlow(await api.runSingleEval(q, { mode: sqMode === 'auto' ? undefined : sqMode }));
      } catch (e) {
        setMsg(`单样例评测失败: ${String(e)}`);
      } finally { setSqRunning(false); }
    }
  }, [abEnabled, abPromptId, abVersionA, abVersionB, sqQuery, sqMode]);

  const back = useCallback(() => {
    setReport(null); setTraceEvents(null); setMsg('');
    liveClose.current?.(); liveClose.current = null;
    setLive(null); setLiveRows([]);
  }, []);

  const abCandidates = (promptConfig?.prompts ?? []).filter(item => item.versions.length > 1);
  const selectedAbPrompt = abCandidates.find(item => item.id === abPromptId) ?? abCandidates[0];
  const abVersions = selectedAbPrompt?.versions ?? [];

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* 顶层视图切换（嵌入评测中心时由外层导航负责） */}
      {!hideSectionNav && (
      <div style={{ display: 'flex', gap: 4, padding: '6px 8px', borderBottom: '1px solid var(--color-border)', flexShrink: 0 }}>
        {([['single', '🔬 单样例'], ['runs', '📊 评测报告']] as const).map(([k, label]) => (
          <button key={k}
            onClick={() => { setView(k); }}
            style={{
              flex: 1, padding: '5px 0', borderRadius: 6, fontSize: 13, fontWeight: 600,
              border: 'none', cursor: 'pointer', fontFamily: 'inherit',
              background: view === k ? 'var(--color-primary-light)' : 'transparent',
              color: view === k ? 'var(--color-primary)' : 'var(--color-text-secondary)',
            }}>{label}</button>
        ))}
      </div>
      )}

      {/* 单样例透明评测 */}
      {view === 'single' && (
        <div style={{ padding: '10px 12px', borderBottom: '1px solid var(--color-border)', flexShrink: 0 }}>
          <div style={{ display: 'flex', gap: 8 }}>
            <input style={{ ...inputStyle, flex: 1 }} value={sqQuery} onChange={e => setSqQuery(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') void runSingle(); }}
              placeholder="输入一个问题，逐节点检查完整 Agent 链路" />
            <select value={sqMode} onChange={e => setSqMode(e.target.value)}
              style={{ ...inputStyle, width: 105, cursor: 'pointer' }}>
              <option value="auto">auto</option>
              <option value="react">react</option>
              <option value="plan">plan</option>
            </select>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8, flexWrap: 'wrap' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--color-text-secondary)' }}>
              <input
                type="checkbox"
                checked={abEnabled}
                disabled={abCandidates.length === 0}
                onChange={e => {
                  const enabled = e.target.checked;
                  setAbEnabled(enabled);
                  if (enabled && selectedAbPrompt && !abPromptId) {
                    setAbPromptId(selectedAbPrompt.id);
                  }
                }}
              />
              A/B 配置版本
            </label>
            {abEnabled && selectedAbPrompt && (
              <>
                <select
                  value={selectedAbPrompt.id}
                  onChange={e => {
                    const prompt = abCandidates.find(item => item.id === e.target.value);
                    if (!prompt) return;
                    setAbPromptId(prompt.id);
                    setAbVersionA(prompt.active?.version ?? prompt.versions[0]?.version ?? '');
                    setAbVersionB(
                      prompt.versions.find(item => item.status === 'canary')?.version
                      ?? prompt.versions.find(item => item.version !== prompt.active?.version)?.version
                      ?? '',
                    );
                  }}
                  style={{ ...inputStyle, width: 180, fontSize: 11 }}
                >
                  {abCandidates.map(item => <option key={item.id} value={item.id}>{item.id}</option>)}
                </select>
                <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>A</span>
                <select value={abVersionA} onChange={e => setAbVersionA(e.target.value)}
                  style={{ ...inputStyle, width: 125, fontSize: 11 }}>
                  {abVersions.map(item => (
                    <option key={item.version} value={item.version}>
                      {item.version} · {item.status}
                    </option>
                  ))}
                </select>
                <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>B</span>
                <select value={abVersionB} onChange={e => setAbVersionB(e.target.value)}
                  style={{ ...inputStyle, width: 125, fontSize: 11 }}>
                  {abVersions.map(item => (
                    <option key={item.version} value={item.version}>
                      {item.version} · {item.status}
                    </option>
                  ))}
                </select>
              </>
            )}
            <button onClick={runSingle} disabled={sqRunning || !sqQuery.trim()}
              style={{ marginLeft: 'auto', cursor: 'pointer', borderRadius: 6, fontSize: 12, fontWeight: 600,
                       border: 'none', padding: '6px 16px', background: 'var(--color-primary)', color: '#fff' }}>
              {sqRunning
                ? `运行中… ${sqElapsed || 0}s${abEnabled ? ` / B ${sqElapsedB || 0}s` : ''}`
                : abEnabled ? '▶ 运行 A/B' : '▶ 运行单次'}
            </button>
          </div>
          {abCandidates.length === 0 && (
            <div style={{ color: 'var(--color-text-tertiary)', fontSize: 10, marginTop: 5 }}>
              暂无多版本 Prompt；发布 canary 版本后可直接在这里并排评测。
            </div>
          )}
          {msg && <div style={{ color: 'var(--color-warning)', fontSize: 12, marginTop: 5 }}>{msg}</div>}
        </div>
      )}

      {view === 'runs' && (
      <div style={{ padding: '8px 10px', borderBottom: '1px solid var(--color-border)', flexShrink: 0 }}>
        <input style={inputStyle} value={manifest} onChange={e => setManifest(e.target.value)}
          placeholder="manifest 路径（JSONL）" />
        <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
          <input style={{ ...inputStyle, width: 60 }} value={limitStr}
            onChange={e => setLimitStr(e.target.value)} placeholder="limit" />
          <button onClick={startRun} disabled={starting}
            style={{ flex: 1, cursor: 'pointer', borderRadius: 6, fontSize: 12, fontWeight: 600,
                     border: 'none', padding: '5px 0',
                     background: 'var(--color-primary)', color: '#fff' }}>
            {starting ? '启动中…' : '▶ 启动评测'}
          </button>
        </div>
        {msg && <div style={{ color: 'var(--color-warning)', fontSize: 12, marginTop: 4 }}>{msg}</div>}
      </div>
      )}

      {/* 内容区：单样例 flow 或 评测报告 */}
      <div style={{ flex: 1, overflow: 'auto', padding: 8 }}>
        {view === 'single' ? (
          flow || flowB ? (
            <>
              <button onClick={() => { setFlow(null); setFlowB(null); }} style={pill}>← 清除</button>
              {flowB ? (
                <div className="eval-ab-grid" style={{ marginTop: 8 }}>
                  <div>
                    <div className="eval-variant-label">
                      A · {abVersionA || '默认配置'}
                    </div>
                    {flow && <SingleFlowView flow={flow} isRunning={sqRunning} />}
                  </div>
                  <div>
                    <div className="eval-variant-label">
                      B · {abVersionB || '候选配置'}
                    </div>
                    <SingleFlowView flow={flowB} isRunning={sqRunning} />
                  </div>
                </div>
              ) : (
                <div style={{ marginTop: 8 }}>
                  {flow && <SingleFlowView flow={flow} isRunning={sqRunning} />}
                </div>
              )}
            </>
          ) : (
            <div className="eval-empty-state">
              <div className="eval-empty-symbol">◎</div>
              <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--color-text)' }}>
                {sqRunning ? '运行已开始，等待首个节点…' : '检查一条问题的完整运行链路'}
              </div>
              <div style={{ maxWidth: 620, fontSize: 12, lineHeight: 1.7 }}>
                运行后按节点展示配置冻结、问题分析、上下文、LLM 思考、工具治理与执行、
                子 Agent、结果解析和最终回答；指标会随节点推进同步更新。
              </div>
            </div>
          )
        ) : (
          <>
            {live && live.type !== 'run_finished' && (
              <LiveRunCard live={live} rows={liveRows} />
            )}
            {report && (
              <>
                <div className="eval-report-toolbar">
                  <button onClick={back} style={pill}>← 返回列表</button>
                  <div className="eval-report-tabs" role="tablist" aria-label="评测报告视图">
                    {([
                      ['overview', '概览'],
                      ['cases', '案例'],
                      ['trace', '轨迹'],
                      ['compare', '对比'],
                      ['config', '配置'],
                    ] as const).map(([key, label]) => (
                      <button
                        key={key}
                        type="button"
                        role="tab"
                        aria-selected={runView === key}
                        className={runView === key ? 'is-active' : ''}
                        onClick={() => setRunView(key)}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </div>
                {runView === 'overview' && (
                  <RunOverview report={report} onBadcase={openBadcase} />
                )}
                {runView === 'cases' && (
                  <RunCases
                    report={report}
                    traceEvents={traceEvents}
                    onBadcase={openBadcase}
                  />
                )}
                {runView === 'trace' && (
                  <RunTrace
                    report={report}
                    traceEvents={traceEvents}
                    onBadcase={openBadcase}
                  />
                )}
                {runView === 'compare' && (
                  <RunCompare report={report} runs={resp?.runs ?? []} />
                )}
                {runView === 'config' && <RunConfig report={report} />}
              </>
            )}
        {!report && (
          <>
            <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 4 }}>
              最近评测{resp ? `（${resp.count}）` : ''}：
            </div>
            {(resp?.runs ?? []).map(r => {
              const ov = (r.overall ?? {}) as Record<string, unknown>;
              const prog = (ov.progress ?? {}) as Record<string, unknown>;
              const isRunning = r.status === 'running';
              const pct = isRunning && Number(prog.total)
                ? Math.min(100, Math.round((Number(prog.done ?? 0) / Number(prog.total)) * 100))
                : 0;
              return (
                <div key={r.run_id} style={card}>
                  <button onClick={() => void openRun(r.run_id)} style={rowBtn}>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>{r.run_id}</span>
                    <span style={pill}>
                      {isRunning ? `running ${prog.done ?? 0}/${prog.total ?? '?'}` : r.status}
                    </span>
                  </button>
                  {isRunning && (
                    <div style={{ height: 4, borderRadius: 2, marginTop: 5,
                                  background: 'var(--color-inset)' }}>
                      <div style={{ width: `${pct}%`, height: '100%', borderRadius: 2,
                                    background: 'var(--color-primary)', transition: 'width .3s' }} />
                    </div>
                  )}
                  <div style={metricGrid}>
                    <div style={metricCell}>recall@5 <b>{fmt(ov['recall@5'])}</b></div>
                    <div style={metricCell}>mrr <b>{fmt(ov.mrr)}</b></div>
                    <div style={metricCell}>task <b>{fmt(ov.task_success_rate)}</b></div>
                  </div>
                  <div style={mono}>
                    n={r.query_count} · {r.dataset_id} · 用时 {fmt(r.duration_s)}s
                    {r.badcase_count ? ` · badcase ${r.badcase_count}` : ''}
                    {' · '}{r.started_at}
                  </div>
                </div>
              );
            })}
            {!resp?.runs?.length && (
              <div style={{ color: 'var(--color-text-secondary)', fontSize: 12, lineHeight: 1.6 }}>
                尚无评测 —— 用上方「启动评测」跑批（默认小样本 limit=20，可改），或 CLI
                <code style={{ display: 'block', margin: '6px 0' }}>
                  C:/Users/30811/miniconda3/envs/demo/python.exe -m evaluation run …</code>
              </div>
            )}
          </>
        )}
            </>
          )}
      </div>
    </div>
  );
}

interface EvalIssue {
  priority: 'P0' | 'P1';
  title: string;
  scope: string;
  evidence: string;
  action: string;
  badcase?: EvalBadcase;
}

interface DeltaInfo {
  prev?: number;
  now?: number;
  delta_pct?: number;
  flag?: string;
}

function asDelta(value: unknown): DeltaInfo {
  return value && typeof value === 'object'
    ? value as DeltaInfo
    : {};
}

function metricValue(record: Record<string, unknown>, key: string): number | null {
  const raw = record[key];
  if (raw === null || raw === undefined || raw === '') return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function deltaText(delta?: DeltaInfo): string {
  if (!delta || !Number.isFinite(Number(delta.delta_pct))) return '—';
  const value = Number(delta.delta_pct);
  return `${value > 0 ? '+' : ''}${fmt(value)}%`;
}

function verdictTone(flag?: string): 'ok' | 'warn' | 'danger' | 'muted' {
  if (flag === 'pass') return 'ok';
  if (flag === 'warn') return 'warn';
  if (flag === 'block' || flag === 'fail') return 'danger';
  return 'muted';
}

function verdictLabel(flag?: string): string {
  if (flag === 'pass') return '通过';
  if (flag === 'warn') return '警戒';
  if (flag === 'block' || flag === 'fail') return '阻塞';
  return '观察';
}

function buildEvalIssues(report: EvalRunReport): EvalIssue[] {
  const ov = (report.overall ?? {}) as Record<string, unknown>;
  const baseline = (report.baseline_delta ?? {}) as Record<string, unknown>;
  const deltas = (baseline.deltas ?? {}) as Record<string, unknown>;
  const badcases = report.badcases ?? [];
  const issues: EvalIssue[] = [];
  const retrievalDelta = asDelta(deltas['recall@5']);
  const taskDelta = asDelta(deltas.task_success_rate);
  const toolSuccess = metricValue(ov, 'tool_success_rate');
  const toolFailures = metricValue(ov, 'tool_failures');
  const latencyP95 = metricValue(ov, 'answer_latency_p95_s');
  const faithfulness = metricValue(report.judge ?? {}, 'faithfulness');

  if (retrievalDelta.flag && retrievalDelta.flag !== 'pass') {
    const retrievalCases = badcases.filter(item =>
      /retrieval|mrr|recall/i.test(item.category),
    );
    issues.push({
      priority: retrievalDelta.flag === 'block' ? 'P0' : 'P1',
      title: '检索质量相对 baseline 下降',
      scope: `Recall@5 ${fmt(ov['recall@5'])} · ${retrievalCases.length || badcases.length} 个相关案例`,
      evidence: `baseline ${fmt(retrievalDelta.prev)}，本次 ${fmt(retrievalDelta.now)}，变化 ${deltaText(retrievalDelta)}。`,
      action: '先检查失败样本的排序位置、chunk 切分和查询改写。',
      badcase: retrievalCases[0] ?? badcases[0],
    });
  }

  if ((toolSuccess !== null && toolSuccess < 1) || (toolFailures ?? 0) > 0) {
    const toolCases = badcases.filter(item => (item.tool_errors ?? 0) > 0);
    issues.push({
      priority: 'P0',
      title: '工具调用失败拉低任务成功率',
      scope: `${fmt(ov.tool_failures)} 次失败 · ${toolCases.length} 个 degraded/failed 案例`,
      evidence: `工具成功率 ${fmt(toolSuccess)}；失败记录已写入 trace，可直接回放。`,
      action: '按首个失败工具聚类，检查超时、重试预算和结果信封。',
      badcase: toolCases[0] ?? badcases[0],
    });
  }

  if (latencyP95 !== null && latencyP95 > 30) {
    issues.push({
      priority: 'P1',
      title: 'P95 延迟超出交互预算',
      scope: `P95 ${fmt(latencyP95)}s · 目标 ≤ 30s`,
      evidence: `p50 ${fmt(ov.answer_latency_p50_s)}s，最长 ${fmt(ov.answer_latency_max_s)}s。`,
      action: '从最慢的工具与 LLM 调用开始定位，优先处理可并行化阶段。',
    });
  }

  if (taskDelta.flag && taskDelta.flag !== 'pass') {
    issues.push({
      priority: taskDelta.flag === 'block' ? 'P0' : 'P1',
      title: '任务达成率出现回归',
      scope: `${fmt(ov.task_failed)} 条未达成`,
      evidence: `baseline ${fmt(taskDelta.prev)}，本次 ${fmt(taskDelta.now)}，变化 ${deltaText(taskDelta)}。`,
      action: '区分检索失败、工具失败和验证未通过，避免只按最终回答判断。',
      badcase: badcases[0],
    });
  }

  if (faithfulness !== null && faithfulness < 0.85) {
    issues.push({
      priority: 'P1',
      title: '回答忠实度低于期望阈值',
      scope: `faithfulness ${fmt(faithfulness)} · n=${fmt((report.judge as Record<string, unknown>).sample_size)}`,
      evidence: 'judge 已标记回答与检索证据不一致的样本。',
      action: '核对回答中的事实陈述与引用片段，补齐 citation 对齐。',
      badcase: badcases[0],
    });
  }

  if (issues.length === 0) {
    issues.push({
      priority: 'P1',
      title: '未发现发布阻塞项',
      scope: `${report.query_count} 条 QA`,
      evidence: '门禁和核心指标均未触发 block/warn。',
      action: '继续观察 P95、成本和 judge 样本，确认没有采样偏差。',
    });
  }

  return issues.slice(0, 4);
}

function badcaseDistribution(report: EvalRunReport): Array<{ label: string; count: number; tone: string }> {
  const counts = new Map<string, number>();
  for (const item of report.badcases ?? []) {
    const key = item.category || 'unknown';
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([label, count], index) => ({
      label,
      count,
      tone: [
        'var(--color-danger)',
        'var(--color-warning)',
        'var(--color-primary)',
        'var(--color-text-secondary)',
        'var(--color-text-tertiary)',
      ][index],
    }));
}

function DecisionMetric({
  label,
  value,
  previous,
  delta,
  threshold,
  flag,
}: {
  label: string;
  value: string;
  previous: string;
  delta: string;
  threshold: string;
  flag?: string;
}) {
  const tone = verdictTone(flag);
  return (
    <div className="eval-decision-metric">
      <div className="eval-decision-metric-head">
        <span>{label}</span>
        <span className={`eval-verdict is-${tone}`}>{verdictLabel(flag)}</span>
      </div>
      <strong>{value}</strong>
      <div className="eval-decision-metric-meta">
        <span>baseline {previous}</span>
        <b className={`eval-delta is-${tone}`}>{delta}</b>
      </div>
      <div className="eval-decision-threshold">{threshold}</div>
    </div>
  );
}

function MetricMatrix({ report }: { report: EvalRunReport }) {
  const ov = (report.overall ?? {}) as Record<string, unknown>;
  const judge = (report.judge ?? {}) as Record<string, unknown>;
  const faithfulness = metricValue(judge, 'faithfulness');
  const toolSuccess = metricValue(ov, 'tool_success_rate');
  const latencyP95 = metricValue(ov, 'answer_latency_p95_s');
  const deltas = (((report.baseline_delta ?? {}) as Record<string, unknown>).deltas ?? {}) as Record<string, unknown>;
  const costPerQuery = metricValue(ov, 'cost_usd') !== null && report.query_count > 0
    ? Number(ov.cost_usd) / report.query_count
    : null;
  const costPerTask = metricValue(ov, 'cost_per_successful_task_usd');
  const tokensPerTask = metricValue(ov, 'tokens_per_successful_task');
  const cacheHitRate = metricValue(ov, 'cache_hit_rate');

  const rows: Array<{
    group: string;
    label: string;
    value: string;
    baseline: string;
    delta: string;
    threshold: string;
    flag?: string;
  }> = [
    {
      group: '质量',
      label: 'Recall@5',
      value: fmt(ov['recall@5']),
      baseline: fmt(asDelta(deltas['recall@5']).prev),
      delta: deltaText(asDelta(deltas['recall@5'])),
      threshold: '相对下降 ≤ 3%',
      flag: asDelta(deltas['recall@5']).flag,
    },
    {
      group: '质量',
      label: 'NDCG@10',
      value: fmt(ov['ndcg@10']),
      baseline: fmt(asDelta(deltas['ndcg@10']).prev),
      delta: deltaText(asDelta(deltas['ndcg@10'])),
      threshold: '相对下降 ≤ 3%',
      flag: asDelta(deltas['ndcg@10']).flag,
    },
    {
      group: '质量',
      label: 'Faithfulness',
      value: fmt(faithfulness),
      baseline: '—',
      delta: '—',
      threshold: '期望 ≥ 0.85',
      flag: faithfulness === null ? undefined : faithfulness >= 0.85 ? 'pass' : 'warn',
    },
    {
      group: '行为',
      label: '任务成功率',
      value: fmt(ov.task_success_rate),
      baseline: fmt(asDelta(deltas.task_success_rate).prev),
      delta: deltaText(asDelta(deltas.task_success_rate)),
      threshold: '相对下降 ≤ 2%',
      flag: asDelta(deltas.task_success_rate).flag,
    },
    {
      group: '行为',
      label: '工具成功率',
      value: fmt(toolSuccess),
      baseline: fmt(asDelta(deltas.tool_success_rate).prev),
      delta: deltaText(asDelta(deltas.tool_success_rate)),
      threshold: '失败 0 次',
      flag: toolSuccess === null ? undefined : toolSuccess >= 1 ? 'pass' : 'block',
    },
    {
      group: '效率',
      label: '回答 P95',
      value: latencyP95 === null ? '—' : `${fmt(latencyP95)}s`,
      baseline: '—',
      delta: '—',
      threshold: '目标 ≤ 30s',
      flag: latencyP95 === null ? undefined : latencyP95 <= 30 ? 'pass' : 'warn',
    },
    {
      group: '效率',
      label: '单题成本',
      value: costPerQuery === null ? '—' : `$${fmt(costPerQuery)}`,
      baseline: '—',
      delta: '—',
      threshold: '按 run 预算观察',
      flag: 'pass',
    },
    {
      group: '效率',
      label: '单成功任务成本',
      value: costPerTask === null ? '—' : `$${fmt(costPerTask)}`,
      baseline: '—',
      delta: '—',
      threshold: '按 run 预算观察',
      flag: costPerTask === null ? undefined : 'pass',
    },
    {
      group: '效率',
      label: '单成功任务 Token',
      value: fmt(tokensPerTask),
      baseline: '—',
      delta: '—',
      threshold: '观察长期趋势',
      flag: tokensPerTask === null ? undefined : 'pass',
    },
    {
      group: '效率',
      label: '读缓存命中率',
      value: cacheHitRate === null ? '—' : `${fmt(Number(cacheHitRate) * 100)}%`,
      baseline: '—',
      delta: '—',
      threshold: '越高越好，需结合正确率',
      flag: cacheHitRate === null ? undefined :
        Number(cacheHitRate) >= 0.2 ? 'pass' : 'warn',
    },
  ];

  return (
    <section className="eval-overview-panel">
      <div className="eval-overview-panel-head">
        <b>核心指标矩阵</b>
        <span>同口径对比 baseline，数值均保留分子/分母语义</span>
      </div>
      <div className="eval-table-wrap">
        <table className="eval-metric-table">
          <thead>
            <tr>
              <th>指标</th>
              <th>本次</th>
              <th>baseline</th>
              <th>变化</th>
              <th>阈值</th>
              <th>状态</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(row => (
              <tr key={`${row.group}-${row.label}`}>
                <td>
                  {row.group !== rows[rows.indexOf(row) - 1]?.group && (
                    <span className="eval-table-group">{row.group}</span>
                  )}
                  <b>{row.label}</b>
                </td>
                <td>{row.value}</td>
                <td>{row.baseline}</td>
                <td>{row.delta}</td>
                <td>{row.threshold}</td>
                <td>
                  <span className={`eval-verdict is-${verdictTone(row.flag)}`}>
                    {verdictLabel(row.flag)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function conclusionMarkdown(report: EvalRunReport): string {
  const ov = (report.overall ?? {}) as Record<string, unknown>;
  const metadata = (report.metadata ?? {}) as Record<string, unknown>;
  const baseline = (report.baseline_delta ?? {}) as Record<string, unknown>;
  const issues = buildEvalIssues(report);
  const lines = [
    `# 评测发布结论`,
    '',
    `- run_id: \`${report.run_id}\``,
    `- dataset_id: \`${report.dataset_id}\``,
    `- gate: **${String(baseline.gate ?? 'unknown')}**`,
    `- status: \`${report.status}\``,
    `- query_count: ${report.query_count}`,
    `- duration_s: ${fmt(report.duration_s)}`,
    `- model: \`${String(metadata.model ?? '—')}\``,
    `- commit: \`${String(metadata.commit ?? '—')}\``,
    '',
    '## 核心指标',
    '',
    '| 指标 | 本次 | baseline | 变化 | 阈值 |',
    '|---|---:|---:|---:|---|',
    `| Recall@5 | ${fmt(ov['recall@5'])} | ${fmt(asDelta(((baseline.deltas ?? {}) as Record<string, unknown>)['recall@5']).prev)} | ${deltaText(asDelta(((baseline.deltas ?? {}) as Record<string, unknown>)['recall@5']))} | 相对下降 ≤ 3% |`,
    `| 任务成功率 | ${fmt(ov.task_success_rate)} | ${fmt(asDelta(((baseline.deltas ?? {}) as Record<string, unknown>).task_success_rate).prev)} | ${deltaText(asDelta(((baseline.deltas ?? {}) as Record<string, unknown>).task_success_rate))} | 相对下降 ≤ 2% |`,
    `| 回答 P95 | ${fmt(ov.answer_latency_p95_s)}s | — | — | 目标 ≤ 30s |`,
    `| 工具失败 | ${fmt(ov.tool_failures)} | 0 | ${(metricValue(ov, 'tool_failures') ?? 0) > 0 ? '需处理' : '无'} | 0 次 |`,
    '',
    '## 优先问题',
    '',
  ];
  for (const issue of issues) {
    lines.push(
      `### ${issue.priority} ${issue.title}`,
      '',
      `- 影响：${issue.scope}`,
      `- 证据：${issue.evidence}`,
      `- 下一步：${issue.action}`,
      '',
    );
  }
  if ((report.notes ?? []).length > 0) {
    lines.push('## 备注', '', ...report.notes.map(note => `- ${note}`), '');
  }
  return lines.join('\n');
}

function downloadEvalConclusion(report: EvalRunReport) {
  const blob = new Blob([conclusionMarkdown(report)], {
    type: 'text/markdown;charset=utf-8',
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `${report.run_id}-release-conclusion.md`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function RunOverview({
  report,
  onBadcase,
}: {
  report: EvalRunReport;
  onBadcase: (b: EvalBadcase) => void;
}) {
  const ov = (report.overall ?? {}) as Record<string, unknown>;
  const metadata = (report.metadata ?? {}) as Record<string, unknown>;
  const baseline = (report.baseline_delta ?? {}) as Record<string, unknown>;
  const deltas = (baseline.deltas ?? {}) as Record<string, unknown>;
  const issues = buildEvalIssues(report);
  const distribution = badcaseDistribution(report);
  const gate = String(baseline.gate ?? 'unknown').toLowerCase();
  const gateTone = gate === 'pass' ? 'ok' : gate === 'warn' || gate === 'unknown' ? 'warn' : 'danger';
  const badcaseCount = report.badcases?.length ?? 0;
  const toolFailures = metricValue(ov, 'tool_failures');
  const latencyP95 = metricValue(ov, 'answer_latency_p95_s');

  return (
    <div className="eval-overview">
      <section className="eval-run-header">
        <div>
          <div className="eval-run-kicker">RUN DECISION</div>
          <h2>{gate === 'pass' ? '本次改动可以进入下一阶段' : '本次改动不建议直接发布'}</h2>
          <p>
            {badcaseCount > 0
              ? `${badcaseCount} 个 badcase 已归因；先处理阻塞项，再回到样本与 trace 验证。`
              : '没有 badcase 命中；继续观察 judge 样本、成本和长尾延迟。'}
          </p>
        </div>
        <div className="eval-run-meta">
          <span>{report.query_count} QA</span>
          <span>{fmt(report.duration_s)}s</span>
          <span>{String(metadata.model ?? '—')}</span>
          <span className="mono">commit {String(metadata.commit ?? '—')}</span>
          <button
            type="button"
            className="eval-export-action"
            onClick={() => downloadEvalConclusion(report)}
          >
            导出发布结论
          </button>
        </div>
      </section>

      <section className="eval-decision-card">
        <div className={`eval-gate is-${gateTone}`}>
          <span>发布门禁</span>
          <strong>{gate === 'pass' ? '通过' : gate === 'warn' ? '需审阅' : '阻塞'}</strong>
          <p>{badcaseCount} 个 badcase · {issues.filter(item => item.priority === 'P0').length} 个 P0</p>
        </div>
        <div className="eval-decision-metrics">
          <DecisionMetric
            label="Recall@5"
            value={fmt(ov['recall@5'])}
            previous={fmt(asDelta(deltas['recall@5']).prev)}
            delta={deltaText(asDelta(deltas['recall@5']))}
            threshold="相对 baseline 下降 ≤ 3%"
            flag={asDelta(deltas['recall@5']).flag}
          />
          <DecisionMetric
            label="任务成功率"
            value={fmt(ov.task_success_rate)}
            previous={fmt(asDelta(deltas.task_success_rate).prev)}
            delta={deltaText(asDelta(deltas.task_success_rate))}
            threshold="相对 baseline 下降 ≤ 2%"
            flag={asDelta(deltas.task_success_rate).flag}
          />
          <DecisionMetric
            label="回答 P95"
            value={latencyP95 === null ? '—' : `${fmt(latencyP95)}s`}
            previous="—"
            delta="—"
            threshold="目标 ≤ 30s"
            flag={latencyP95 === null ? undefined : latencyP95 <= 30 ? 'pass' : 'warn'}
          />
          <DecisionMetric
            label="工具失败"
            value={fmt(toolFailures)}
            previous="0"
            delta={toolFailures === null ? '—' : toolFailures > 0 ? '需处理' : '无'}
            threshold="失败调用 0 次"
            flag={toolFailures === null ? undefined : toolFailures > 0 ? 'block' : 'pass'}
          />
        </div>
      </section>

      <section className="eval-section-block">
        <div className="eval-section-block-head">
          <div>
            <h3>优先处理的问题</h3>
            <p>按阻塞度排序，每条都包含影响范围、证据和下一步动作。</p>
          </div>
          <span>{issues.length} 个摘要项</span>
        </div>
        <div className="eval-issue-list">
          {issues.map((issue, index) => (
            <article className="eval-issue-row" key={`${issue.title}-${index}`}>
              <span className={`eval-priority is-${issue.priority === 'P0' ? 'danger' : 'warn'}`}>
                {issue.priority}
              </span>
              <div className="eval-issue-main">
                <b>{issue.title}</b>
                <span>{issue.scope}</span>
              </div>
              <div className="eval-issue-evidence">{issue.evidence}</div>
              <div className="eval-issue-action">
                <span>下一步</span>
                <b>{issue.action}</b>
                {issue.badcase && (
                  <button type="button" onClick={() => void onBadcase(issue.badcase!)}>
                    查看案例
                  </button>
                )}
              </div>
            </article>
          ))}
        </div>
      </section>

      <div className="eval-overview-grid">
        <MetricMatrix report={report} />
        <section className="eval-overview-panel">
          <div className="eval-overview-panel-head">
            <b>Badcase 分布</b>
            <span>按首个失败原因归因</span>
          </div>
          <div className="eval-distribution">
            {distribution.length === 0 && (
              <div className="eval-empty-inline">未记录 badcase，或当前报告尚未完成归因。</div>
            )}
            {distribution.map(item => {
              const max = distribution[0]?.count || 1;
              return (
                <div className="eval-distribution-row" key={item.label}>
                  <span>{item.label}</span>
                  <div className="eval-distribution-track">
                    <i style={{ width: `${Math.max(8, (item.count / max) * 100)}%`, background: item.tone }} />
                  </div>
                  <b>{item.count}</b>
                </div>
              );
            })}
          </div>
          <div className="eval-overview-note">
            先把高影响、可复现的问题送进回归集，再处理单点低收益优化。
          </div>
        </section>
      </div>
    </div>
  );
}

function RunCases({
  report,
  traceEvents,
  onBadcase,
}: {
  report: EvalRunReport;
  traceEvents: EvalTraceEvent[] | null;
  onBadcase: (b: EvalBadcase) => void;
}) {
  const badcases = report.badcases ?? [];
  const [activeCase, setActiveCase] = useState('');
  const [caseAction, setCaseAction] = useState('');
  const [caseBusy, setCaseBusy] = useState('');
  const activeBadcase = badcases.find((item, index) =>
    (item.trace_id || item.query_id || String(index)) === activeCase);

  const addToRegression = async () => {
    if (!activeBadcase?.trace_id) {
      setCaseAction('该案例没有 trace_id，无法回流。');
      return;
    }
    setCaseBusy('regression');
    setCaseAction('');
    try {
      const result = await api.addEvalRegressionCase(activeBadcase.trace_id);
      setCaseAction(`已写入回归候选：${result.qa_id}`);
    } catch (error) {
      setCaseAction(`加入回归集失败：${String(error)}`);
    } finally {
      setCaseBusy('');
    }
  };

  const markHandled = async () => {
    if (!activeBadcase?.trace_id) {
      setCaseAction('该案例没有 trace_id，无法回写 LangSmith。');
      return;
    }
    setCaseBusy('handled');
    setCaseAction('');
    try {
      await api.submitEvalFeedback({
        trace_id: activeBadcase.trace_id,
        key: 'review_status',
        score: 1,
        comment: 'marked handled from evaluation center',
        run_id: report.run_id,
      });
      setCaseAction('已标记为处理完成。');
    } catch (error) {
      setCaseAction(`标记失败：${String(error)}`);
    } finally {
      setCaseBusy('');
    }
  };

  return (
    <div className="eval-case-layout">
      <section className="eval-case-list">
        <div className="eval-section-block-head">
          <div>
            <h3>Badcase</h3>
            <p>{badcases.length} 条失败或降级案例，点击后加载 trace。</p>
          </div>
          <span>{report.query_count} QA</span>
        </div>
        {badcases.length === 0 && (
          <div className="eval-empty-inline">本 run 没有达到 badcase 阈值的样本。</div>
        )}
        {badcases.map((item, index) => {
          const id = item.trace_id || item.query_id || String(index);
          return (
            <button
              type="button"
              key={`${id}-${index}`}
              className={`eval-case-row ${activeCase === id ? 'is-active' : ''}`}
              onClick={() => {
                setActiveCase(id);
                setCaseAction('');
                void onBadcase(item);
              }}
            >
              <span className="eval-case-query">{item.query}</span>
              <span className="eval-case-meta">
                <span className="eval-verdict is-danger">{item.category || 'unknown'}</span>
                <span>MRR {fmt(item.mrr)}</span>
                <span>{fmt(item.duration_s)}s</span>
              </span>
              <span className="eval-case-code">
                {item.query_id || '—'} · {item.trace_id || 'no trace'}
              </span>
            </button>
          );
        })}
      </section>

      <section className="eval-evidence-panel">
        <div className="eval-evidence-head">
          <div>
            <span>TRACE EVIDENCE</span>
            <h3>{traceEvents ? `已加载 ${traceEvents.length} 个事件` : '选择一个 badcase 查看证据链'}</h3>
          </div>
          {activeBadcase && (
            <div className="eval-evidence-actions">
              <button
                type="button"
                className="eval-secondary-action"
                disabled={caseBusy !== ''}
                onClick={() => void addToRegression()}
              >
                {caseBusy === 'regression' ? '写入中…' : '加入回归集'}
              </button>
              <button
                type="button"
                className="eval-primary-action"
                disabled={caseBusy !== ''}
                onClick={() => void markHandled()}
              >
                {caseBusy === 'handled' ? '提交中…' : '标记已处理'}
              </button>
            </div>
          )}
        </div>
        {caseAction && <div className="eval-case-action-message">{caseAction}</div>}
        {traceEvents ? (
          <div className="eval-trace-events">
            <MessageSteps steps={eventsToSteps(traceEvents)} defaultExpanded />
          </div>
        ) : (
          <div className="eval-evidence-empty">
            <b>证据将在选择案例后出现在这里</b>
            <span>保留工具入参、结果信封、错误、耗时和原始 trace_id。</span>
          </div>
        )}
      </section>
    </div>
  );
}

function RunTrace({
  report,
  traceEvents,
  onBadcase,
}: {
  report: EvalRunReport;
  traceEvents: EvalTraceEvent[] | null;
  onBadcase: (b: EvalBadcase) => void;
}) {
  return (
    <div className="eval-trace-layout">
      <section className="eval-overview-panel">
        <div className="eval-overview-panel-head">
          <b>可追溯样本</b>
          <span>选择后查看原始原子事件</span>
        </div>
        <div className="eval-trace-picker">
          {(report.badcases ?? []).map((item, index) => (
            <button
              type="button"
              key={`${item.trace_id || item.query_id || index}`}
              onClick={() => void onBadcase(item)}
            >
              <span>{item.query}</span>
              <code>{item.trace_id || 'no trace'}</code>
            </button>
          ))}
          {(report.badcases ?? []).length === 0 && (
            <div className="eval-empty-inline">没有 badcase 可追溯到 trace。</div>
          )}
        </div>
      </section>
      <section className="eval-overview-panel">
        <div className="eval-overview-panel-head">
          <b>原始事件</b>
          <span>{traceEvents ? `${traceEvents.length} 个事件` : '等待选择'}</span>
        </div>
        <div className="eval-trace-events">
          {traceEvents ? (
            <MessageSteps steps={eventsToSteps(traceEvents)} defaultExpanded />
          ) : (
            <div className="eval-empty-inline">从左侧选择一条 badcase。</div>
          )}
        </div>
      </section>
    </div>
  );
}

interface CompareMetricSpec {
  key: string;
  label: string;
  unit?: string;
  group: string;
  better: 'higher' | 'lower';
}

const COMPARE_METRICS: CompareMetricSpec[] = [
  { key: 'recall@5', label: 'Recall@5', group: '质量', better: 'higher' },
  { key: 'ndcg@10', label: 'NDCG@10', group: '质量', better: 'higher' },
  { key: 'mrr', label: 'MRR', group: '质量', better: 'higher' },
  { key: 'task_success_rate', label: '任务成功率', group: '行为', better: 'higher' },
  { key: 'tool_success_rate', label: '工具成功率', group: '行为', better: 'higher' },
  { key: 'answer_latency_p95_s', label: '回答 P95', unit: 's', group: '效率', better: 'lower' },
  { key: 'cost_usd', label: '总成本', unit: '$', group: '效率', better: 'lower' },
  { key: 'tokens_total', label: 'Token', group: '效率', better: 'lower' },
  { key: 'cost_per_successful_task_usd', label: '单成功任务成本', unit: '$', group: '效率', better: 'lower' },
  { key: 'tokens_per_successful_task', label: '单成功任务 Token', group: '效率', better: 'lower' },
  { key: 'cache_hit_rate', label: '读缓存命中率', group: '效率', better: 'higher' },
];

function confidenceInterval(report: EvalRunReport, key: string): {
  low: number;
  high: number;
} | null {
  const ov = (report.overall ?? {}) as Record<string, unknown>;
  const intervals = (ov.confidence_intervals ?? {}) as Record<string, unknown>;
  const item = (intervals[key] ?? {}) as Record<string, unknown>;
  const low = Number(item.ci95_low);
  const high = Number(item.ci95_high);
  return Number.isFinite(low) && Number.isFinite(high) ? { low, high } : null;
}

function CompareMetricCard({
  spec,
  current,
  baseline,
}: {
  spec: CompareMetricSpec;
  current: EvalRunReport;
  baseline: EvalRunReport;
}) {
  const currentOv = (current.overall ?? {}) as Record<string, unknown>;
  const baselineOv = (baseline.overall ?? {}) as Record<string, unknown>;
  const currentValue = metricValue(currentOv, spec.key);
  const baselineValue = metricValue(baselineOv, spec.key);
  if (currentValue === null || baselineValue === null) {
    return (
      <article className="eval-compare-metric is-muted">
        <div className="eval-compare-metric-head">
          <span>{spec.group}</span>
          <b>{spec.label}</b>
        </div>
        <div className="eval-compare-empty">缺少可比较数据</div>
      </article>
    );
  }
  const maxValue = Math.max(
    Math.abs(currentValue),
    Math.abs(baselineValue),
    spec.better === 'higher' ? 1 : 0.0001,
  );
  const currentWidth = Math.max(2, (Math.abs(currentValue) / maxValue) * 100);
  const baselineWidth = Math.max(2, (Math.abs(baselineValue) / maxValue) * 100);
  const delta = baselineValue === 0
    ? null
    : ((currentValue - baselineValue) / Math.abs(baselineValue)) * 100;
  const improved = delta === null
    ? null
    : spec.better === 'higher' ? delta >= 0 : delta <= 0;
  const currentCi = confidenceInterval(current, spec.key);
  const baselineCi = confidenceInterval(baseline, spec.key);
  const format = (value: number) => {
    if (spec.unit === '$') return `$${fmt(value)}`;
    return `${fmt(value)}${spec.unit ?? ''}`;
  };

  return (
    <article className="eval-compare-metric">
      <div className="eval-compare-metric-head">
        <span>{spec.group}</span>
        <b>{spec.label}</b>
        <em className={improved === null ? 'is-muted' : improved ? 'is-ok' : 'is-danger'}>
          {delta === null ? '—' : `${delta > 0 ? '+' : ''}${fmt(delta)}%`}
        </em>
      </div>
      <div className="eval-compare-bars">
        <div className="eval-compare-bar-row">
          <span>current</span>
          <div className="eval-compare-track">
            <i style={{ width: `${currentWidth}%` }} />
            {currentCi && (
              <b
                title={`95% CI ${fmt(currentCi.low)}–${fmt(currentCi.high)}`}
                style={{
                  left: `${(currentCi.low / maxValue) * 100}%`,
                  width: `${Math.max(1, ((currentCi.high - currentCi.low) / maxValue) * 100)}%`,
                }}
              />
            )}
          </div>
          <strong>{format(currentValue)}</strong>
        </div>
        <div className="eval-compare-bar-row is-baseline">
          <span>baseline</span>
          <div className="eval-compare-track">
            <i style={{ width: `${baselineWidth}%` }} />
            {baselineCi && (
              <b
                title={`95% CI ${fmt(baselineCi.low)}–${fmt(baselineCi.high)}`}
                style={{
                  left: `${(baselineCi.low / maxValue) * 100}%`,
                  width: `${Math.max(1, ((baselineCi.high - baselineCi.low) / maxValue) * 100)}%`,
                }}
              />
            )}
          </div>
          <strong>{format(baselineValue)}</strong>
        </div>
      </div>
      <div className="eval-compare-ci">
        {currentCi
          ? `current 95% CI ${fmt(currentCi.low)}–${fmt(currentCi.high)}`
          : '当前报告未包含该指标 95% CI'}
      </div>
    </article>
  );
}

function RunCompare({
  report,
  runs,
}: {
  report: EvalRunReport;
  runs: EvalRunsResp['runs'];
}) {
  const candidates = runs.filter(item => item.run_id !== report.run_id);
  const [baselineId, setBaselineId] = useState(
    () => candidates.find(item => item.status === 'done')?.run_id
      ?? candidates[0]?.run_id
      ?? '',
  );
  const [baseline, setBaseline] = useState<EvalRunReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!baselineId) {
      setBaseline(null);
      return;
    }
    let alive = true;
    setLoading(true);
    setError('');
    void api.getEvalRun(baselineId)
      .then(data => {
        if (alive) setBaseline(data);
      })
      .catch(reason => {
        if (alive) setError(`加载 baseline 失败：${String(reason)}`);
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => { alive = false; };
  }, [baselineId]);

  if (candidates.length === 0) {
    return (
      <section className="eval-overview-panel">
        <div className="eval-empty-inline">至少需要两个 run 才能进行对比。</div>
      </section>
    );
  }

  const currentMetadata = (report.metadata ?? {}) as Record<string, unknown>;
  const baselineMetadata = (baseline?.metadata ?? {}) as Record<string, unknown>;
  const configRows: Array<[string, string, string]> = baseline ? [
    ['dataset_id', baseline.dataset_id, report.dataset_id],
    ['model', String(baselineMetadata.model ?? '—'), String(currentMetadata.model ?? '—')],
    ['commit', String(baselineMetadata.commit ?? '—'), String(currentMetadata.commit ?? '—')],
    ['config_fingerprint', String(baselineMetadata.config_fingerprint ?? '—'), String(currentMetadata.config_fingerprint ?? '—')],
    ['tools_fingerprint', String(baselineMetadata.tools_fingerprint ?? '—'), String(currentMetadata.tools_fingerprint ?? '—')],
    ['query_count', String(baseline.query_count), String(report.query_count)],
  ] : [];
  const datasetComparable = baseline?.dataset_id === report.dataset_id;
  const modelComparable = String(baselineMetadata.model ?? '') === String(currentMetadata.model ?? '');

  return (
    <div className="eval-compare-layout">
      <section className="eval-compare-toolbar">
        <div>
          <span>比较对象</span>
          <b>{report.run_id}</b>
          <span>vs</span>
          <select
            value={baselineId}
            onChange={event => setBaselineId(event.target.value)}
            aria-label="选择 baseline run"
          >
            {candidates.map(item => (
              <option key={item.run_id} value={item.run_id}>
                {item.run_id} · {item.dataset_id} · {item.status}
              </option>
            ))}
          </select>
          {loading && <i className="step-spinner" aria-label="加载中" />}
        </div>
      </section>

      {error && <div className="eval-compare-warning is-danger">{error}</div>}
      {!loading && !baseline && !error && (
        <div className="eval-empty-inline">选择 baseline 后加载配置与指标。</div>
      )}

      {baseline && (
        <>
          {!datasetComparable && (
            <div className="eval-compare-warning is-danger">
              数据集不同（baseline {baseline.dataset_id} / current {report.dataset_id}），
              指标差异包含数据分布变化，不能作为纯回归结论。
            </div>
          )}
          {datasetComparable && !modelComparable && (
            <div className="eval-compare-warning is-warn">
              模型不同（baseline {String(baselineMetadata.model ?? '—')} /
              current {String(currentMetadata.model ?? '—')}），对比结论应同时归因模型变化。
            </div>
          )}
          {datasetComparable && modelComparable && (
            <div className="eval-compare-warning is-ok">
              数据集与模型一致，可按相对 baseline 的变化判断回归。
            </div>
          )}

          <section className="eval-overview-panel">
            <div className="eval-overview-panel-head">
              <b>配置差异</b>
              <span>先判断可比性，再看指标</span>
            </div>
            <div className="eval-table-wrap">
              <table className="eval-compare-config-table">
                <thead>
                  <tr><th>配置</th><th>baseline</th><th>current</th><th>状态</th></tr>
                </thead>
                <tbody>
                  {configRows.map(([label, before, now]) => (
                    <tr key={label}>
                      <td>{label}</td>
                      <td><code>{before}</code></td>
                      <td><code>{now}</code></td>
                      <td>
                        <span className={`eval-verdict ${before === now ? 'is-ok' : 'is-warn'}`}>
                          {before === now ? '一致' : '变化'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <div className="eval-compare-grid">
            {COMPARE_METRICS.map(spec => (
              <CompareMetricCard
                key={spec.key}
                spec={spec}
                current={report}
                baseline={baseline}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function RunConfig({ report }: { report: EvalRunReport }) {
  const metadata = (report.metadata ?? {}) as Record<string, unknown>;
  const overall = (report.overall ?? {}) as Record<string, unknown>;
  const baseline = (report.baseline_delta ?? {}) as Record<string, unknown>;
  const rows: Array<[string, string]> = [
    ['run_id', report.run_id],
    ['dataset_id', report.dataset_id],
    ['status', report.status || '—'],
    ['query_count', String(report.query_count)],
    ['model', String(metadata.model ?? '—')],
    ['commit', String(metadata.commit ?? '—')],
    ['config_fingerprint', String(metadata.config_fingerprint ?? '—')],
    ['tools_fingerprint', String(metadata.tools_fingerprint ?? '—')],
    ['tokens_total', String(overall.tokens_total ?? '—')],
    ['cost_usd', String(overall.cost_usd ?? '—')],
    ['gate', String(baseline.gate ?? '—')],
    ['started_at', String(report.started_at ?? '—')],
    ['finished_at', String(report.finished_at ?? '—')],
  ];

  return (
    <section className="eval-overview-panel">
      <div className="eval-overview-panel-head">
        <b>本次运行冻结配置</b>
        <span>结论必须和配置、数据集及 gate 一起审阅</span>
      </div>
      <div className="eval-config-grid">
        {rows.map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <code>{value}</code>
          </div>
        ))}
      </div>
      {report.notes?.length > 0 && (
        <div className="eval-overview-note">{report.notes.join(' · ')}</div>
      )}
    </section>
  );
}
