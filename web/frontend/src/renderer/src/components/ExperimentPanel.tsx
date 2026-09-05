/**
 * ExperimentPanel.tsx — 右侧工作台「实验」面板（对话中心化 L2）。
 *
 * 聚焦当前对话绑定的实验项目（thread.project，SSE 归因）：
 * 项目下拉（绑定优先）→ manifest 摘要（entry.run / paper / 委托契约）→ Run 表单
 * → 运行列表（status/指标）→ 详情（metrics + 日志尾部）。3s 轮询活动项目。
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { api, whenBackendReady, type Experiment, type ProjectManifest } from '../api';
import { MetricSparkline, STATUS_LABEL, colTitle, primaryBtn, statusStyle } from './ExperimentView';

const btnStyle: React.CSSProperties = {
  padding: '4px 10px', borderRadius: 6, fontSize: 13,
  border: '1px solid var(--color-border)', background: 'var(--color-surface)',
  display: 'inline-flex', alignItems: 'center', gap: 4,
};

interface Props {
  /** 当前对话绑定实验项目（SSE 归因）；空 → 回落列表视图。 */
  project?: string;
}

export function ExperimentPanel({ project }: Props) {
  const [projects, setProjects] = useState<string[]>([]);
  const [activeProject, setActiveProject] = useState<string>(project ?? '');
  const [manifest, setManifest] = useState<ProjectManifest | null>(null);
  const [exps, setExps] = useState<Experiment[]>([]);
  const [expId, setExpId] = useState<string | null>(null);
  const [exp, setExp] = useState<Experiment | null>(null);
  const [showRun, setShowRun] = useState(false);
  const [cmd, setCmd] = useState('');
  const [runName, setRunName] = useState('');
  const [error, setError] = useState('');
  const logRef = useRef<HTMLPreElement | null>(null);

  // 跟随对话绑定：project prop 变化 → 聚焦该项目（并自动视角切到列表新实验）
  useEffect(() => {
    if (project && project !== activeProject) {
      setActiveProject(project);
      setManifest(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project]);

  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight);
  }, [exp?.log_tail]);

  const loadManifest = useCallback(async (proj: string) => {
    try {
      const r = await api.getProjectManifest(proj);
      setManifest(r.manifest);
    } catch { setManifest(null); }
  }, []);

  const refreshProjects = useCallback(async () => {
    await whenBackendReady();
    try {
      const r = await api.listExperimentProjects();
      setProjects(r.projects);
      setActiveProject(p => (r.projects.includes(p) ? p : (project ?? p)));
    } catch { setError('实验服务不可用'); }
  }, [project]);

  useEffect(() => { void refreshProjects(); }, [refreshProjects]);

  useEffect(() => {
    if (!activeProject) { setExps([]); setExpId(null); setManifest(null); return; }
    void loadManifest(activeProject);
    void (async () => {
      try {
        const r = await api.listExperiments(activeProject);
        setExps(r.experiments);
        const first = r.experiments[0]?.exp_id ?? null;
        setExpId(e => (e && r.experiments.some(x => x.exp_id === e) ? e : first));
      } catch { setExps([]); }
    })();
  }, [activeProject, loadManifest]);

  useEffect(() => {
    if (!expId) { setExp(null); return; }
    void api.getExperiment(expId).then(setExp).catch(() => setExp(null));
  }, [expId]);

  // 3s 轮询：列表 + 活动实验详情 + manifest
  useEffect(() => {
    if (!activeProject) return;
    const timer = setInterval(() => {
      void api.listExperiments(activeProject).then(r => setExps(r.experiments)).catch(() => {});
      if (expId) void api.getExperiment(expId).then(setExp).catch(() => {});
      void loadManifest(activeProject);
    }, 3000);
    return () => clearInterval(timer);
  }, [activeProject, expId, loadManifest]);

  const handleRun = useCallback(async () => {
    const command = cmd.trim();
    if (!command || !activeProject) return;
    setError('');
    try {
      const r = await api.runExperiment(activeProject, command, runName.trim());
      setCmd(''); setRunName(''); setShowRun(false);
      const list = await api.listExperiments(activeProject);
      setExps(list.experiments);
      setExpId(r.exp_id);
    } catch { setError('启动实验失败'); }
  }, [cmd, runName, activeProject]);

  const expsList = Array.isArray(exps) ? exps : [];
  const activeExp = expsList.find(e => e.exp_id === expId) ?? exp;
  const mf = manifest;

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* 头：项目选择 + Run */}
      <div style={colTitle}>🧪 实验{project ? ' · 对话绑定' : ''}</div>
      <div style={{ padding: '6px 10px', borderBottom: '1px solid var(--color-border)', display: 'flex', gap: 6, alignItems: 'center' }}>
        <select
          value={activeProject}
          onChange={e => setActiveProject(e.target.value)}
          style={{ flex: 1, padding: '4px 8px', fontSize: 13, borderRadius: 6, border: '1px solid var(--color-border)' }}
        >
          <option value="">（未绑定项目）</option>
          {projects.map(p => <option key={p} value={p}>{p}</option>)}
        </select>
        <button style={btnStyle} onClick={() => void refreshProjects()} title="刷新">⟳</button>
        <button style={primaryBtn} onClick={() => setShowRun(s => !s)}>▶ Run</button>
      </div>
      {error && <div style={{ padding: '4px 12px', fontSize: 12, color: 'var(--color-danger)', background: '#fdecea' }}>{error}</div>}

      {showRun && (
        <div style={{ display: 'flex', gap: 6, padding: '6px 10px', borderBottom: '1px solid var(--color-border)', background: 'var(--color-inset)' }}>
          <input
            value={runName} onChange={e => setRunName(e.target.value)}
            placeholder="实验名（可选）" style={{ width: 110, padding: '4px 8px', fontSize: 13 }}
          />
          <input
            value={cmd} onChange={e => setCmd(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') void handleRun(); }}
            placeholder={activeProject ? `命令…（在 ${activeProject}/ 下）` : '命令…（自动创建项目）'}
            style={{ flex: 1, padding: '4px 8px', fontSize: 12, fontFamily: 'var(--font-mono)' }}
          />
          <button style={primaryBtn} onClick={() => void handleRun()} disabled={!cmd.trim()}>启动</button>
        </div>
      )}

      {/* manifest 摘要（委托契约条目） */}
      {mf && (mf.entry?.run || mf.paper || mf.description) && (
        <div style={{ padding: '6px 10px', borderBottom: '1px solid var(--color-border)', background: 'var(--color-inset)', fontSize: 12 }}>
          {mf.paper && <div><b>论文</b> {mf.paper}</div>}
          {mf.entry?.run && <div style={{ fontFamily: 'var(--font-mono)' }}>🖥 {mf.entry.run}</div>}
          {mf.status && <div><span style={statusStyle(mf.status)}>{STATUS_LABEL[mf.status] ?? mf.status}</span>{mf.last_commit_sha ? ` · @${mf.last_commit_sha}` : ''}</div>}
          {mf.description && <div style={{ color: 'var(--color-text-secondary)' }}>{mf.description}</div>}
        </div>
      )}

      {/* 运行列表 */}
      <div style={{ ...colTitle, fontSize: 11 }}>实验（{expsList.length}）</div>
      <div style={{ flexShrink: 0, maxHeight: '30%', overflow: 'auto', padding: 6 }}>
        {expsList.length === 0 ? (
          <div style={{ padding: 10, fontSize: 12, color: 'var(--color-text-tertiary)' }}>
            暂无实验。在对话里让 agent 复现/优化，或直接 Run 一条命令。
          </div>
        ) : expsList.map(e => (
          <div
            key={e.exp_id}
            onClick={() => setExpId(e.exp_id)}
            style={{
              padding: '6px 8px', borderRadius: 6, cursor: 'pointer', marginBottom: 3, fontSize: 12,
              background: e.exp_id === expId ? 'var(--color-primary-light)' : 'transparent',
              border: '1px solid transparent',
              borderColor: e.exp_id === expId ? 'var(--color-primary)' : 'transparent',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={statusStyle(e.status)}>{STATUS_LABEL[e.status] ?? e.status}</span>
              <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.name || e.exp_id}</span>
              <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>{e.created_at?.slice(5, 16) ?? ''}</span>
            </div>
            <div style={{ marginTop: 2, fontSize: 11, color: 'var(--color-text-tertiary)', display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {Object.entries(e.metrics || {}).slice(0, 2).map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(4) : v}`).join(' ') || '—'}
            </div>
          </div>
        ))}
      </div>

      {/* 详情：指标 + 日志 */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ ...colTitle, fontSize: 11 }}>详情{activeExp ? ` · ${activeExp.exp_id}` : ''}</div>
        {activeExp ? (
          <>
            <div style={{ flexShrink: 0, maxHeight: '35%', overflow: 'auto', padding: 6, borderBottom: '1px solid var(--color-border)' }}>
              {Object.keys(activeExp.metrics || {}).length === 0 ? (
                <div style={{ fontSize: 12, color: 'var(--color-text-tertiary)' }}>暂无指标。命令写 metrics.json/csv 后自动解析。</div>
              ) : (
                Object.entries(activeExp.metrics).map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 12, padding: '2px 0' }}>
                    <span style={{ color: 'var(--color-text-secondary)' }}>{k}</span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      {Array.isArray(v) && v.length > 1 && v.every(x => typeof x === 'number')
                        ? <MetricSparkline values={v as number[]} />
                        : <span style={{ fontFamily: 'var(--font-mono)' }}>{typeof v === 'number' ? v.toFixed(6) : String(v)}</span>}
                    </span>
                  </div>
                ))
              )}
            </div>
            <pre
              ref={logRef}
              style={{
                flex: 1, overflow: 'auto', margin: 0, padding: 8,
                fontSize: 11, lineHeight: 1.5, fontFamily: 'var(--font-mono)',
                color: 'var(--color-text)', whiteSpace: 'pre-wrap', wordBreak: 'break-all',
              }}
            >
              {activeExp.log_tail || ''}
            </pre>
          </>
        ) : (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-text-tertiary)', fontSize: 12 }}>
            ← 选择或启动一个实验
          </div>
        )}
      </div>
    </div>
  );
}