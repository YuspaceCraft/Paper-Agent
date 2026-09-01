/**
 * ExperimentView.tsx — 实验工作区（v10 / Phase D）。
 *
 * 顶部：项目选择 + Run Experiment + git 只读面板切换。
 * 左栏：实验列表（status 徽章 + 指标摘要 + git_sha）。右栏：实验详情
 * （指标表格 + 数值时序 sparkline + 日志尾部）。数据来自 /api/experiments/*。
 * 3s 轮询活动项目 → 实验卡/指标实时滚动（后台 run_experiment 驱动）。
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { api, whenBackendReady, type Experiment } from '../api';

const toolbarStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 10,
  padding: '8px 16px', flexShrink: 0,
  borderBottom: '1px solid var(--color-border)',
  background: 'var(--color-surface)',
};
const btnStyle: React.CSSProperties = {
  padding: '4px 10px', borderRadius: 6, fontSize: 13,
  border: '1px solid var(--color-border)', background: 'var(--color-surface)',
  display: 'inline-flex', alignItems: 'center', gap: 4,
};
const primaryBtn: React.CSSProperties = {
  ...btnStyle, background: 'var(--color-primary)',
  borderColor: 'var(--color-primary)', color: '#fff',
};
const colTitle: React.CSSProperties = {
  padding: '8px 12px', fontSize: 12, fontWeight: 600,
  color: 'var(--color-text-secondary)', flexShrink: 0,
  borderBottom: '1px solid var(--color-border)', background: 'var(--color-inset)',
};

const statusStyle = (s: string): React.CSSProperties => ({
  fontSize: 11, padding: '1px 8px', borderRadius: 10, flexShrink: 0,
  color: s === 'done' ? 'var(--color-success)' : s === 'running' || s === 'pending' ? 'var(--color-warning)' : 'var(--color-danger)',
  background: s === 'done' ? 'rgba(74,190,110,0.15)' : s === 'running' || s === 'pending' ? 'rgba(245,158,11,0.15)' : 'rgba(239,68,68,0.1)',
});

const STATUS_LABEL: Record<string, string> = {
  pending: '⏳ 待启动', running: '▶ 运行中', done: '✓ 完成', failed: '✗ 失败', unknown: '—',
};

function MetricSparkline({ values }: { values: number[] }) {
  /** 数值序列 → 手写 SVG polyline（主题绿，无外部图表库）。 */
  if (values.length < 2) return null;
  const w = 120, h = 28;
  const min = Math.min(...values), max = Math.max(...values);
  const span = max - min || 1;
  const pts = values.map((v, i) =>
    `${(i / (values.length - 1)) * w},${h - 3 - ((v - min) / span) * (h - 6)}`).join(' ');
  return (
    <svg width={w} height={h} style={{ display: 'block', marginTop: 4 }}>
      <polyline points={pts} fill="none" stroke="var(--color-primary)" strokeWidth={1.5} />
    </svg>
  );
}

export function ExperimentView() {
  const [projects, setProjects] = useState<string[]>([]);
  const [project, setProject] = useState('');
  const [exps, setExps] = useState<Experiment[]>([]);
  const [expId, setExpId] = useState<string | null>(null);
  const [exp, setExp] = useState<Experiment | null>(null);

  const [showRun, setShowRun] = useState(false);
  const [cmd, setCmd] = useState('');
  const [runName, setRunName] = useState('');
  const [busy, setBusy] = useState(false);

  const [gitOpen, setGitOpen] = useState(false);
  const [gitKind, setGitKind] = useState<'diff' | 'log' | 'status'>('diff');
  const [gitOut, setGitOut] = useState('');
  const [error, setError] = useState('');

  const logRef = useRef<HTMLPreElement | null>(null);

  // 自动滚动日志尾部
  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight);
  }, [exp?.log_tail]);

  // 加载项目列表
  const refreshProjects = useCallback(async () => {
    await whenBackendReady();  // Electron dev：renderer 先于后端就绪，首请求会 proxy ECONNREFUSED
    try {
      const r = await api.listExperimentProjects();
      setProjects(r.projects);
      setProject(p => (r.projects.includes(p) ? p : (r.projects[0] ?? '')));
    } catch { setError('实验服务不可用'); }
  }, []);

  useEffect(() => { void refreshProjects(); }, [refreshProjects]);

  // 项目变化 → 加载实验 + git
  useEffect(() => {
    if (!project) { setExps([]); setExpId(null); return; }
    void (async () => {
      try {
        const r = await api.listExperiments(project);
        setExps(r.experiments);
        setExpId(e => (e && r.experiments.some(x => x.exp_id === e) ? e : (r.experiments[0]?.exp_id ?? null)));
      } catch { setExps([]); }
    })();
  }, [project]);

  // 活动实验详情
  useEffect(() => {
    if (!expId) { setExp(null); return; }
    void api.getExperiment(expId).then(setExp).catch(() => setExp(null));
  }, [expId]);

  // 3s 轮询：实验列表状态/指标 + 活动实验详情滚动
  useEffect(() => {
    if (!project) return;
    const timer = setInterval(() => {
      void api.listExperiments(project).then(setExps).catch(() => {});
      if (expId) void api.getExperiment(expId).then(setExp).catch(() => {});
      if (gitOpen) void refreshGit();
    }, 3000);
    return () => clearInterval(timer);
  }, [project, expId, gitOpen]);

  const refreshGit = useCallback(async () => {
    if (!project) return;
    try { const r = await api.getProjectGit(project, gitKind); setGitOut(r.output); }
    catch (e) { setGitOut(String(e)); }
  }, [project, gitKind]);

  useEffect(() => { if (gitOpen) void refreshGit(); }, [gitOpen, refreshGit]);

  const handleRun = useCallback(async () => {
    const command = cmd.trim();
    if (!command || !project) return;
    setBusy(true); setError('');
    try {
      const r = await api.runExperiment(project, command, runName.trim());
      setCmd(''); setRunName(''); setShowRun(false);
      const list = await api.listExperiments(project);
      setExps(list.experiments);
      setExpId(r.exp_id);
    } catch { setError('启动实验失败'); }
    finally { setBusy(false); }
  }, [cmd, runName, project]);

  const activeExp = exps.find(e => e.exp_id === expId) ?? exp;

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* 工具条 */}
      <div style={toolbarStyle}>
        <span style={{ fontWeight: 700, fontSize: 14 }}>🧪 实验</span>
        <select
          value={project}
          onChange={e => setProject(e.target.value)}
          style={{ padding: '4px 8px', fontSize: 13, borderRadius: 6, border: '1px solid var(--color-border)' }}
        >
          {!project && <option value="">（无项目，输入命令即创建）</option>}
          {projects.map(p => <option key={p} value={p}>{p}</option>)}
        </select>
        <button style={btnStyle} onClick={() => void refreshProjects()} title="刷新项目">⟳</button>
        <button style={primaryBtn} onClick={() => setShowRun(s => !s)}>▶ Run Experiment</button>
        <div style={{ flex: 1 }} />
        <button style={{ ...btnStyle, color: gitOpen && gitKind === 'diff' ? 'var(--color-primary)' : undefined }}
          onClick={() => { setGitOpen(true); setGitKind('diff'); }}>git diff</button>
        <button style={{ ...btnStyle, color: gitOpen && gitKind === 'log' ? 'var(--color-primary)' : undefined }}
          onClick={() => { setGitOpen(true); setGitKind('log'); }}>git log</button>
        <button style={{ ...btnStyle, color: gitOpen && gitKind === 'status' ? 'var(--color-primary)' : undefined }}
          onClick={() => { setGitOpen(true); setGitKind('status'); }}>git status</button>
        {gitOpen && <button style={btnStyle} onClick={() => setGitOpen(false)}>✕</button>}
      </div>

      {error && <div style={{ padding: '4px 16px', fontSize: 12, color: 'var(--color-danger)', background: '#fdecea' }}>{error}</div>}

      {/* Run 表单 */}
      {showRun && (
        <div style={{ display: 'flex', gap: 8, padding: '8px 16px', borderBottom: '1px solid var(--color-border)', background: 'var(--color-inset)' }}>
          <input
            value={runName} onChange={e => setRunName(e.target.value)}
            placeholder="实验名（可选）" style={{ width: 180, padding: '4px 10px', fontSize: 13 }}
          />
          <input
            value={cmd} onChange={e => setCmd(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') void handleRun(); }}
            placeholder={project ? `命令，如: python train.py（在 experiments/${project}/ 下执行）` : '输入命令将自动创建项目目录，如: python train.py'}
            style={{ flex: 1, padding: '4px 10px', fontSize: 13, fontFamily: 'var(--font-mono)' }}
          />
          <button style={primaryBtn} onClick={() => void handleRun()} disabled={busy || !cmd.trim()}>
            {busy ? '启动中…' : '启动'}
          </button>
          <button style={btnStyle} onClick={() => setShowRun(false)}>取消</button>
        </div>
      )}

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* 左：实验列表 */}
        <div style={{ display: 'flex', flexDirection: 'column', width: 250, flexShrink: 0, borderRight: '1px solid var(--color-border)', overflow: 'hidden' }}>
          <div style={colTitle}>实验（{exps.length}）</div>
          <div style={{ flex: 1, overflow: 'auto', padding: 6 }}>
            {!project || exps.length === 0 ? (
              <div style={{ padding: 12, fontSize: 12, color: 'var(--color-text-tertiary)' }}>
                暂无实验。选择项目或用「▶ Run」输入命令启动一个后台实验。
              </div>
            ) : exps.map(e => (
              <div
                key={e.exp_id}
                onClick={() => setExpId(e.exp_id)}
                style={{
                  padding: '8px 10px', borderRadius: 6, cursor: 'pointer', marginBottom: 4,
                  background: e.exp_id === expId ? 'var(--color-primary-light)' : 'transparent',
                  border: '1px solid transparent',
                  borderColor: e.exp_id === expId ? 'var(--color-primary)' : 'transparent',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={statusStyle(e.status)}>{STATUS_LABEL[e.status] ?? e.status}</span>
                  <span style={{ fontSize: 13, fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {e.name || e.exp_id}
                  </span>
                </div>
                <div style={{ marginTop: 3, fontSize: 11, color: 'var(--color-text-tertiary)', display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <span>{Object.entries(e.metrics || {}).slice(0, 3).map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(4) : v}`).join(' ') || '—'}</span>
                  {e.git_sha && <span style={{ fontFamily: 'var(--font-mono)' }}>@{e.git_sha}</span>}
                </div>
                <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginTop: 2 }}>
                  {e.created_at?.slice(5, 16) ?? ''}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 右：详情 */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {activeExp ? (
            <>
              {/* 头部 */}
              <div style={{ ...colTitle, display: 'flex', alignItems: 'center', gap: 8 }}>
                <span>{activeExp.name || activeExp.exp_id}</span>
                <span style={statusStyle(activeExp.status)}>{STATUS_LABEL[activeExp.status] ?? activeExp.status}</span>
                {activeExp.git_sha && <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--color-text-tertiary)' }}>@{activeExp.git_sha}</span>}
                <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>{activeExp.created_at?.slice(5, 16)} → {activeExp.finished_at?.slice(5, 16) || '…'}</span>
                <div style={{ flex: 1 }} />
                <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)', fontFamily: 'var(--font-mono)' }}>
                  {activeExp.command?.slice(0, 80)}
                </span>
              </div>

              <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
                {/* 指标 */}
                <div style={{ width: 300, flexShrink: 0, borderRight: '1px solid var(--color-border)', overflow: 'auto' }}>
                  <div style={colTitle}>指标</div>
                  {Object.keys(activeExp.metrics || {}).length === 0 ? (
                    <div style={{ padding: 12, fontSize: 12, color: 'var(--color-text-tertiary)' }}>
                      暂无指标。命令在所在目录写 metrics.json 或 metrics.csv 后自动解析。
                    </div>
                  ) : (
                    <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                      <tbody>
                        {Object.entries(activeExp.metrics).map(([k, v]) => (
                          <tr key={k} style={{ borderBottom: '1px solid var(--color-border)' }}>
                            <td style={{ padding: '6px 10px', color: 'var(--color-text-secondary)' }}>{k}</td>
                            <td style={{ padding: '6px 10px', fontFamily: 'var(--font-mono)', textAlign: 'right' }}>
                              {Array.isArray(v) && v.length > 1 && v.every(x => typeof x === 'number')
                                ? <MetricSparkline values={v as number[]} />
                                : typeof v === 'number' ? v.toFixed(6) : String(v)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>

                {/* 日志 + git */}
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                  <div style={{ ...colTitle, display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>日志</span>
                    <div style={{ flex: 1 }} />
                    <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>后台运行中每 3s 自动刷新</span>
                  </div>
                  {gitOpen ? (
                    <pre style={{
                      flex: 1, overflow: 'auto', margin: 0, padding: 10,
                      fontSize: 12, lineHeight: 1.6, fontFamily: 'var(--font-mono)',
                      color: 'var(--color-text)', whiteSpace: 'pre-wrap', wordBreak: 'break-all',
                    }}>
                      {gitOut || '—'}
                    </pre>
                  ) : (
                    <pre
                      ref={logRef}
                      style={{
                        flex: 1, overflow: 'auto', margin: 0, padding: 10,
                        fontSize: 12, lineHeight: 1.6, fontFamily: 'var(--font-mono)',
                        color: 'var(--color-text)', whiteSpace: 'pre-wrap', wordBreak: 'break-all',
                      }}
                    >
                      {activeExp.log_tail || ''}
                    </pre>
                  )}
                </div>
              </div>
            </>
          ) : (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-text-tertiary)', fontSize: 13 }}>
              ← 选择或启动一个实验
            </div>
          )}
        </div>
      </div>
    </div>
  );
}