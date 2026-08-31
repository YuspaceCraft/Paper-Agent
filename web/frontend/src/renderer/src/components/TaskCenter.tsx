/**
 * TaskCenter.tsx — 任务区域（聊天区上方，可折叠）。
 *
 * 折叠（默认）：单行摘要「N 个任务运行中 + 最近任务」；点标题展开。
 * 展开分两区：
 *   当前任务 —— 本对话 运行中/排队中（+ 系统级运行中任务），恒显示进度条
 *              与进度文案；运行中超时（>20min 无更新）提示「可能已中断」。
 *   历史任务 —— 本对话 已完成/失败，最新在前，可关闭。
 *
 * 展示 3 态：运行中(pending|running) / 完成(done) / 失败(failed)。
 * 当前/历史是纯派生：状态迁移后自动换区，不需要手动 dismiss 才能移走。
 * 归属（threadId）由 App 前端补齐；null = 系统级任务。
 */

import { useEffect, useRef, useState, type FC } from 'react';
import type { BackgroundTask } from '../state';
import { parseTaskTs } from '../state';

interface Props {
  tasks: BackgroundTask[];
  activeThreadId: string | null;
  onDismiss: (taskId: string) => void;
}

// 运行中超时判「可能已中断」的阈值
const STALE_MS = 20 * 60 * 1000;
const HISTORY_CAP = 100;

// ---- 派生工具 ----

function taskTitle(t: BackgroundTask): string {
  if (t.kind === 'ingest') return `入库「${t.paperName || '论文'}」`;
  if (t.paperName) return `处理 ${t.paperName}`;
  if (t.kind) return `${t.kind} 任务`;
  return '后台任务';
}

function stageText(t: BackgroundTask): string {
  if (t.stage === 'parse') return '解析中';
  if (t.stage === 'index') return '向量化入库中';
  return '';
}

function progressLine(t: BackgroundTask): string {
  if (t.progress) return t.progress;
  if (t.status === 'pending') return '排队中…';
  return stageText(t) || '进行中…';
}

function isStale(t: BackgroundTask): boolean {
  if (t.status !== 'running') return false;
  const ms = parseTaskTs(t.updatedAt || t.createdAt);
  return Number.isFinite(ms) && Date.now() - ms > STALE_MS;
}

function fmtClock(ts: string): string {
  const ms = parseTaskTs(ts);
  if (!Number.isFinite(ms)) return '';
  const d = new Date(ms);
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  return `${hh}:${mm}`;
}

function fmtDuration(start: string | null, end: string | null): string {
  if (!start || !end) return '';
  const ms0 = parseTaskTs(start);
  const ms1 = parseTaskTs(end);
  if (!Number.isFinite(ms0) || !Number.isFinite(ms1) || ms1 < ms0) return '';
  const sec = Math.round((ms1 - ms0) / 1000);
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return s ? `${m}m ${s}s` : `${m}m`;
}

function resultMessage(t: BackgroundTask): string {
  const r = t.result;
  if (r && typeof r.message === 'string' && r.message) return r.message;
  return '';
}

// ---- 样式 ----

const containerStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  flexShrink: 0,
  background: 'var(--color-surface)',
  borderBottom: '1px solid var(--color-border)',
  fontSize: 12,
  color: 'var(--color-text-secondary)',
};

const headerStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  padding: '5px 12px',
  cursor: 'pointer',
  userSelect: 'none',
};

const badgeStyle: React.CSSProperties = {
  background: 'var(--color-primary-light)',
  color: 'var(--color-primary)',
  borderRadius: 8,
  padding: '0 6px',
  fontSize: 11,
  fontWeight: 600,
};

const spinnerStyle: React.CSSProperties = {
  width: 10,
  height: 10,
  border: '2px solid var(--color-border)',
  borderTopColor: 'var(--color-primary)',
  borderRadius: '50%',
  animation: 'spin 0.6s linear infinite',
  flexShrink: 0,
};

const sectionHeader: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 600,
  opacity: 0.75,
  marginBottom: 4,
  display: 'flex',
  alignItems: 'center',
  gap: 6,
};

const itemCard: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  padding: '5px 8px',
  borderRadius: 6,
  border: '1px solid var(--color-border)',
  background: 'var(--color-bg)',
  marginBottom: 4,
};

const ellipsis: React.CSSProperties = {
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
  minWidth: 0,
};

const barTrack: React.CSSProperties = {
  width: '100%',
  height: 3,
  marginTop: 4,
  borderRadius: 2,
  background: 'var(--color-inset)',
  overflow: 'hidden',
  position: 'relative',
};

const barFill: React.CSSProperties = {
  position: 'absolute',
  top: 0,
  bottom: 0,
  left: 0,
  width: '40%',
  borderRadius: 2,
  background: 'var(--color-primary)',
  opacity: 0.7,
  animation: 'taskbar-slide 1.1s ease-in-out infinite',
};

const dismissStyle: React.CSSProperties = {
  border: 'none',
  background: 'transparent',
  color: 'var(--color-text-secondary)',
  opacity: 0.6,
  cursor: 'pointer',
  fontSize: 13,
  lineHeight: 1,
  padding: '0 2px',
  flexShrink: 0,
};

const historyCard: React.CSSProperties = {
  ...itemCard,
  background: 'transparent',
};

// ---- components ----

const CurrentItem: FC<{ t: BackgroundTask; onDismiss: (id: string) => void }> = ({ t, onDismiss }) => {
  const stale = isStale(t);
  const stage = stageText(t);
  return (
    <div style={itemCard} title={`任务 ${t.taskId}`}>
      <span style={spinnerStyle} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ ...ellipsis, color: 'var(--color-text)', fontWeight: 500 }}>{taskTitle(t)}</span>
          {stage && (
            <span style={{
              flexShrink: 0, fontSize: 10, padding: '0 5px', borderRadius: 3,
              background: 'var(--color-inset)', color: 'var(--color-primary)',
            }}>{stage}</span>
          )}
          {stale && (
            <span style={{
              flexShrink: 0, fontSize: 10, padding: '0 5px', borderRadius: 3,
              background: '#fef2f2', color: 'var(--color-danger)',
            }}>可能已中断</span>
          )}
        </div>
        <div style={{ ...ellipsis, marginTop: 2 }}>{progressLine(t)}</div>
        <div style={barTrack}><span style={barFill} /></div>
      </div>
      <button style={dismissStyle} onClick={() => onDismiss(t.taskId)} title="关闭">×</button>
    </div>
  );
};

const HistoryItem: FC<{ t: BackgroundTask; onDismiss: (id: string) => void }> = ({ t, onDismiss }) => {
  const ok = t.status === 'done';
  const summary = ok ? (resultMessage(t) || '已完成') : (t.error ? `失败：${t.error}` : '失败');
  const meta = `${fmtClock(t.createdAt)}${fmtDuration(t.startedAt, t.finishedAt) ? ' · ' + fmtDuration(t.startedAt, t.finishedAt) : ''}`;
  return (
    <div style={historyCard} title={`任务 ${t.taskId}`}>
      <span style={{ flexShrink: 0 }}>{ok ? '✅' : '❌'}</span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ ...ellipsis, color: ok ? 'var(--color-success)' : 'var(--color-danger)', fontWeight: 500 }}>
            {taskTitle(t)}
          </span>
          <span style={{ flexShrink: 0, opacity: 0.7, fontSize: 11 }}>{meta}</span>
        </div>
        <div style={{ ...ellipsis, marginTop: 2, opacity: 0.85 }}>{summary}</div>
      </div>
      <button style={dismissStyle} onClick={() => onDismiss(t.taskId)} title="关闭">×</button>
    </div>
  );
};

const Empty: FC<{ text: string }> = ({ text }) => (
  <div style={{ opacity: 0.55, padding: '2px 4px 6px' }}>{text}</div>
);

export const TaskCenter: FC<Props> = ({ tasks, activeThreadId, onDismiss }) => {
  const [open, setOpen] = useState(false);

  // 当前区：本对话 + 系统级 的运行中任务；历史区：本对话的终端任务
  const running = tasks
    .filter(t => (t.status === 'running' || t.status === 'pending')
      && (t.threadId === activeThreadId || t.threadId === null))
    .sort((a, b) => (b.createdAt || '').localeCompare(a.createdAt || ''));
  const history = tasks
    .filter(t => (t.status === 'done' || t.status === 'failed') && t.threadId === activeThreadId)
    .sort((a, b) => (b.createdAt || '').localeCompare(a.createdAt || ''))
    .slice(0, HISTORY_CAP);

  const latestRunning = running[0];

  // 有新任务开始运行 → 自动展开，避免任务区在对话进行中"看起来消失了"；
  // 用户手动折叠后（期间无新运行任务）不会被打扰。
  const hadRunning = useRef(false);
  useEffect(() => {
    const hasRunning = running.length > 0;
    if (hasRunning && !hadRunning.current) setOpen(true);
    hadRunning.current = hasRunning;
  }, [running.length]);

  return (
    <div style={containerStyle}>
      <div
        style={headerStyle}
        onClick={() => setOpen(v => !v)}
        title={open ? '收起任务区' : '展开任务区'}
      >
        <span style={{ fontWeight: 600, color: 'var(--color-text)' }}>任务</span>
        {running.length > 0 && <span style={badgeStyle}>{running.length} 运行中</span>}
        <span style={{ flex: 1 }} />
        <span style={{ opacity: 0.7 }}>{open ? '▾' : '▸'}</span>
      </div>

      {open ? (
        <div style={{ padding: '2px 12px 8px', overflowY: 'auto', maxHeight: 240 }}>
          <section>
            <div style={sectionHeader}>当前任务 {running.length > 0 && `(${running.length})`}</div>
            {running.length === 0
              ? <Empty text="暂无运行中的任务" />
              : running.map(t => <CurrentItem key={t.taskId} t={t} onDismiss={onDismiss} />)}
          </section>
          <section style={{ marginTop: 8 }}>
            <div style={sectionHeader}>历史任务 {history.length > 0 && `(${history.length})`}</div>
            {history.length === 0
              ? <Empty text="暂无历史任务" />
              : history.map(t => <HistoryItem key={t.taskId} t={t} onDismiss={onDismiss} />)}
          </section>
        </div>
      ) : (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '0 12px 6px', minWidth: 0 }}>
          {running.length === 0 ? (
            <span style={{ opacity: 0.55 }}>
              暂无运行中任务{history.length > 0 ? ` · 历史 ${history.length} 项` : ''}（点击标题展开）
            </span>
          ) : (
            <>
              <span style={spinnerStyle} />
              <span style={{ fontWeight: 500, color: 'var(--color-text)' }}>{running.length} 个任务运行中</span>
              {latestRunning && (
                <>
                  <span style={{ opacity: 0.6 }}>·</span>
                  <span style={{ ...ellipsis, opacity: 0.85 }}>{taskTitle(latestRunning)}</span>
                </>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default TaskCenter;