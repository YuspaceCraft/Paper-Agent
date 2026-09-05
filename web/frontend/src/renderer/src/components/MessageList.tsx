/**
 * MessageList.tsx — scrollable conversation, CowAgent-style bubbles.
 *
 * Assistant turns show collapsible tool steps above a markdown answer; user
 * turns are accent-filled right-aligned bubbles. Plan mode additionally renders
 * a live TODO checklist (per-step status) + progress + verification banner.
 */

import { type FC, type ReactNode, useEffect, useRef } from 'react';
import type { Message, PlanStep } from '../state';
import { useApp } from '../state/appContext';
import { Markdown } from './Markdown';
import { MessageSteps } from './MessageSteps';

interface Props {
  messages: Message[];
  onSuggestion: (text: string) => void;
}

const SUGGESTIONS = [
  { icon: '🔍', title: '检索文献', sub: '查找相关论文', prompt: '帮我检索关于检索增强生成（RAG）的论文，并总结其核心方法' },
  { icon: '📄', title: '解读论文', sub: '方法与实验结果', prompt: '解读一篇论文的方法和实验结果' },
  { icon: '📚', title: '对比分析', sub: '多篇论文差异', prompt: '对比两篇论文的技术方案差异' },
  { icon: '🧮', title: '解释公式', sub: '公式的含义与作用', prompt: '解释论文中的损失函数公式及其作用' },
  { icon: '📑', title: '阅读章节', sub: '定位具体章节', prompt: '总结某篇论文的实验部分' },
  { icon: '💡', title: '概念解释', sub: '领域基础知识', prompt: '解释向量数据库的倒排索引与向量索引原理' },
];

// ---- plan TODO checklist ----

const PLAN_STATUS_META: Record<NonNullable<PlanStep['status']>, { icon: ReactNode; color: string }> = {
  pending: { icon: <span style={{ opacity: 0.45 }}>○</span>, color: 'var(--color-text-secondary)' },
  running: { icon: <span className="step-spinner" />, color: 'var(--color-primary)' },
  done: { icon: <span style={{ color: 'var(--color-success)' }}>✓</span>, color: 'var(--color-success)' },
  failed: { icon: <span style={{ color: 'var(--color-danger)' }}>✗</span>, color: 'var(--color-danger)' },
  skipped: { icon: <span style={{ opacity: 0.45 }}>⊘</span>, color: 'var(--color-text-secondary)' },
};

const planHeaderStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  fontSize: 11,
  color: 'var(--color-text-secondary)',
  marginBottom: 4,
};

const planTodoStyle: React.CSSProperties = {
  margin: '4px 0 8px',
  padding: '6px 9px',
  borderRadius: 6,
  background: 'var(--color-inset)',
  border: '1px solid var(--color-border)',
  fontSize: 12,
  lineHeight: 1.5,
};

const planRowStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'flex-start',
  gap: 7,
  padding: '2px 0',
  color: 'var(--color-text-secondary)',
};

const PlanTodo: FC<{ plan: PlanStep[]; progress?: { done: number; total: number } | null }> = ({ plan, progress }) => {
  if (!plan || plan.length === 0) return null;
  const total = progress?.total ?? plan.length;
  const done = progress?.done ?? plan.filter(s => s.status === 'done' || s.status === 'skipped').length;
  return (
    <div style={planTodoStyle}>
      <div style={planHeaderStyle}>
        <span style={{ opacity: 0.6, fontWeight: 600 }}>🧩 计划</span>
        {total > 0 && (
          <span style={{ opacity: 0.7 }}>
            已执行 {done}/{total} 步
          </span>
        )}
      </div>
      {plan.map((s, i) => {
        const meta = PLAN_STATUS_META[s.status ?? 'pending'];
        return (
          <div key={s.id} style={{ ...planRowStyle, color: meta.color }}>
            <span style={{ flexShrink: 0, width: 14, display: 'inline-flex', justifyContent: 'center' }}>{meta.icon}</span>
            <span style={{ flexShrink: 0, opacity: 0.55 }}>{i + 1}.</span>
            <span style={{ flex: 1 }} title={s.status === 'skipped' ? s.output : undefined}>
              {s.description}
            </span>
          </div>
        );
      })}
    </div>
  );
};

// ---- verification banner ----

const VERIFY_META: Record<string, { label: string; fg: string; bg: string }> = {
  satisfied: { label: '验证通过：计划步骤全部完成', fg: 'var(--color-success)', bg: 'rgba(46,160,67,0.10)' },
  partial: { label: '部分完成：仍有步骤未达成', fg: '#b58914', bg: 'rgba(219,171,9,0.12)' },
  failed: { label: '有步骤失败，未能完整回答', fg: 'var(--color-danger)', bg: 'rgba(197,48,48,0.10)' },
  no_evidence: { label: '无可用执行结果', fg: 'var(--color-text-secondary)', bg: 'rgba(120,120,120,0.10)' },
};

const verifyBannerStyle: React.CSSProperties = {
  margin: '4px 0 8px',
  padding: '6px 9px',
  borderRadius: 6,
  border: '1px solid var(--color-border)',
  fontSize: 12,
  lineHeight: 1.5,
};

const VerifyBanner: FC<{ verify: Message['verify'] }> = ({ verify }) => {
  if (!verify) return null;
  const meta = VERIFY_META[verify.status] ?? VERIFY_META.no_evidence;
  return (
    <div style={{ ...verifyBannerStyle, background: meta.bg, color: meta.fg }}>
      <div style={{ fontWeight: 600 }}>
        {verify.status === 'satisfied' ? '✅ ' : verify.status === 'failed' ? '❌ ' : '⚠️ '}
        {meta.label}
      </div>
      {verify.outstanding && verify.outstanding.length > 0 && (
        <div style={{ marginTop: 3, opacity: 0.9 }}>
          {verify.outstanding.map(o => (
            <div key={o.id || o.description} style={{ display: 'flex', gap: 6 }}>
              <span style={{ flexShrink: 0 }}>·</span>
              <span>
                <strong>{o.description || o.id}</strong>
                {o.reason ? ` — ${o.reason}` : ''}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// ---- mode badge ----

const modeBadgeStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 3,
  fontSize: 10,
  fontWeight: 600,
  padding: '1px 8px',
  borderRadius: 10,
  marginBottom: 5,
  background: 'var(--color-inset)',
  border: '1px solid var(--color-border)',
  color: 'var(--color-text-secondary)',
};

const tsWrap: React.CSSProperties = { textAlign: 'right', fontSize: 10, color: 'var(--color-text-tertiary)', marginTop: 3 };
const tsWrapLeft: React.CSSProperties = { ...tsWrap, textAlign: 'left', paddingLeft: 40 };

const MessageTimestamp: FC<{ time: string; right?: boolean }> = ({ time, right }) => {
  const t = new Date(time);
  if (Number.isNaN(t.getTime())) return null;
  return (
    <div style={right ? tsWrap : tsWrapLeft}>
      {t.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
    </div>
  );
};

export const MessageList: FC<Props> = ({ messages, onSuggestion }) => {
  const { uiConfig } = useApp();
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const compact = uiConfig.density === 'compact';

  if (messages.length === 0) {
    return (
      <div className="chat-home">
        <div className="chat-home-logo">📚</div>
        <h1>科研文献助手</h1>
        <p className="chat-home-sub">上传论文后，用自然语言提问。助手会检索文献、定位章节、引用原文，并逐步展示执行过程。</p>
        <div className="suggestion-grid">
          {SUGGESTIONS.map(s => (
            <button key={s.title} className="suggestion-card" onClick={() => onSuggestion(s.prompt)}>
              <span className="suggestion-icon">{s.icon}</span>
              <span className="suggestion-title">{s.title}</span>
              <span className="suggestion-sub">{s.sub}</span>
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className={`msg-list${compact ? ' msg-list-compact' : ''}`}>
      {messages.map(m => {
        if (m.role === 'user') {
          return (
            <div key={m.id} className="msg-row msg-row-user" style={{ flexDirection: 'column', alignItems: 'flex-end' }}>
              <div className="bubble-user">{m.content}</div>
              {uiConfig.showTimestamps && <MessageTimestamp time={m.timestamp} right />}
            </div>
          );
        }

        const streaming = m.status === 'streaming';
        const hasSteps = !!(m.steps && m.steps.length > 0);
        const showCursor = streaming && !!m.content;
        const showTyping = streaming && !m.content && !hasSteps;

        return (
          <div key={m.id} className="msg-row msg-row-assistant">
            <div className="msg-avatar">🤖</div>
            <div className="bubble-assistant">
              {m.mode && (
                <div style={modeBadgeStyle}>
                  {m.mode === 'plan' ? '🧩 Plan' : '⚡ ReAct'}
                </div>
              )}
              <PlanTodo plan={m.plan ?? []} progress={m.planProgress ?? null} />
              <VerifyBanner verify={m.verify ?? null} />
              {hasSteps && (
                <div className="bubble-steps">
                  <MessageSteps steps={m.steps!} defaultExpanded={uiConfig.stepsExpanded} />
                </div>
              )}
              {!!m.workNotes && m.workNotes.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4, margin: '4px 0' }}>
                  {m.workNotes.map(note => (
                    <div key={note.id} style={{
                      display: 'inline-flex', alignItems: 'center', gap: 6,
                      fontSize: 12, padding: '3px 10px', borderRadius: 10,
                      alignSelf: 'flex-start',
                      color: note.status === 'done' ? 'var(--color-success)'
                        : note.status === 'running' || note.status === 'writing' || note.status === 'created' ? 'var(--color-warning)'
                        : 'var(--color-text-secondary)',
                      background: note.status === 'done' ? 'rgba(74,190,110,0.12)'
                        : note.status === 'running' || note.status === 'writing' || note.status === 'created' ? 'rgba(245,158,11,0.12)'
                        : 'var(--color-inset)',
                    }}>
                      <span>{note.kind === 'doc' ? '📄' : '🧪'}</span>
                      <span>{note.text}</span>
                    </div>
                  ))}
                </div>
              )}
              {m.content && <Markdown content={m.content} />}
              {showTyping && (
                <div className="typing-row">
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                </div>
              )}
              {showCursor && <span className="stream-cursor" />}
              {m.status === 'aborted' && <div className="msg-aborted">已停止</div>}
              {uiConfig.showTimestamps && <MessageTimestamp time={m.timestamp} />}
            </div>
          </div>
        );
      })}
      <div ref={bottom} />
    </div>
  );
};