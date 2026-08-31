/**
 * MessageList.tsx — scrollable conversation, CowAgent-style bubbles.
 *
 * Assistant turns show collapsible tool steps above a markdown answer; user
 * turns are accent-filled right-aligned bubbles.
 */

import { type FC, useEffect, useRef } from 'react';
import type { Message } from '../state';
import { Markdown } from './Markdown';
import { MessageSteps, labelFor } from './MessageSteps';

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

// Plan summary — one muted line of numbered step chips above the step cards.
const planSummaryStyle: React.CSSProperties = {
  display: 'flex', flexWrap: 'wrap', gap: 4, alignItems: 'center',
  fontSize: 11, color: 'var(--color-text-secondary)', marginBottom: 6,
};
const planChipStyle: React.CSSProperties = {
  display: 'inline-flex', alignItems: 'center', gap: 4,
  padding: '1px 8px', borderRadius: 10,
  background: 'var(--color-inset)', border: '1px solid var(--color-border)',
};

export const MessageList: FC<Props> = ({ messages, onSuggestion }) => {
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

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
    <div className="msg-list">
      {messages.map(m => {
        if (m.role === 'user') {
          return (
            <div key={m.id} className="msg-row msg-row-user">
              <div className="bubble-user">{m.content}</div>
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
              {m.plan && m.plan.length > 0 && (
                <div style={planSummaryStyle}>
                  <span style={{ opacity: 0.6 }}>计划</span>
                  {m.plan.map((s, i) => (
                    <span key={s.id} style={planChipStyle} title={s.description}>
                      <span style={{ opacity: 0.5 }}>{i + 1}.</span>
                      {labelFor(s.target)}
                      <span style={{ opacity: 0.75 }}>
                        {s.description.length > 24 ? s.description.slice(0, 24) + '…' : s.description}
                      </span>
                    </span>
                  ))}
                </div>
              )}
              {hasSteps && (
                <div className="bubble-steps">
                  <MessageSteps steps={m.steps!} />
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
            </div>
          </div>
        );
      })}
      <div ref={bottom} />
    </div>
  );
};
