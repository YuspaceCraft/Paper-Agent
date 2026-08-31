/**
 * ChatInput.tsx — message composer at the bottom of MainContent.
 *
 * Streaming state: input disabled, shows a stop button instead of send.
 * CowAgent-style single rounded card: borderless textarea over a toolbar row.
 */

import { type FC, useState, useRef, useCallback } from 'react';

interface Props {
  isStreaming: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
}

export const ChatInput: FC<Props> = ({ isStreaming, onSend, onStop }) => {
  const [text, setText] = useState('');
  const ref = useRef<HTMLTextAreaElement>(null);

  const handleSend = useCallback(() => {
    const trimmed = text.trim();
    if (!trimmed || isStreaming) return;
    onSend(trimmed);
    setText('');
    if (ref.current) ref.current.style.height = 'auto';
  }, [text, isStreaming, onSend]);

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const canSend = !!text.trim();

  return (
    <div className="composer">
      <div className="composer-card">
        <textarea
          ref={ref}
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={handleKey}
          placeholder={isStreaming ? 'Agent 正在回复中...' : '输入你的问题... (Enter 发送, Shift+Enter 换行)'}
          disabled={isStreaming}
          rows={1}
          className="composer-textarea"
        />
        {isStreaming ? (
          <button className="stop-btn" onClick={onStop} title="停止">
            <span className="stop-icon" />
          </button>
        ) : (
          <button
            className="send-btn"
            onClick={handleSend}
            disabled={!canSend}
            title="发送"
            aria-label="发送"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 19V5" />
              <path d="M5 12l7-7 7 7" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
};
