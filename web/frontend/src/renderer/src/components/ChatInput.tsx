/**
 * ChatInput.tsx — message composer at the bottom of MainContent.
 *
 * Streaming state: input disabled, shows a stop button instead of send.
 * CowAgent-style single rounded card: borderless textarea over a toolbar row.
 *
 * 执行模式切换归并为一个按钮（显示当前选择），点击弹出三个选项（自动 /
 * Plan / ReAct），选中即切换；isStreaming 时禁用。
 */

import { type FC, useState, useRef, useCallback, useEffect } from 'react';
import type { AgentMode } from '../state';

interface Props {
  isStreaming: boolean;
  mode: AgentMode;
  onModeChange: (m: AgentMode) => void;
  onSend: (text: string) => void;
  onStop: () => void;
  /** 上传 PDF（输入框内上传图标触发，取代原 TopBar 按钮）。 */
  onUpload: (file: File) => void;
}

const MODES: Array<{ value: AgentMode; label: string; title: string }> = [
  { value: 'auto', label: '🤖 自动', title: '根据任务复杂程度自动选择执行模式' },
  { value: 'plan', label: '🧩 Plan', title: '先规划步骤清单（TODO）再执行，执行后验证计划完成情况' },
  { value: 'react', label: '⚡ ReAct', title: '直接调用工具逐步执行（实时步骤卡片）' },
];

// 当前模式按钮：小而圆润，悬停高亮；放在发送按钮旁
const currentBtnStyle = (disabled: boolean): React.CSSProperties => ({
  display: 'inline-flex',
  alignItems: 'center',
  gap: 4,
  padding: '5px 10px',
  border: '1px solid var(--color-border)',
  borderRadius: 8,
  background: 'transparent',
  color: 'var(--color-text-secondary)',
  fontSize: 12,
  lineHeight: 1.4,
  cursor: disabled ? 'not-allowed' : 'pointer',
  whiteSpace: 'nowrap',
  opacity: disabled ? 0.5 : 1,
  transition: 'border-color 0.12s, color 0.12s',
});

// 下拉选项浮层：从按钮正上方弹出，右对齐防越界
const menuStyle: React.CSSProperties = {
  position: 'absolute',
  right: 0,
  bottom: 'calc(100% + 6px)',
  zIndex: 20,
  minWidth: 150,
  padding: 4,
  background: 'var(--color-surface)',
  border: '1px solid var(--color-border)',
  borderRadius: 10,
  boxShadow: '0 6px 20px rgba(0,0,0,0.14)',
};

const menuItemStyle = (active: boolean): React.CSSProperties => ({
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  width: '100%',
  padding: '6px 10px',
  border: 'none',
  borderRadius: 7,
  background: active ? 'var(--color-primary)' : 'transparent',
  color: active ? '#fff' : 'var(--color-text-primary)',
  fontSize: 12,
  textAlign: 'left',
  cursor: 'pointer',
  whiteSpace: 'nowrap',
});

const uploadBtnStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: '5px 8px',
  border: '1px solid var(--color-border)',
  borderRadius: 8,
  background: 'transparent',
  color: 'var(--color-text-secondary)',
  cursor: 'pointer',
  transition: 'color 0.12s, border-color 0.12s',
};

export const ChatInput: FC<Props> = ({ isStreaming, mode, onModeChange, onSend, onStop, onUpload }) => {
  const [text, setText] = useState('');
  const [menuOpen, setMenuOpen] = useState(false);
  const ref = useRef<HTMLTextAreaElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const current = MODES.find(m => m.value === mode) ?? MODES[0];

  // 点击浮层外部关闭
  useEffect(() => {
    const onDocDown = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', onDocDown);
    return () => document.removeEventListener('mousedown', onDocDown);
  }, []);

  /** 文本域随内容自适应增高（上限 160px，超出滚动）。 */
  const autoResize = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, []);

  const handleSend = useCallback(() => {
    const trimmed = text.trim();
    if (!trimmed || isStreaming) return;
    onSend(trimmed);
    setText('');
    requestAnimationFrame(autoResize);
  }, [text, isStreaming, onSend, autoResize]);

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleModeClick = useCallback((m: AgentMode) => {
    onModeChange(m);
    setMenuOpen(false);
  }, [onModeChange]);

  const canSend = !!text.trim();

  return (
    <div className="composer">
      <div className="composer-card">
        <textarea
          ref={ref}
          value={text}
          onChange={e => {
            setText(e.target.value);
            autoResize();
          }}
          onKeyDown={handleKey}
          placeholder={isStreaming ? 'Agent 正在回复中...' : '输入你的问题... (Enter 发送, Shift+Enter 换行)'}
          disabled={isStreaming}
          rows={1}
          className="composer-textarea"
        />
        <div className="composer-actions">
          <button style={uploadBtnStyle} onClick={() => fileRef.current?.click()} title="上传 PDF" aria-label="上传 PDF">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 17V7" />
              <path d="M6 11l6-6 6 6" />
              <rect x="4" y="18" width="16" height="2" rx="1" />
            </svg>
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,application/pdf"
            style={{ display: 'none' }}
            onChange={e => {
              const f = e.target.files?.[0];
              if (f) onUpload(f);
              e.target.value = '';
            }}
          />
          <div ref={wrapRef} style={{ position: 'relative' }}>
            <button
              style={currentBtnStyle(isStreaming)}
              disabled={isStreaming}
              onClick={() => setMenuOpen(v => !v)}
              title={`执行模式：${current.title}\n点击选择切换`}
              aria-label="切换执行模式"
              aria-haspopup="menu"
              aria-expanded={menuOpen}
            >
              <span>{current.label}</span>
              <span style={{ fontSize: 9, opacity: 0.7 }}>{menuOpen ? '▴' : '▾'}</span>
            </button>
            {menuOpen && (
              <div style={menuStyle} role="menu">
                {MODES.map(m => (
                  <button
                    key={m.value}
                    style={menuItemStyle(mode === m.value)}
                    onClick={() => handleModeClick(m.value)}
                    title={m.title}
                    role="menuitemradio"
                    aria-checked={mode === m.value}
                  >
                    <span style={{ flex: 1 }}>{m.label}</span>
                    {mode === m.value && <span>✓</span>}
                  </button>
                ))}
              </div>
            )}
          </div>
          <div style={{ flex: 1 }} />
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
    </div>
  );
};