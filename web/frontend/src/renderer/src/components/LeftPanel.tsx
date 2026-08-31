/**
 * LeftPanel.tsx — conversation thread manager.
 */

import type { FC } from 'react';
import type { ThreadMeta } from '../state';

interface Props {
  open: boolean;
  threads: Record<string, ThreadMeta>;
  threadOrder: string[];
  activeThreadId: string | null;
  onCreate: () => void;
  onSwitch: (id: string) => void;
  onDelete: (id: string) => void;
}

const panelStyle: React.CSSProperties = {
  width: 'var(--left-w)',
  background: 'var(--color-surface)',
  borderRight: '1px solid var(--color-border)',
  display: 'flex',
  flexDirection: 'column',
  flexShrink: 0,
  overflow: 'hidden',
};

export const LeftPanel: FC<Props> = ({ open, threads, threadOrder, activeThreadId, onCreate, onSwitch, onDelete }) => {
  if (!open) return null;

  return (
    <aside style={panelStyle}>
      <button
        onClick={onCreate}
        style={{
          margin: 12,
          padding: '10px 0',
          borderRadius: 6,
          background: 'var(--color-primary)',
          color: '#fff',
          fontWeight: 600,
          fontSize: 14,
        }}
      >
        + 新建对话
      </button>

      <div style={{ flex: 1, overflowY: 'auto' }}>
        {threadOrder.map(id => {
          const t = threads[id];
          const active = id === activeThreadId;
          return (
            <div
              key={id}
              onClick={() => onSwitch(id)}
              style={{
                padding: '10px 12px',
                margin: '0 8px 2px',
                borderRadius: 6,
                cursor: 'pointer',
                background: active ? 'var(--color-primary-light)' : 'transparent',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <div style={{ minWidth: 0 }}>
                <div style={{
                  fontWeight: 500,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}>
                  {t.title || '(新对话)'}
                </div>
                <div style={{ fontSize: 11, color: 'var(--color-text-secondary)', marginTop: 2 }}>
                  {t.messageCount} 条消息 · {new Date(t.updatedAt).toLocaleDateString()}
                </div>
              </div>
              <button
                onClick={e => { e.stopPropagation(); onDelete(id); }}
                style={{
                  fontSize: 12,
                  color: 'var(--color-text-secondary)',
                  padding: '2px 4px',
                  visibility: active ? 'visible' : undefined,
                  opacity: 0.5,
                }}
                onMouseEnter={e => { (e.target as HTMLElement).style.opacity = '1'; }}
                onMouseLeave={e => { (e.target as HTMLElement).style.opacity = '0.5'; }}
                title="删除对话"
              >
                ✕
              </button>
            </div>
          );
        })}
      </div>
    </aside>
  );
};
