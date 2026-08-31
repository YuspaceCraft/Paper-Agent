/**
 * StatusBar.tsx — bottom system info bar。
 * 任务进度已统一收敛到顶部任务区（TaskCenter），这里只留被动系统信息。
 */

import { type FC } from 'react';

interface Props {
  indexStats: { backend: string; collection_name: string; count: number } | null;
  agentHealth: { model: string } | null;
}

const barStyle: React.CSSProperties = {
  minHeight: 'var(--statusbar-h)',
  display: 'flex',
  alignItems: 'center',
  gap: 16,
  padding: '0 16px',
  background: 'var(--color-surface)',
  borderTop: '1px solid var(--color-border)',
  fontSize: 11,
  color: 'var(--color-text-secondary)',
  flexShrink: 0,
  flexWrap: 'wrap',
};

export const StatusBar: FC<Props> = ({ indexStats, agentHealth }) => {
  return (
    <footer style={barStyle}>
      <div style={{ flex: 1 }} />
      <span>Qdrant: {indexStats?.count ?? '?'} chunks</span>
      <span>|</span>
      <span>Model: {agentHealth?.model ?? '?'}</span>
    </footer>
  );
};

export default StatusBar;