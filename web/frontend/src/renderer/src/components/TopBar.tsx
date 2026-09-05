/**
 * TopBar.tsx — app header bar.
 *
 * 对话中心化后不再有「领域工作区切换」（右侧工作台 WorkspacePanel 的 Tab 承担）；
 * Upload PDF 移到聊天输入框（ChatInput 上传图标），此处只管侧栏/配置中心/在线灯。
 * 右上角「⚙ 配置」打开配置中心（ConfigCenter 模态）；路径配置移入「实验配置」Tab。
 */

import type { FC } from 'react';
import type { Settings } from '../api';

interface Props {
  leftPanelOpen: boolean;
  rightPanelOpen: boolean;
  onToggleLeft: () => void;
  onToggleRight: () => void;
  apiOnline: boolean;
  settings: Settings | null;
  onOpenConfig: () => void;
}

const barStyle: React.CSSProperties = {
  height: 'var(--topbar-h)',
  display: 'flex',
  alignItems: 'center',
  gap: 12,
  padding: '0 16px',
  background: 'var(--color-surface)',
  borderBottom: '1px solid var(--color-border)',
  flexShrink: 0,
  zIndex: 10,
};

const btnStyle: React.CSSProperties = {
  padding: '4px 8px',
  borderRadius: 4,
  fontSize: 13,
  display: 'flex',
  alignItems: 'center',
  gap: 4,
};

export const TopBar: FC<Props> = ({ leftPanelOpen, rightPanelOpen, onToggleLeft, onToggleRight, apiOnline, settings, onOpenConfig }) => {
  const isCustom = !!settings?.project_path;

  return (
  <header style={barStyle}>
    <button style={btnStyle} onClick={onToggleLeft} title="Toggle sidebar">
      {leftPanelOpen ? '◀' : '▶'}
    </button>

    <span style={{ fontWeight: 700, fontSize: 15 }}>📚 Demo</span>

    <div style={{ flex: 1 }} />

    {/* 配置中心（通用/实验/工具/MCP/Skills 五板块；路径配置在「实验配置」Tab） */}
    <button
      style={{ ...btnStyle, border: '1px solid var(--color-border)', borderRadius: 6, cursor: 'pointer' }}
      onClick={onOpenConfig}
      title="配置中心（通用 / 实验 / 工具 / MCP / Skills）"
    >
      ⚙ 配置{isCustom ? ' ·已配置' : ''}
    </button>

    <span style={{
      width: 8, height: 8, borderRadius: '50%',
      background: apiOnline ? 'var(--color-success)' : 'var(--color-danger)',
      flexShrink: 0,
    }} title={apiOnline ? 'API online' : 'API offline'} />

    <button style={btnStyle} onClick={onToggleRight} title="Toggle workspace panel">
      {rightPanelOpen ? '▶' : '◀'}
    </button>
  </header>
  );
};