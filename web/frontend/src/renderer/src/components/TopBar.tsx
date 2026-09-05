/**
 * TopBar.tsx — app header bar.
 *
 * 对话中心化后不再有「领域工作区切换」（右侧工作台 WorkspacePanel 的 Tab 承担）；
 * Upload PDF 移到聊天输入框（ChatInput 上传图标），此处只管侧栏/路径/在线灯。
 */

import { useState } from 'react';
import type { FC } from 'react';
import type { Settings } from '../api';
import { ProjectPathPicker } from './ProjectPathPicker';

interface Props {
  leftPanelOpen: boolean;
  rightPanelOpen: boolean;
  onToggleLeft: () => void;
  onToggleRight: () => void;
  apiOnline: boolean;
  settings: Settings | null;
  onUpdatePaths: (patch: { project_path?: string | null; experiments_path?: string | null }) => void;
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

export const TopBar: FC<Props> = ({ leftPanelOpen, rightPanelOpen, onToggleLeft, onToggleRight, apiOnline, settings, onUpdatePaths }) => {
  const [pickerOpen, setPickerOpen] = useState(false);
  const isCustom = !!settings?.project_path;

  return (
  <header style={barStyle}>
    <button style={btnStyle} onClick={onToggleLeft} title="Toggle sidebar">
      {leftPanelOpen ? '◀' : '▶'}
    </button>

    <span style={{ fontWeight: 700, fontSize: 15 }}>📚 Demo</span>

    <div style={{ flex: 1 }} />

    {/* 项目路径配置（文献问答 + 写作根；实验根在实验工作区单独配） */}
    <button
      style={{ ...btnStyle, border: '1px solid var(--color-border)', borderRadius: 6, cursor: 'pointer' }}
      onClick={() => setPickerOpen(o => !o)}
      title={isCustom ? `项目路径: ${settings!.project_path}` : '项目路径: 默认（代码仓库根）'}
    >
      ⚙ 路径{isCustom ? ' ·已配置' : ''}
    </button>
    {pickerOpen && (
      <ProjectPathPicker
        label="项目路径"
        hint={settings ? `写作保存到 {路径}/writing` : ''}
        value={settings?.project_path ?? ''}
        allowClear
        onPick={p => { onUpdatePaths({ project_path: p || null }); setPickerOpen(false); }}
        onClose={() => setPickerOpen(false)}
      />
    )}

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
