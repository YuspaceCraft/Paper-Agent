/**
 * TopBar.tsx — app header bar.
 */

import { useState } from 'react';
import type { FC } from 'react';
import type { Settings } from '../api';
import { ProjectPathPicker } from './ProjectPathPicker';

export type Domain = 'chat' | 'write' | 'experiment';

interface Props {
  leftPanelOpen: boolean;
  rightPanelOpen: boolean;
  onToggleLeft: () => void;
  onToggleRight: () => void;
  onUpload: (file: File) => void;
  apiOnline: boolean;
  domain: Domain;
  onSwitchDomain: (d: Domain) => void;
  settings: Settings | null;
  onUpdatePaths: (patch: { project_path?: string | null; experiments_path?: string | null }) => void;
}

const DOMAINS: Array<{ key: Domain; label: string; icon: string }> = [
  { key: 'chat', label: '文献问答', icon: '📚' },
  { key: 'write', label: '论文写作', icon: '✍️' },
  { key: 'experiment', label: '实验', icon: '🧪' },
];

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

export const TopBar: FC<Props> = ({ leftPanelOpen, rightPanelOpen, onToggleLeft, onToggleRight, onUpload, apiOnline, domain, onSwitchDomain, settings, onUpdatePaths }) => {
  const [pickerOpen, setPickerOpen] = useState(false);
  const isCustom = !!settings?.project_path;

  return (
  <header style={barStyle}>
    <button style={btnStyle} onClick={onToggleLeft} title="Toggle sidebar">
      {leftPanelOpen ? '◀' : '▶'}
    </button>

    <span style={{ fontWeight: 700, fontSize: 15 }}>📚 Demo</span>

    {/* 领域工作区 Tab（v10 / Phase B） */}
    <div style={{ display: 'flex', gap: 4, marginLeft: 8 }}>
      {DOMAINS.map(d => (
        <button
          key={d.key}
          onClick={() => onSwitchDomain(d.key)}
          style={{
            padding: '4px 12px', borderRadius: 6, fontSize: 13,
            display: 'flex', alignItems: 'center', gap: 5,
            background: domain === d.key ? 'var(--color-primary-light)' : 'transparent',
            color: domain === d.key ? 'var(--color-primary)' : 'var(--color-text-secondary)',
            fontWeight: domain === d.key ? 600 : 400,
          }}
        >{d.icon} {d.label}</button>
      ))}
    </div>

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

    <label style={{ ...btnStyle, background: 'var(--color-primary-light)', color: 'var(--color-primary)', borderRadius: 6, cursor: 'pointer' }}>
      ⬆ Upload PDF
      <input
        type="file"
        accept=".pdf"
        style={{ display: 'none' }}
        onChange={e => {
          const f = e.target.files?.[0];
          if (f) onUpload(f);
          e.target.value = '';
        }}
      />
    </label>

    <span style={{
      width: 8, height: 8, borderRadius: '50%',
      background: apiOnline ? 'var(--color-success)' : 'var(--color-danger)',
      flexShrink: 0,
    }} title={apiOnline ? 'API online' : 'API offline'} />

    <button style={btnStyle} onClick={onToggleRight} title="Toggle file panel">
      {rightPanelOpen ? '▶' : '◀'}
    </button>
  </header>
  );
};
