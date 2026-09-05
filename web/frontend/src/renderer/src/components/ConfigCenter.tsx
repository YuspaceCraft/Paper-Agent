/**
 * ConfigCenter.tsx — 配置中心（居中模态弹窗）。
 *
 * 左侧竖排导航切换五大板块：通用 / 实验 / 工具 / MCP / Skills。
 * 通用 = 前端本地即时生效；其余 = 后端 GET/PUT（/api/config/*）持久化。
 * 被 ProjectPathPicker 内的 z-index 100 覆盖，故遮罩层用 90、面板用 91。
 */
import { useEffect, useState } from 'react';
import type { FC } from 'react';
import type { AgentHealth, Settings } from '../api';
import type { UIConfig } from '../state/uiConfig';
import { GeneralPanel } from './config/GeneralPanel';
import { ExperimentPanel } from './config/ExperimentPanel';
import { ToolsPanel } from './config/ToolsPanel';
import { McpPanel } from './config/McpPanel';
import { SkillsPanel } from './config/SkillsPanel';

export type ConfigTab = 'general' | 'experiment' | 'tools' | 'mcp' | 'skills';

const NAV: Array<{ id: ConfigTab; icon: string; label: string }> = [
  { id: 'general', icon: '⚙', label: '通用配置' },
  { id: 'experiment', icon: '🧪', label: '实验配置' },
  { id: 'tools', icon: '🛠', label: '工具配置' },
  { id: 'mcp', icon: '🔌', label: 'MCP 配置' },
  { id: 'skills', icon: '📜', label: 'Skills 配置' },
];

interface Props {
  onClose: () => void;
  settings: Settings | null;
  onUpdatePaths: (patch: { project_path?: string | null; experiments_path?: string | null }) => void;
  agentHealth: AgentHealth | null;
  config: UIConfig;
  onUiChange: (patch: Partial<UIConfig>) => void;
}

export const ConfigCenter: FC<Props> = ({ onClose, settings, onUpdatePaths, agentHealth, config, onUiChange }) => {
  const [tab, setTab] = useState<ConfigTab>((config.configTab as ConfigTab) || 'general');
  // 已访问过的板块保持挂载（再切回不丢编辑）；未访问的不挂载（打开配置中心不触发全量请求）
  const [visited, setVisited] = useState<Set<ConfigTab>>(() => new Set([(config.configTab as ConfigTab) || 'general']));

  const switchTab = (t: ConfigTab) => {
    setTab(t);
    setVisited(prev => new Set(prev).add(t));
    onUiChange({ configTab: t });
  };

  // 显示只随激活 Tab（左侧切换即单板块）；visited 只控制「是否已挂载」，
  // 已访问板块保留其 state（未保存编辑），未访问的不挂载（不触发多余请求）。
  const shown = (t: ConfigTab) => tab === t;
  const mounted = (t: ConfigTab) => tab === t || visited.has(t);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div
      style={{
        position: 'fixed', inset: 0, background: 'rgba(15, 23, 42, 0.45)', zIndex: 90,
        display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '4vh 3vw',
      }}
      onClick={onClose}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          width: 'min(880px, 94vw)', height: 'min(76vh, 720px)', maxWidth: '94vw',
          background: 'var(--color-bg)', border: '1px solid var(--color-border)',
          borderRadius: 12, boxShadow: 'var(--shadow-md)', zIndex: 91,
          display: 'flex', flexDirection: 'column', overflow: 'hidden',
        }}
      >
        {/* 头部 */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '12px 16px', borderBottom: '1px solid var(--color-border)',
          background: 'var(--color-surface)', borderTopLeftRadius: 12, borderTopRightRadius: 12,
          flexShrink: 0,
        }}>
          <span style={{ fontSize: 15, fontWeight: 700 }}>⚙️ 配置中心</span>
          <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>通用（本地即时） · 实验 / 工具 / MCP / Skills（后端持久化）</span>
          <div style={{ flex: 1 }} />
          <button
            onClick={onClose}
            title="关闭 (Esc)"
            style={{
              width: 28, height: 28, borderRadius: 6, border: 'none', cursor: 'pointer',
              background: 'transparent', color: 'var(--color-text-secondary)', fontSize: 14,
            }}
          >✕</button>
        </div>

        {/* 主体：左导航 + 右内容 */}
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
          <nav style={{
            width: 158, flexShrink: 0, borderRight: '1px solid var(--color-border)',
            background: 'var(--color-surface)', padding: '10px 8px', overflowY: 'auto',
          }}>
            {NAV.map(n => {
              const active = n.id === tab;
              return (
                <button
                  key={n.id}
                  onClick={() => switchTab(n.id)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 8, width: '100%',
                    padding: '8px 10px', marginBottom: 4, borderRadius: 6, cursor: 'pointer',
                    border: active ? '1px solid var(--color-primary)' : '1px solid transparent',
                    background: active ? 'var(--color-primary-light)' : 'transparent',
                    color: active ? 'var(--color-primary)' : 'var(--color-text-secondary)',
                    fontSize: 13, textAlign: 'left',
                  }}
                >
                  <span>{n.icon}</span>
                  <span>{n.label}</span>
                </button>
              );
            })}
          </nav>

          {/* 显示只跟激活 Tab；已访问板块保持挂载不丢编辑，未访问的不挂载 */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '16px 18px 20px' }}>
            <div style={{ display: shown('general') ? 'block' : 'none' }}>
              {mounted('general') && <GeneralPanel config={config} onChange={onUiChange} agentHealth={agentHealth} />}
            </div>
            <div style={{ display: shown('experiment') ? 'block' : 'none' }}>
              {mounted('experiment') && <ExperimentPanel settings={settings} onUpdatePaths={onUpdatePaths} />}
            </div>
            <div style={{ display: shown('tools') ? 'block' : 'none' }}>
              {mounted('tools') && <ToolsPanel />}
            </div>
            <div style={{ display: shown('mcp') ? 'block' : 'none' }}>
              {mounted('mcp') && <McpPanel />}
            </div>
            <div style={{ display: shown('skills') ? 'block' : 'none' }}>
              {mounted('skills') && <SkillsPanel />}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConfigCenter;