/**
 * GeneralPanel.tsx — 通用配置（外观/对话偏好/数据管理/关于）。
 * 纯前端：变更即应用（App 层 merge+持久化+applyUIConfig），后端无需参与。
 */
import { type FC } from 'react';
import type { UIConfig } from '../../state/uiConfig';
import { FieldRow, Section, Segmented, selectStyle, Toggle, btnBase } from './shared';

interface Props {
  config: UIConfig;
  onChange: (patch: Partial<UIConfig>) => void;
  agentHealth: { status: string; model: string; tools: number } | null;
}

const zoomOptions = [
  { value: '0.9', label: '90%' },
  { value: '1', label: '100%' },
  { value: '1.1', label: '110%' },
  { value: '1.25', label: '125%' },
];
const densityOptions = [
  { value: 'comfortable', label: '舒适' },
  { value: 'compact', label: '紧凑' },
];

export const GeneralPanel: FC<Props> = ({ config, onChange, agentHealth }) => {
  const wipeLocalData = () => {
    const ok = window.confirm('将清空全部对话历史、后台任务记录与本机 UI 配置（不影响论文库/索引/磁盘文件）。确认？');
    if (!ok) return;
    try {
      const keys: string[] = [];
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && (k.startsWith('demo_') || k.startsWith('demo_threads') || k.startsWith('demo_msgs_'))) keys.push(k);
      }
      for (const k of keys) localStorage.removeItem(k);
    } catch { /* ignore */ }
    window.location.reload();
  };

  return (
    <div>
      <Section title="外观">
        <FieldRow
          title="主题"
          hint="亮色 / 暗色（暗色调色板即时切换）"
          control={<Segmented value={config.theme} options={[{ value: 'light', label: '亮' }, { value: 'dark', label: '暗' }]} onChange={v => onChange({ theme: v as UIConfig['theme'] })} />}
        />
        <FieldRow
          title="界面缩放"
          hint="整体缩放（含布局与字号），100% 为默认"
          control={(
            <select
              value={String(config.zoom)}
              style={selectStyle}
              onChange={e => onChange({ zoom: Number(e.target.value) })}
            >
              {zoomOptions.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          )}
        />
      </Section>

      <Section title="对话">
        <FieldRow
          title="显示消息时间戳"
          hint="在每条消息下显示发送时间"
          control={<Toggle checked={config.showTimestamps} onChange={v => onChange({ showTimestamps: v })} />}
        />
        <FieldRow
          title="工具步骤默认展开"
          hint="助手消息中的工具调用卡片默认展开明细"
          control={<Toggle checked={config.stepsExpanded} onChange={v => onChange({ stepsExpanded: v })} />}
        />
        <FieldRow
          title="消息密度"
          hint="紧凑模式缩小气泡间距与正文留白"
          control={(
            <select
              value={config.density}
              style={selectStyle}
              onChange={e => onChange({ density: e.target.value as UIConfig['density'] })}
            >
              {densityOptions.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          )}
        />
      </Section>

      <Section title="数据管理">
        <FieldRow
          title="清空本地缓存"
          hint="删除对话历史/任务记录/UI 配置并刷新（不可恢复；论文库与索引不受影响）"
          control={<button style={{ ...btnBase, color: 'var(--color-danger)' }} onClick={wipeLocalData}>清空并刷新</button>}
        />
      </Section>

      <Section title="关于">
        <div style={{ fontSize: 12, lineHeight: 1.8, color: 'var(--color-text-secondary)' }}>
          <div>Demo 桌面客户端 · v0.1.0（Electron + React + FastAPI）</div>
          <div>
            后端状态：
            <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', margin: '0 6px 0 4px', background: agentHealth ? 'var(--color-success)' : 'var(--color-danger)' }} />
            {agentHealth ? `在线 · ${agentHealth.model} · ${agentHealth.tools} 工具` : '离线'}
          </div>
        </div>
      </Section>
    </div>
  );
};

export default GeneralPanel;