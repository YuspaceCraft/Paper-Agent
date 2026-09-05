/**
 * WorkspacePanel.tsx — 右侧工作台（对话中心化 L2）。
 *
 * 三 Tab（文档/实验/文件），内容绑定当前对话的 active 工件：
 *   - 📄 文档   → DocPanel（对话绑定的 docId）
 *   - 🧪 实验   → ExperimentPanel（对话绑定的 project）
 *   - 📁 文件   → FileExplorer（原右侧文件浏览器）
 * 宿主宽度沿用 --right-w（与 APP 拖拽手柄共用），对话永远留在主区不被替换。
 */

import { type FC } from 'react';
import { DocPanel } from './DocPanel';
import { ExperimentPanel } from './ExperimentPanel';
import { FileExplorer } from './FileExplorer';

export type PanelTab = 'doc' | 'experiment' | 'files';

interface Props {
  tab: PanelTab;
  onTab: (t: PanelTab) => void;
  /** 对话绑定的文档/项目（SSE 归因）。 */
  docId?: string;
  project?: string;
  writingDir?: string;
}

const TABS: Array<{ key: PanelTab; label: string; icon: string }> = [
  { key: 'doc', label: '文档', icon: '📄' },
  { key: 'experiment', label: '实验', icon: '🧪' },
  { key: 'files', label: '文件', icon: '📁' },
];

const panelStyle: React.CSSProperties = {
  width: 'var(--right-w)',
  background: 'var(--color-surface)',
  borderLeft: '1px solid var(--color-border)',
  display: 'flex',
  flexDirection: 'column',
  flexShrink: 0,
  overflow: 'hidden',
};

export const WorkspacePanel: FC<Props> = ({ tab, onTab, docId, project, writingDir }) => {
  return (
    <aside style={panelStyle}>
      {/* Tab 头 */}
      <div style={{ display: 'flex', gap: 4, padding: '6px 8px', borderBottom: '1px solid var(--color-border)', flexShrink: 0 }}>
        {TABS.map(t => (
          <button
            key={t.key}
            onClick={() => onTab(t.key)}
            style={{
              flex: 1, padding: '5px 0', borderRadius: 6, fontSize: 13,
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4,
              background: tab === t.key ? 'var(--color-primary-light)' : 'transparent',
              color: tab === t.key ? 'var(--color-primary)' : 'var(--color-text-secondary)',
              fontWeight: tab === t.key ? 600 : 400,
            }}
          >
            <span>{t.icon}</span><span>{t.label}</span>
          </button>
        ))}
      </div>

      {/* 内容区 */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {tab === 'doc' ? (
          <DocPanel docId={docId} writingDir={writingDir} />
        ) : tab === 'experiment' ? (
          <ExperimentPanel project={project} />
        ) : (
          <FileExplorer open root="project" />
        )}
      </div>
    </aside>
  );
};