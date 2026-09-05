/**
 * SkillsPanel.tsx — Skills 配置：SKILL.md 标准技能清单 + 启用/停用 + 打开目录。
 * 停用集合持久化到 config_store 后随 reload_tools 过滤（skill__list 不再出现）。
 */
import { useEffect, useState } from 'react';
import type { FC } from 'react';
import { api, type SkillInfo } from '../../api';
import {
  btnBase, InfoLine, Notify, PanelShell, SaveBar, Section, Spinner, Toggle,
} from './shared';

export const SkillsPanel: FC = () => {
  const [list, setList] = useState<SkillInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [blocked, setBlocked] = useState<Set<string>>(new Set());
  const [saving, setSaving] = useState(false);
  const [notice, setNotice] = useState('');

  const load = () => {
    setLoading(true); setError('');
    api.getSkillsConfig()
      .then(r => {
        setList(r.skills);
        setBlocked(new Set(r.skills.filter(s => !s.enabled).map(s => s.name)));
        setLoading(false);
      })
      .catch(e => { setError(String(e.message ?? e)); setLoading(false); });
  };
  useEffect(load, []);

  const toggle = (name: string, on: boolean) => {
    setBlocked(prev => {
      const next = new Set(prev);
      if (on) next.delete(name); else next.add(name);
      return next;
    });
  };

  const save = () => {
    setSaving(true);
    api.updateSkillsConfig([...blocked])
      .then(r => { setList(r.skills); setNotice('已保存：停用技能将不再出现在新会话的 skill__list。'); })
      .catch(e => setError(String(e.message ?? e)))
      .finally(() => setSaving(false));
  };

  if (loading) return <Spinner />;
  if (error && list.length === 0) return <PanelShell error={error} />;

  const openDir = (path: string) => { void window.electronAPI?.shellOpenPath(path); };
  const canOpen = !!window.electronAPI?.shellOpenPath;

  return (
    <div>
      {notice && <Notify kind="ok">{notice}</Notify>}
      <InfoLine>
        技能目录：<span style={{ fontFamily: 'var(--font-mono)' }}>skills/&lt;name&gt;/SKILL.md</span>
        （Anthropic Agent Skills 开放标准：YAML frontmatter + Markdown body + 可选 resources/）。
        新增技能 = 新建目录放入 SKILL.md。停用后再启用即刻生效（保存触发工具表重建）。
      </InfoLine>

      <Section title="技能清单">
        {list.length === 0 && (
          <div style={{ padding: '10px 0', fontSize: 12, color: 'var(--color-text-tertiary)' }}>
            （skills/ 下暂无含 SKILL.md 的技能）
          </div>
        )}
        {list.map(s => {
          const off = blocked.has(s.name);
          return (
            <div key={s.name} style={{
              display: 'flex', alignItems: 'flex-start', gap: 10,
              padding: '9px 10px', border: '1px solid var(--color-border)', borderRadius: 8, marginBottom: 8,
              opacity: off ? 0.6 : 1, background: s.enabled === false && !off ? 'rgba(250,204,21,0.06)' : 'transparent',
            }}>
              <Toggle checked={!off} onChange={v => toggle(s.name, v)} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                  <span style={{ fontWeight: 600, fontFamily: 'var(--font-mono)' }}>{s.name}</span>
                  {s.resources.length > 0 && (
                    <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>
                      resources: {s.resources.join(', ')}
                    </span>
                  )}
                </div>
                {s.description && (
                  <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginTop: 3, lineHeight: 1.5 }}>
                    {s.description}
                  </div>
                )}
                <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginTop: 3, fontFamily: 'var(--font-mono)', wordBreak: 'break-all' }}>
                  {s.path}
                </div>
              </div>
              {canOpen && s.path && (
                <button style={btnBase} onClick={() => openDir(s.path)} title="在文件管理器中打开该技能目录">📂 打开</button>
              )}
            </div>
          );
        })}
      </Section>

      <SaveBar
        onSave={save} saving={saving} error={error}
        warn="停用技能 = 从技能列表移除（不在 skill__list 出现），文件保留"
        saveLabel="保存 Skills 配置"
      />
    </div>
  );
};

export default SkillsPanel;