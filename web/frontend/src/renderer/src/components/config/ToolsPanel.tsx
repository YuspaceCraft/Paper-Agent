/**
 * ToolsPanel.tsx — 工具配置：按子 agent（父/arxiv/ingest/creator/coder）分组展示工具清单，
 * 支持停用/启用单个工具 + 搜索过滤。
 *
 * 停用集合持久化到 config_store 后在下次工具构建时过滤（新会话生效；
 * 运行中任务持有旧快照）。max_steps 只读展示（生效权威在 agent/config.yaml）。
 */
import { useEffect, useMemo, useState } from 'react';
import type { FC } from 'react';
import { api, type ConfigTools } from '../../api';
import { InfoLine, PanelShell, SaveBar, Section, Spinner, inputStyle } from './shared';

const AGENT_ORDER = ['parent', 'arxiv', 'ingest', 'creator', 'coder'];

const sourceColor: Record<string, string> = {
  builtin: 'var(--color-primary)',
  generic: 'var(--color-text-secondary)',
  mcp: 'var(--color-warning)',
  skill: 'var(--color-danger)',
};

export const ToolsPanel: FC = () => {
  const [data, setData] = useState<ConfigTools | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [draft, setDraft] = useState<Record<string, string[]>>({});
  const [filter, setFilter] = useState('');
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [saving, setSaving] = useState(false);

  const load = () => {
    setLoading(true); setError('');
    api.getConfigTools()
      .then(ct => {
        setData(ct);
        const d: Record<string, string[]> = {};
        for (const [name, g] of Object.entries(ct.agents)) {
          d[name] = g.tools.filter(t => !t.enabled).map(t => t.name);
        }
        setDraft(d);
        setOpen({ parent: true, ...Object.fromEntries(Object.keys(ct.agents).slice(1).map(k => [k, false])) });
        setLoading(false);
      })
      .catch(e => { setError(String(e.message ?? e)); setLoading(false); });
  };
  useEffect(load, []);

  const toggle = (agent: string, tool: string, on: boolean) => {
    setDraft(prev => {
      const cur = prev[agent] ?? [];
      return { ...prev, [agent]: on ? cur.filter(t => t !== tool) : [...cur, tool] };
    });
  };

  const save = () => {
    setSaving(true);
    api.updateConfigTools({ disabled: draft })
      .then(res => setData(prev => (prev ? { ...prev, agents: res.agents } : prev)))
      .catch(e => setError(String(e.message ?? e)))
      .finally(() => setSaving(false));
  };

  const q = filter.trim().toLowerCase();
  const visibleCount = useMemo(() => {
    if (!data) return 0;
    return Object.values(data.agents).reduce(
      (n, g) => n + g.tools.filter(t => t.name.toLowerCase().includes(q)).length, 0);
  }, [data, q]);

  if (loading) return <Spinner />;
  if (error) return <PanelShell error={error} />;
  if (!data) return null;

  return (
    <div>
      <InfoLine>
        停用/启用工具后点「保存并生效」：新会话按新工具面构建，运行中的任务保持旧快照。
        最大步数为生效配置（agent/config.yaml），只读。
      </InfoLine>
      <input
        placeholder={`搜索 ${visibleCount} 个工具…`}
        value={filter}
        onChange={e => setFilter(e.target.value)}
        style={{ ...inputStyle, marginBottom: 10 }}
      />
      {AGENT_ORDER.map(name => {
        const g = data.agents[name];
        if (!g) return null;
        const blocked = new Set(draft[name] ?? []);
        const tools = g.tools.filter(t => t.name.toLowerCase().includes(q));
        if (q && tools.length === 0) return null;
        const enabled = g.tools.filter(t => !blocked.has(t.name)).length;
        const expanded = q ? true : !!open[name];
        return (
          <Section key={name} title={`${g.label} · ${name}`}>
            <div style={{ border: '1px solid var(--color-border)', borderRadius: 8, overflow: 'hidden' }}>
              <button
                type="button"
                onClick={() => setOpen(o => ({ ...o, [name]: !o[name] }))}
                style={{
                  display: 'flex', alignItems: 'center', gap: 8, width: '100%', textAlign: 'left',
                  padding: '8px 10px', background: 'var(--color-inset)',
                  border: 'none', borderBottom: expanded ? '1px solid var(--color-border)' : 'none',
                  fontSize: 12, cursor: 'pointer',
                }}
              >
                <span style={{ transform: expanded ? 'rotate(90deg)' : 'none', transition: 'transform 0.12s' }}>▶</span>
                <span style={{ flex: 1, fontWeight: 600 }}>{name}</span>
                <span style={{ color: 'var(--color-text-tertiary)' }}>max_steps {g.max_steps}</span>
                <span style={{ color: 'var(--color-text-secondary)' }}>{enabled}/{g.tools.length} 启用</span>
              </button>
              {expanded && (
                <div>
                  {tools.map(t => {
                    const off = blocked.has(t.name);
                    return (
                      <label
                        key={t.name}
                        title={t.description || t.name}
                        style={{
                          display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px',
                          borderBottom: '1px solid var(--color-border)',
                          cursor: 'pointer', fontSize: 12,
                          opacity: off ? 0.55 : 1, background: t.loaded ? 'transparent' : 'rgba(250,204,21,0.06)',
                        }}
                      >
                        <input
                          type="checkbox"
                          style={{ accentColor: 'var(--color-primary)', cursor: 'pointer' }}
                          checked={!off}
                          onChange={e => toggle(name, t.name, e.target.checked)}
                        />
                        <span style={{ fontFamily: 'var(--font-mono)' }}>{t.name}</span>
                        <span style={{
                          marginLeft: 'auto', padding: '1px 6px', borderRadius: 4, fontSize: 10,
                          color: sourceColor[t.source] ?? 'var(--color-text-secondary)',
                          border: '1px solid currentColor', background: 'transparent',
                        }}>
                          {t.loaded ? t.source : '⚠ 未加载'}
                        </span>
                      </label>
                    );
                  })}
                  {tools.length === 0 && (
                    <div style={{ padding: 8, fontSize: 12, color: 'var(--color-text-tertiary)' }}>（无匹配工具）</div>
                  )}
                </div>
              )}
            </div>
          </Section>
        );
      })}

      <SaveBar
        onSave={save} saving={saving} error={error}
        warn="保存后新会话生效（工具表重建）；运行中任务不受影响"
        saveLabel="保存并生效"
      />
      {saving && <span style={{ display: 'inline-block', marginLeft: 8, fontSize: 12, color: 'var(--color-text-tertiary)' }}>重建工具表可能耗时数秒…</span>}
    </div>
  );
};

export default ToolsPanel;