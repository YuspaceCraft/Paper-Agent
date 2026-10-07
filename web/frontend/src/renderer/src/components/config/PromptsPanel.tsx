/** Prompt version and deterministic canary/A-B configuration. */

import { type FC, useEffect, useState } from 'react';
import { api, whenBackendReady, type PromptsConfigInfo } from '../../api';
import {
  InfoLine,
  Notify,
  PanelShell,
  SaveBar,
  Section,
  inputStyle,
  selectStyle,
  useNotify,
} from './shared';

export const PromptsPanel: FC = () => {
  const [data, setData] = useState<PromptsConfigInfo | null>(null);
  const [draft, setDraft] = useState<Record<string, { version: string; percent: number }>>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const { notice, notify, cleanup } = useNotify();

  const load = async () => {
    setLoading(true);
    setError('');
    try {
      await whenBackendReady();
      const payload = await api.getPromptsConfig();
      setData(payload);
      setDraft(Object.fromEntries(payload.prompts.map(item => [
        item.id,
        {
          version: item.canary?.version ?? '',
          percent: item.canary?.percent ?? 0,
        },
      ])));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void load(); }, []);

  const save = async () => {
    setSaving(true);
    cleanup();
    try {
      const canary = Object.fromEntries(
        Object.entries(draft).filter(([, value]) => value.version),
      );
      await api.updatePromptsConfig(canary);
      notify('ok', 'Prompt canary 分流已保存');
      setData(await api.getPromptsConfig());
    } catch (e) {
      notify('warn', String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <PanelShell loading={loading} error={error}>
      <InfoLine>
        分流按 thread_id 的稳定 hash 计算，同一会话始终命中同一版本；
        canary 响应仍记录 prompt version、checksum 与 evaluation suite。
      </InfoLine>
      {notice && <Notify kind={notice.kind}>{notice.text}</Notify>}
      <Section title="Canary 分流">
        {(data?.prompts ?? []).map(item => {
          const canaryVersions = item.versions.filter(version => version.status === 'canary');
          const rule = draft[item.id] ?? { version: '', percent: 0 };
          return (
            <div key={item.id} style={{
              borderBottom: '1px solid var(--color-border)',
              padding: '9px 0',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 13, fontWeight: 600 }}>{item.id}</div>
                  <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)', marginTop: 2 }}>
                    active: {item.active?.version ?? '-'} · suite: {item.active?.evaluation_suite || '-'}
                  </div>
                </div>
                <select
                  style={selectStyle}
                  value={rule.version}
                  onChange={e => setDraft(prev => ({
                    ...prev,
                    [item.id]: { ...rule, version: e.target.value },
                  }))}
                >
                  <option value="">关闭</option>
                  {canaryVersions.map(version => (
                    <option key={version.version} value={version.version}>
                      {version.version}
                    </option>
                  ))}
                </select>
                <input
                  style={{ ...inputStyle, width: 72 }}
                  type="number"
                  min={0}
                  max={100}
                  value={rule.percent}
                  onChange={e => setDraft(prev => ({
                    ...prev,
                    [item.id]: { ...rule, percent: Number(e.target.value) },
                  }))}
                />
                <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>%</span>
              </div>
            </div>
          );
        })}
      </Section>
      <SaveBar onSave={save} saving={saving} error="" />
    </PanelShell>
  );
};

export default PromptsPanel;
