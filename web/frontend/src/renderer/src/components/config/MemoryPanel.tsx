/** Long-term memory panel: inspect, add, disable and delete typed records. */

import { type FC, useEffect, useState } from 'react';
import { api, whenBackendReady, type MemoryPayload, type MemoryRecordData } from '../../api';
import {
  InfoLine,
  Notify,
  PanelShell,
  Section,
  btnBase,
  btnPrimary,
  inputStyle,
  selectStyle,
  setBtnDisabled,
  useNotify,
} from './shared';

const MEMORY_TYPES = [
  'preference',
  'long_term_fact',
  'workspace',
  'task',
  'profile',
  'summary',
  'postmortem',
];

export const MemoryPanel: FC = () => {
  const [data, setData] = useState<MemoryPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [content, setContent] = useState('');
  const [memoryType, setMemoryType] = useState('preference');
  const [confidence, setConfidence] = useState(0.9);
  const [saving, setSaving] = useState(false);
  const { notice, notify, cleanup } = useNotify();

  const load = async () => {
    setLoading(true);
    setError('');
    try {
      await whenBackendReady();
      setData(await api.getMemory());
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void load(); }, []);

  const add = async () => {
    if (!content.trim()) return;
    setSaving(true);
    cleanup();
    try {
      await api.createMemory({
        content: content.trim(),
        type: memoryType,
        confidence,
        source_ref: 'config-center',
      });
      setContent('');
      notify('ok', '已写入长期记忆');
      setData(await api.getMemory());
    } catch (e) {
      notify('warn', String(e));
    } finally {
      setSaving(false);
    }
  };

  const remove = async (record: MemoryRecordData) => {
    if (!window.confirm(`删除这条长期记忆？\n\n${record.content}`)) return;
    await api.deleteMemory(record.memory_id);
    setData(await api.getMemory());
  };

  const approve = async (record: MemoryRecordData) => {
    await api.approveMemory(record.memory_id);
    notify('ok', '已批准该复盘记忆');
    setData(await api.getMemory());
  };

  const setEnabled = async (record: MemoryRecordData, enabled: boolean) => {
    const disabledIds = new Set(data?.policy.disabled_ids ?? []);
    if (enabled) disabledIds.delete(record.memory_id);
    else disabledIds.add(record.memory_id);
    const result = await api.updateMemoryPolicy({ disabled_ids: [...disabledIds] });
    setData(prev => prev ? { ...prev, policy: { ...prev.policy, ...result.policy } } : prev);
  };

  return (
    <PanelShell loading={loading} error={error}>
      <InfoLine>
        长期记忆只在显式记录或用户明确说“记住/以后请…”时写入；注入前统一经过
        confidence / TTL / consent / 禁用策略。
      </InfoLine>
      {notice && <Notify kind={notice.kind}>{notice.text}</Notify>}

      <Section title="新增记忆">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 140px 90px auto', gap: 8 }}>
          <input
            style={inputStyle}
            value={content}
            placeholder="例如：以后请用中文、先给结论"
            onChange={e => setContent(e.target.value)}
          />
          <select style={selectStyle} value={memoryType} onChange={e => setMemoryType(e.target.value)}>
            {MEMORY_TYPES.map(type => <option key={type} value={type}>{type}</option>)}
          </select>
          <input
            style={inputStyle}
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={confidence}
            onChange={e => setConfidence(Number(e.target.value))}
          />
          <button
            style={{ ...btnPrimary, ...(saving || !content.trim() ? setBtnDisabled : {}) }}
            disabled={saving || !content.trim()}
            onClick={add}
          >
            添加
          </button>
        </div>
      </Section>

      <Section title={`记忆记录（${data?.records.length ?? 0}）`}>
        {(data?.records.length ?? 0) === 0 && (
          <div style={{ fontSize: 12, color: 'var(--color-text-tertiary)', padding: '10px 0' }}>
            暂无长期记忆。
          </div>
        )}
        {(data?.records ?? []).map(record => (
          <div key={record.memory_id} style={{
            border: '1px solid var(--color-border)',
            borderRadius: 6,
            padding: '8px 10px',
            marginBottom: 8,
            opacity: record.enabled ? 1 : 0.55,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 12, fontWeight: 600, flex: 1 }}>{record.content}</span>
              <button
                style={btnBase}
                onClick={() => void setEnabled(record, !record.enabled)}
              >
                {record.enabled ? '禁用' : '启用'}
              </button>
              {record.status === 'candidate' && (
                <button style={btnPrimary} onClick={() => void approve(record)}>
                  批准
                </button>
              )}
              <button
                style={{ ...btnBase, color: 'var(--color-danger)' }}
                onClick={() => void remove(record)}
              >
                删除
              </button>
            </div>
            <div style={{ marginTop: 4, fontSize: 10, color: 'var(--color-text-tertiary)' }}>
              {record.type} · {record.status || 'active'} · confidence {record.confidence.toFixed(2)} · source {record.source_ref || '-'}
            </div>
          </div>
        ))}
      </Section>

      <InfoLine>存储文件：{data?.path || '-'}</InfoLine>
    </PanelShell>
  );
};

export default MemoryPanel;
