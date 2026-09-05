/**
 * ExperimentPanel.tsx — 实验配置（工作区路径 + 委托 coding agent 配置）。
 *
 * - 路径：复用 ProjectPathPicker / /api/settings（唯一生效来源）。
 * - 委托：delegate_prefer 接线生效（coding.delegate_code_task 读 config_store）；
 *   delegate_timeout / auto_git_commit / manifest_auto_update 本期「持久化+展示，V2 接线」。
 */
import { useCallback, useEffect, useState } from 'react';
import type { FC } from 'react';
import { api, type ExperimentConfig, type Settings } from '../../api';
import { ProjectPathPicker } from '../ProjectPathPicker';
import {
  btnBase, FieldRow, inputStyle, Notify, PanelShell, SaveBar, Section, Segmented, Toggle,
} from './shared';

interface Props {
  settings: Settings | null;
  onUpdatePaths: (patch: { project_path?: string | null; experiments_path?: string | null }) => void;
}

export const ExperimentPanel: FC<Props> = ({ settings, onUpdatePaths }) => {
  const [cfg, setCfg] = useState<ExperimentConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  const [notice, setNotice] = useState('');
  const [picker, setPicker] = useState<'' | 'project' | 'experiments'>('');

  const load = useCallback(() => {
    setLoading(true);
    setError('');
    api.getExperimentConfig()
      .then(c => { setCfg(c); setLoading(false); })
      .catch(e => { setError(String(e.message ?? e)); setLoading(false); });
  }, []);
  useEffect(load, [load]);

  const patch = (p: Partial<ExperimentConfig>) => setCfg(c => (c ? { ...c, ...p } : c));

  const save = async () => {
    if (!cfg) return;
    setSaving(true);
    setNotice('');
    try {
      await api.updateExperimentConfig({
        delegate_prefer: cfg.delegate_prefer,
        delegate_timeout: cfg.delegate_timeout,
        auto_git_commit: cfg.auto_git_commit,
        manifest_auto_update: cfg.manifest_auto_update,
      });
      setNotice('已保存（超时/自动行为在 V2 接线生效；委托方式与路径即时生效）');
    } catch (e) {
      setNotice('');
      setError(String((e as Error)?.message ?? e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <PanelShell loading={loading} error={error}>
      {notice && <Notify kind="ok">{notice}</Notify>}

      <Section title="工作区路径">
        <FieldRow
          title="项目路径（文献问答 + 写作根）"
          hint={settings?.project_path
            ? `写作保存到 {项目路径}/writing（当前：${settings.project_path}/writing）`
            : '未设置：写作保存到 web/workspace/docs'}
          control={(
            <button style={btnBase} onClick={() => setPicker(picker === 'project' ? '' : 'project')}>
              {settings?.project_path ? '已配置 · 修改' : '选择…'}
            </button>
          )}
        />
        <FieldRow
          title="实验根"
          hint={settings ? `实验与子项目存放于此：${settings.experiments_path}` : ''}
          control={(
            <button style={btnBase} onClick={() => setPicker(picker === 'experiments' ? '' : 'experiments')}>
              选择…
            </button>
          )}
        />
        {picker === 'project' && (
          <ProjectPathPicker
            label="项目路径"
            hint="写作保存到 {路径}/writing"
            value={settings?.project_path ?? ''}
            allowClear
            onPick={p => { onUpdatePaths({ project_path: p || null }); setPicker(''); }}
            onClose={() => setPicker('')}
          />
        )}
        {picker === 'experiments' && (
          <ProjectPathPicker
            label="实验根"
            hint="实验项目根目录"
            value={settings?.experiments_path ?? ''}
            onPick={p => { onUpdatePaths({ experiments_path: p || null }); setPicker(''); }}
            onClose={() => setPicker('')}
          />
        )}
      </Section>

      <Section title="委托 Coding Agent">
        <FieldRow
          title="委托通道"
          hint="选 CLI 时跳过 MCP bridge、直接走编码 CLI（claude/codex）；保存后对后续委托立即生效"
          control={(
            <Segmented
              value={cfg?.delegate_prefer ?? 'mcp'}
              options={[{ value: 'mcp', label: 'MCP bridge' }, { value: 'cli', label: 'CLI' }]}
              onChange={v => patch({ delegate_prefer: v as ExperimentConfig['delegate_prefer'] })}
            />
          )}
        />
        <FieldRow
          title="委托超时（秒）"
          hint="delegate_code_task 默认超时（V2 接线，当前 30–36000）"
          control={(
            <input
              type="number" min={30} max={36000} value={cfg?.delegate_timeout ?? 600}
              onChange={e => patch({ delegate_timeout: Math.max(30, Math.min(36000, Number(e.target.value) || 30)) })}
              style={{ ...inputStyle, width: 140 }}
            />
          )}
        />
        <FieldRow
          title="跑完实验自动 git commit"
          hint="实验完成后自动提交一个逻辑检查点（V2 接线）"
          disable
          control={<Toggle checked={cfg?.auto_git_commit ?? false} onChange={v => patch({ auto_git_commit: v })} />}
        />
        <FieldRow
          title="自动维护项目 manifest"
          hint="跑实验/委托后自动更新 project.json 的 changelog（V2 接线）"
          disable
          control={<Toggle checked={cfg?.manifest_auto_update ?? true} onChange={v => patch({ manifest_auto_update: v })} />}
        />
      </Section>

      <SaveBar
        onSave={() => void save()} saving={saving} error={error}
        warn="普通配置项即时生效；实验行为开关待 V2 接线"
        saveLabel="保存实验配置"
      />
    </PanelShell>
  );
};

export default ExperimentPanel;