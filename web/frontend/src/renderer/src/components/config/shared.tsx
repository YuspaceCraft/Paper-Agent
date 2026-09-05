/**
 * shared.tsx — 配置中心板块共用 UI 原语（Toggle / 行 / 分组 / 按钮 / 通知条）。
 * ponytail: inline styles + CSS 变量，与项目其余组件一致。
 */
import { type ReactNode, useState } from 'react';
import type { FC } from 'react';

export const rowStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 10, padding: '8px 0',
  borderBottom: '1px solid var(--color-border)',
};
export const rowBody: React.CSSProperties = { flex: 1, minWidth: 0 };
export const rowTitle: React.CSSProperties = { fontSize: 13, fontWeight: 600 };
export const rowHint: React.CSSProperties = { fontSize: 11, color: 'var(--color-text-tertiary)', marginTop: 2, lineHeight: 1.5 };

export const btnBase: React.CSSProperties = {
  padding: '5px 12px', borderRadius: 6, fontSize: 12, cursor: 'pointer',
  border: '1px solid var(--color-border)', background: 'transparent',
  color: 'var(--color-text)', transition: 'background 0.12s',
};
export const btnPrimary: React.CSSProperties = {
  ...btnBase, background: 'var(--color-primary)', borderColor: 'var(--color-primary)',
  color: '#fff',
};
export const setBtnDisabled: React.CSSProperties = { opacity: 0.5, cursor: 'not-allowed' };

export const selectStyle: React.CSSProperties = {
  padding: '4px 8px', borderRadius: 6, fontSize: 12,
  border: '1px solid var(--color-border)', background: 'var(--color-surface)',
  color: 'var(--color-text)',
};
export const inputStyle: React.CSSProperties = {
  padding: '5px 8px', borderRadius: 6, fontSize: 12,
  border: '1px solid var(--color-border)', background: 'var(--color-surface)',
  color: 'var(--color-text)', width: '100%', boxSizing: 'border-box',
};

export const FieldRow: FC<{ title: string; hint?: ReactNode; control: ReactNode; disable?: boolean }> = ({
  title, hint, control, disable,
}) => (
  <div style={{ ...rowStyle, opacity: disable ? 0.5 : 1 }}>
    <div style={rowBody}>
      <div style={rowTitle}>{title}</div>
      {hint && <div style={rowHint}>{hint}</div>}
    </div>
    <div style={{ flexShrink: 0 }}>{control}</div>
  </div>
);

export const Section: FC<{ title: string; children: ReactNode }> = ({ title, children }) => (
  <section style={{ marginBottom: 18 }}>
    <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--color-text-secondary)', marginBottom: 6, letterSpacing: 0.3 }}>
      {title}
    </div>
    {children}
  </section>
);

export const Toggle: FC<{ checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }> = ({
  checked, onChange, disabled,
}) => (
  <button
    type="button"
    role="switch"
    aria-checked={checked}
    disabled={disabled}
    onClick={() => onChange(!checked)}
    style={{
      width: 36, height: 20, borderRadius: 10, border: 'none', cursor: disabled ? 'not-allowed' : 'pointer',
      background: checked ? 'var(--color-primary)' : 'var(--color-border)',
      position: 'relative', flexShrink: 0, opacity: disabled ? 0.6 : 1,
      transition: 'background 0.15s',
    }}
  >
    <span style={{
      position: 'absolute', top: 2, left: checked ? 18 : 2, width: 16, height: 16, borderRadius: '50%',
      background: '#fff', transition: 'left 0.15s',
    }} />
  </button>
);

/** 分段选择（如亮/暗主题）。 */
export const Segmented: FC<{
  value: string; options: Array<{ value: string; label: string }>;
  onChange: (v: string) => void; disabled?: boolean;
}> = ({ value, options, onChange, disabled }) => (
  <div style={{ display: 'inline-flex', border: '1px solid var(--color-border)', borderRadius: 6, overflow: 'hidden' }}>
    {options.map(o => {
      const active = o.value === value;
      return (
        <button
          key={o.value}
          type="button"
          disabled={disabled}
          onClick={() => onChange(o.value)}
          style={{
            padding: '4px 12px', fontSize: 12, border: 'none', cursor: disabled ? 'not-allowed' : 'pointer',
            background: active ? 'var(--color-primary)' : 'transparent',
            color: active ? '#fff' : 'var(--color-text-secondary)',
          }}
        >
          {o.label}
        </button>
      );
    })}
  </div>
);

/** 保存条：底部左侧失败信息 + 右侧保存按钮。 */
export const SaveBar: FC<{
  onSave: () => void; saving: boolean; error: string; warn?: string;
  saveLabel?: string;
}> = ({ onSave, saving, error, warn, saveLabel = '保存' }) => (
  <div style={{
    display: 'flex', alignItems: 'center', gap: 10,
    marginTop: 14, paddingTop: 10, borderTop: '1px solid var(--color-border)',
  }}>
    {error ? (
      <span style={{ flex: 1, fontSize: 12, color: 'var(--color-danger)', wordBreak: 'break-all' }}>⚠️ {error}</span>
    ) : warn ? (
      <span style={{ flex: 1, fontSize: 11, color: 'var(--color-text-tertiary)' }}>{warn}</span>
    ) : <span style={{ flex: 1 }} />}
    <button
      style={{ ...btnPrimary, ...(saving ? setBtnDisabled : {}) }}
      disabled={saving}
      onClick={onSave}
    >
      {saving ? '保存中…' : saveLabel}
    </button>
  </div>
);

export const Spinner: FC<{ label?: string }> = ({ label = '加载中…' }) => (
  <div style={{ padding: '24px 0', textAlign: 'center', color: 'var(--color-text-tertiary)', fontSize: 13 }}>
    <span className="step-spinner" style={{ display: 'inline-block', marginRight: 8 }} />{label}
  </div>
);

/** 面板通用容器：加载/失败/正常三态。children 可缺省（纯错误态）。 */
export const PanelShell: FC<{
  loading?: boolean; error?: string; children?: ReactNode;
}> = ({ loading, error, children }) => {
  if (loading) return <Spinner />;
  if (error) return (
    <div style={{ padding: '16px 0', color: 'var(--color-danger)', fontSize: 13 }}>
      ⚠️ {error}
      <div style={{ marginTop: 6, fontSize: 12, color: 'var(--color-text-tertiary)' }}>
        请确认后端已启动（后端维护工具表与磁盘配置）。
      </div>
    </div>
  );
  return <>{children}</>;
};

/** 面板内部「带保存」的头部信息条。 */
export const InfoLine: FC<{ children: ReactNode }> = ({ children }) => (
  <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', lineHeight: 1.6, marginBottom: 10 }}>{children}</div>
);

export const useNotify = () => {
  const [notice, setNotice] = useState<{ kind: 'ok' | 'warn'; text: string } | null>(null);
  return {
    notice,
    notify: (kind: 'ok' | 'warn', text: string) => setNotice({ kind, text }),
    cleanup: () => setNotice(null),
  };
};

export const Notify: FC<{ kind: 'ok' | 'warn'; children: ReactNode }> = ({ kind, children }) => (
  <div style={{
    margin: '8px 0', padding: '6px 10px', borderRadius: 6, fontSize: 12,
    background: kind === 'ok' ? 'var(--color-primary-light)' : 'rgba(250,204,21,0.15)',
    color: kind === 'ok' ? 'var(--color-success)' : 'var(--color-warning)',
  }}>
    {children}
  </div>
);