/**
 * ProjectPathPicker.tsx — 本地目录选择器（项目路径 / 实验根选择）。
 *
 * 通过 /api/workspace/browse 逐级浏览本地文件系统（空 path → 盘符列表），
 * 输入框可直接键入绝对路径。确认后调用 onPick(path)（由父组件调
 * /api/settings 持久化）。桌面本地应用场景，刻意允许浏览根外目录。
 *
 * ponytail: 无外部库；inline styles + CSS 变量。
 */

import { type FC, useCallback, useEffect, useState } from 'react';
import { api } from '../api';

interface Entry { name: string; is_dir: boolean }

interface Props {
  label: string;                    // e.g. "项目路径" / "实验根目录"
  hint?: string;                    // 选中后的效果说明
  value: string;                    // 当前值（绝对路径或 ''）
  allowClear?: boolean;             // true → 显示「清除」按钮（回到默认）
  onPick: (path: string) => void;   // onPick('') = 清除
  onClose: () => void;
}

const norm = (p: string) => (p || '').replace(/\\/g, '/');
const join = (a: string, b: string) => {
  const an = norm(a);
  return an ? (an.endsWith('/') ? an + b : an + '/' + b) : b;
};
/** 上一级目录；盘符根（如 C:/）→ ''（显示盘符列表）。 */
const parentOf = (p: string): string => {
  const t = norm(p).replace(/\/+$/, '');
  if (t.length <= 2) return '';
  const i = t.lastIndexOf('/');
  if (i < 0) return '';
  const up = t.slice(0, i);
  return /^[a-z]:$/i.test(up) ? up + '/' : up;
};

const panelStyle: React.CSSProperties = {
  position: 'fixed', top: 'var(--topbar-h)', left: '50%', transform: 'translateX(-50%)',
  width: 460, maxWidth: '92vw',
  background: 'var(--color-surface)', border: '1px solid var(--color-border)',
  borderRadius: 8, boxShadow: '0 8px 28px rgba(0,0,0,0.18)', zIndex: 100,
  fontSize: 13, color: 'var(--color-text)',
};

const rowBtn: React.CSSProperties = {
  padding: '3px 8px', borderRadius: 4, border: '1px solid var(--color-border)',
  background: 'transparent', fontSize: 12, cursor: 'pointer',
};
const primaryBtn: React.CSSProperties = {
  ...rowBtn, background: 'var(--color-primary)', borderColor: 'var(--color-primary)',
  color: '#fff',
};

export const ProjectPathPicker: FC<Props> = ({ label, hint, value, allowClear, onPick, onClose }) => {
  const [path, setPath] = useState(value);
  const [browsePath, setBrowsePath] = useState('');
  const [entries, setEntries] = useState<Entry[]>([]);
  const [error, setError] = useState('');

  const load = useCallback((p: string) => {
    api.browseDir(p)
      .then(r => { setEntries(r.data.entries); setError(''); })
      .catch(e => setError(String(e)));
  }, []);

  useEffect(() => { load(''); }, [load]);   // 初始：盘符列表（浏览器 dev 兜底用）

  const go = (p: string) => { setBrowsePath(p); setPath(norm(p)); load(p); };

  // Electron 下优先用原生「选择文件夹」对话框（与上传文件同款原生对话框）；
  // 选定后直接生效（onPick），不需要再点确认——和上传文件「选完即用」一致。
  const canNative = !!window.electronAPI?.selectDirectory;
  const nativeBrowse = useCallback(async () => {
    try {
      const p = await window.electronAPI!.selectDirectory?.(path || undefined);
      if (p) onPick(norm(p));
    } catch { /* 忽略取消/异常 */ }
  }, [path, onPick]);

  return (
    <div style={panelStyle} onClick={e => e.stopPropagation()}>
      {/* 头部 */}
      <div style={{
        padding: '8px 12px', display: 'flex', alignItems: 'center', gap: 8,
        borderBottom: '1px solid var(--color-border)', background: 'var(--color-inset)',
        borderTopLeftRadius: 8, borderTopRightRadius: 8,
      }}>
        <span style={{ fontWeight: 600 }}>{label}</span>
        {hint && <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>{hint}</span>}
        <div style={{ flex: 1 }} />
        <button style={rowBtn} onClick={onClose}>✕</button>
      </div>

      {/* 输入 + 确认 */}
      <div style={{ padding: '10px 12px', display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        <input
          value={path}
          onChange={e => setPath(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') onPick(norm(path)); }}
          placeholder="输入绝对路径，或点击下方「选择文件夹…」"
          style={{ flex: 1, minWidth: 200, padding: '4px 8px', fontSize: 13, fontFamily: 'var(--font-mono)' }}
          spellCheck={false}
        />
        <button style={primaryBtn} onClick={() => onPick(norm(path))}>选择当前目录</button>
        {allowClear && (
          <button style={rowBtn} onClick={() => onPick('')} title="恢复默认（项目根=代码根）">清除</button>
        )}
      </div>

      {/* 浏览区：Electron → 原生「选择文件夹」对话框；浏览器 dev → 目录树兜底 */}
      {canNative ? (
        <div style={{ padding: '0 12px 10px' }}>
          <button style={primaryBtn} onClick={() => void nativeBrowse()}>📂 选择文件夹…</button>
          <div style={{ marginTop: 6, fontSize: 11, color: 'var(--color-text-tertiary)' }}>
            通过系统对话框选择目录（与上传文件同款），选定后立即生效。
          </div>
        </div>
      ) : (
      <div style={{ padding: '0 12px 10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
          <button
            style={{ ...rowBtn, opacity: browsePath ? 1 : 0.4 }}
            onClick={() => go(parentOf(browsePath))}
            disabled={!browsePath}
            title="上一级"
          >↑</button>
          <button style={rowBtn} onClick={() => go('')} title="盘符">盘符</button>
          <span style={{
            flex: 1, fontSize: 11, color: 'var(--color-text-secondary)',
            fontFamily: 'var(--font-mono)', overflow: 'hidden', textOverflow: 'ellipsis',
            whiteSpace: 'nowrap', textAlign: 'right',
          }}>
            {browsePath ? norm(browsePath) : '本地磁盘'}
          </span>
        </div>
        {error && <div style={{ padding: 4, fontSize: 12, color: 'var(--color-danger)' }}>⚠️ {error}</div>}
        {!error && entries.length === 0 && (
          <div style={{ padding: 8, fontSize: 12, color: 'var(--color-text-tertiary)' }}>（无子目录）</div>
        )}
        <div style={{ maxHeight: 240, overflowY: 'auto', border: '1px solid var(--color-border)', borderRadius: 6 }}>
          {entries.map(e => {
            const full = join(browsePath, e.name);
            return (
              <button
                key={e.name}
                onClick={() => go(full)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 6, width: '100%', textAlign: 'left',
                  padding: '5px 8px', border: 'none', borderBottom: '1px solid var(--color-border)',
                  background: norm(path) === norm(full) ? 'var(--color-primary-light)' : 'transparent',
                  cursor: 'pointer', fontSize: 12, fontFamily: 'var(--font-mono)',
                }}
              >
                <span>📁</span>
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.name}</span>
              </button>
            );
          })}
        </div>
      </div>
      )}
    </div>
  );
};

export default ProjectPathPicker;