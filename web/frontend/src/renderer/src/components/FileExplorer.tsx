/**
 * FileExplorer.tsx — workspace file browser (coding-agent style).
 *
 * Lists the workspace directory tree and shows file contents inline.
 * Reuses Markdown.tsx for .md/.markdown; other text files render as <pre>.
 * Backend guards path escape via /api/workspace/*.
 */

import { type FC, useEffect, useState, useCallback } from 'react';
import { api, whenBackendReady } from '../api';
import { Markdown } from './Markdown';

interface Entry {
  name: string;
  is_dir: boolean;
  size: number | null;
}

interface Props {
  open: boolean;
}

const join = (a: string, b: string) => (a === '.' || a === '' ? b : `${a}/${b}`);
const parentOf = (p: string): string | null => {
  if (p === '.' || p === '') return null;
  const i = p.lastIndexOf('/');
  if (i < 0) return '.';
  return p.slice(0, i) || '.';
};
const isMarkdown = (name: string) => /\.(md|markdown)$/i.test(name);
const fmtSize = (n: number | null) =>
  n === null ? '' : n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`;

const panelStyle: React.CSSProperties = {
  width: 'var(--right-w)',
  background: 'var(--color-surface)',
  borderLeft: '1px solid var(--color-border)',
  display: 'flex',
  flexDirection: 'column',
  flexShrink: 0,
  overflow: 'hidden',
};

export const FileExplorer: FC<Props> = ({ open }) => {
  const [cwd, setCwd] = useState('.');
  const [entries, setEntries] = useState<Entry[]>([]);
  const [selected, setSelected] = useState<{ path: string; content: string; is_binary: boolean } | null>(null);
  const [error, setError] = useState('');

  const loadDir = useCallback((path: string) => {
    api.listWorkspace(path)
      .then(r => { setEntries(r.data.entries); setError(''); })
      .catch(e => setError(String(e)));
  }, []);

  // Gate the initial load on backend readiness (Electron): mounting in the
  // first second fires before uvicorn is up, and a one-shot fetch would leave
  // the tree stuck in an error state forever. Once resolved this is a no-op.
  useEffect(() => {
    if (!open) return;
    let alive = true;
    whenBackendReady().then(() => { if (alive) loadDir(cwd); });
    return () => { alive = false; };
  }, [cwd, open, loadDir]);

  const openFile = useCallback((path: string) => {
    api.readWorkspaceFile(path)
      .then(r => setSelected({ path, content: r.data.content, is_binary: r.data.is_binary }))
      .catch(e => setError(String(e)));
  }, []);

  if (!open) return null;

  const up = parentOf(cwd);

  return (
    <aside style={panelStyle}>
      {/* breadcrumb toolbar */}
      <div style={{
        padding: '8px 10px', borderBottom: '1px solid var(--color-border)',
        display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0,
      }}>
        <button
          onClick={() => { if (up !== null) { setCwd(up); setSelected(null); } }}
          disabled={up === null}
          title="上一级"
          style={toolBtn(up === null)}
        >↑</button>
        <button onClick={() => { loadDir(cwd); }} title="刷新" style={toolBtn(false)}>↻</button>
        <span style={{
          flex: 1, fontSize: 12, color: 'var(--color-text-secondary)',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          fontFamily: 'monospace',
        }}>
          {cwd === '.' ? '/' : cwd}
        </span>
      </div>

      {/* directory listing */}
      <div style={{ flex: '0 0 auto', maxHeight: '45%', overflowY: 'auto', borderBottom: '1px solid var(--color-border)' }}>
        {error && <div style={{ padding: 8, fontSize: 12, color: 'var(--color-danger)' }}>⚠️ {error}</div>}
        {entries.length === 0 && !error && (
          <div style={{ padding: 16, fontSize: 12, color: 'var(--color-text-secondary)' }}>(空目录)</div>
        )}
        {entries.map(e => (
          <button
            key={e.name}
            onClick={() => (e.is_dir ? setCwd(join(cwd, e.name)) : openFile(join(cwd, e.name)))}
            style={{
              display: 'flex', alignItems: 'center', gap: 6, width: '100%',
              padding: '5px 10px', border: 'none',
              cursor: 'pointer', fontSize: 12, textAlign: 'left',
              borderLeft: selected?.path === join(cwd, e.name)
                ? '3px solid var(--color-primary)' : '3px solid transparent',
              background: selected?.path === join(cwd, e.name) ? 'var(--color-inset)' : 'transparent',
            }}
          >
            <span>{e.is_dir ? '📁' : '📄'}</span>
            <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.name}</span>
            {!e.is_dir && <span style={{ fontSize: 10, color: 'var(--color-text-secondary)' }}>{fmtSize(e.size)}</span>}
          </button>
        ))}
      </div>

      {/* content viewer */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {selected ? (
          <div>
            <div style={{
              padding: '6px 10px', fontSize: 11, color: 'var(--color-text-secondary)',
              borderBottom: '1px solid var(--color-border)', fontFamily: 'monospace',
            }}>
              {selected.path}
            </div>
            {selected.is_binary ? (
              <div style={{ padding: 16, fontSize: 12, color: 'var(--color-text-secondary)' }}>
                二进制文件，无法预览
              </div>
            ) : isMarkdown(selected.path) ? (
              <div style={{ padding: '4px 10px' }}>
                <Markdown content={selected.content} />
              </div>
            ) : (
              <pre style={{ margin: 0, padding: 10, fontSize: 12, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                {selected.content}
              </pre>
            )}
          </div>
        ) : (
          <div style={{
            padding: 24, fontSize: 12, color: 'var(--color-text-secondary)', textAlign: 'center',
          }}>
            📂 点击文件查看内容
          </div>
        )}
      </div>
    </aside>
  );
};

const toolBtn = (disabled: boolean): React.CSSProperties => ({
  padding: '2px 8px', borderRadius: 4, border: '1px solid var(--color-border)',
  background: '#fff', fontSize: 12, cursor: disabled ? 'not-allowed' : 'pointer',
  opacity: disabled ? 0.4 : 1,
});

export default FileExplorer;
