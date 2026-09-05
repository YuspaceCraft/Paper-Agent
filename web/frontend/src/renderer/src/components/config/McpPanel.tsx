/**
 * McpPanel.tsx — MCP 配置：读取/重写仓库根 .mcp.json；增删改查 server、
 * 逐个测试连接。保存后由后端 reload_tools 重建工具表。
 */
import { useCallback, useEffect, useState } from 'react';
import type { FC } from 'react';
import { api, type McpInfo, type McpServerInfo } from '../../api';
import {
  btnBase, btnPrimary, FieldRow, InfoLine, inputStyle, Notify, PanelShell, SaveBar,
  Section, selectStyle, setBtnDisabled, Spinner, Toggle,
} from './shared';

interface Draft {
  name: string;
  transport: string;
  command: string;
  args: string;
  url: string;
  env: string;     // key=value 每行一个
  headers: string;
  disabled: boolean;
}

const emptyDraft: Draft = {
  name: '', transport: 'stdio', command: '', args: '', url: '',
  env: '', headers: '', disabled: false,
};

function entryToDraft(e: McpServerInfo): Draft {
  const env = e.env ? Object.entries(e.env).map(([k, v]) => `${k}=${v}`).join('\n') : '';
  const headers = e.headers ? Object.entries(e.headers).map(([k, v]) => `${k}=${v}`).join('\n') : '';
  return {
    name: e.name,
    transport: (e.transport ?? 'stdio') as string,
    command: e.command ?? '',
    args: (e.args ?? []).join(' '),
    url: e.url ?? '',
    env, headers,
    disabled: !!e.disabled,
  };
}

function parseKV(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of text.split('\n')) {
    const t = line.trim();
    if (!t) continue;
    const i = t.indexOf('=');
    if (i > 0) out[t.slice(0, i).trim()] = t.slice(i + 1).trim();
  }
  return out;
}

export const McpPanel: FC = () => {
  const [info, setInfo] = useState<McpInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  const [notice, setNotice] = useState('');
  const [editing, setEditing] = useState<Draft | null>(null);   // 非空 = 表单（添加/编辑）
  const [editName, setEditName] = useState('');                 // 编辑中的原 name（改名用）
  const [tests, setTests] = useState<Record<string, { ok: boolean; tool_count: number; error: string | null }>>({});
  const [testing, setTesting] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true); setError('');
    api.getMcpConfig()
      .then(m => { setInfo(m); setLoading(false); })
      .catch(e => { setError(String(e.message ?? e)); setLoading(false); });
  }, []);
  useEffect(load, [load]);

  const test = async (name: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setTesting(name);
    try {
      const r = await api.testMcpServer(name);
      setTests(t => ({ ...t, [name]: r }));
    } catch (err) {
      setTests(t => ({ ...t, [name]: { ok: false, tool_count: 0, error: String(err) } }));
    } finally {
      setTesting(null);
    }
  };

  const remove = (name: string) => {
    if (!info) return;
    if (!window.confirm(`删除 MCP server「${name}」？保存后 \`.mcp.json\` 随即更新。`)) return;
    setInfo({ ...info, servers: info.servers.filter(s => s.name !== name) });
  };

  const save = async () => {
    if (!info) return;
    setSaving(true); setNotice('');
    try {
      const res = await api.updateMcpConfig(info.servers);
      setInfo({ ...info, servers: res.servers });
      setNotice('已保存并重建工具表（新会话生效）。');
    } catch (err) {
      setError(String((err as Error).message ?? err));
    } finally {
      setSaving(false);
    }
  };

  const submitDraft = () => {
    if (!editing || !info) return;
    const name = editing.name.trim();
    if (!name) { setError('server name 必填'); return; }
    const isHttp = editing.transport !== 'stdio';
    const entry: McpServerInfo = {
      name,
      transport: editing.transport,
      ...(isHttp
        ? { url: editing.url.trim() }
        : { command: editing.command.trim(), args: editing.args.split(/\s+/).filter(Boolean) }),
      ...(editing.env.trim() ? { env: parseKV(editing.env) } : {}),
      ...(editing.headers.trim() ? { headers: parseKV(editing.headers) } : {}),
      disabled: editing.disabled,
      status: 'unknown', tools: 0,
    };
    if (isHttp && !entry.url) { setError('http server 需要 url'); return; }
    if (!isHttp && !entry.command) { setError('stdio server 需要 command'); return; }
    if (info.servers.some(s => s.name === name && name !== editName)) {
      setError(`server「${name}」已存在`); return;
    }
    const next = editName
      ? info.servers.map(s => (s.name === editName ? entry : s))
      : [...info.servers, entry];
    setInfo({ ...info, servers: next });
    setEditing(null); setEditName('');
    setError('');
  };

  if (loading) return <Spinner />;
  if (error && !info) return <PanelShell error={error} />;
  if (!info) return null;

  const totalTools = info.servers.reduce((n, s) => n + (s.tools ?? 0), 0);

  return (
    <div>
      {notice && <Notify kind="ok">{notice}</Notify>}
      <InfoLine>
        配置文件：<span style={{ fontFamily: 'var(--font-mono)' }}>{info.path}</span>
        （仓库根 .mcp.json）。保存后随机重建工具表。
        {info.servers.length} 个 server · {totalTools} 个已加载工具
      </InfoLine>

      <Section title="MCP Servers">
        {info.servers.length === 0 && (
          <div style={{ padding: '10px 0', fontSize: 12, color: 'var(--color-text-tertiary)' }}>
            尚未配置 MCP server。点击下方「添加 Server」。
          </div>
        )}
        {info.servers.map(s => {
          const t = tests[s.name];
          return (
            <div key={s.name} style={{ border: '1px solid var(--color-border)', borderRadius: 8, padding: '8px 10px', marginBottom: 8 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: s.disabled ? 'var(--color-text-tertiary)' : (s.status === 'ok' ? 'var(--color-success)' : 'var(--color-warning)') }} />
                <span style={{ fontWeight: 600, fontFamily: 'var(--font-mono)' }}>{s.name}</span>
                <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>{s.transport}</span>
                <span style={{
                  fontSize: 11, color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)',
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 340,
                }}>
                  {s.command ? `${s.command} ${(s.args ?? []).join(' ')}` : s.url}
                </span>
                {s.disabled && <span style={{ fontSize: 11, color: 'var(--color-danger)' }}>已停用</span>}
                <div style={{ flex: 1 }} />
                {t && (
                  <span style={{ fontSize: 11, color: t.ok ? 'var(--color-success)' : 'var(--color-danger)' }}>
                    {t.ok ? `✓ ${t.tool_count} 工具` : `✗ ${t.error}`}
                  </span>
                )}
                <button style={btnBase} disabled={testing === s.name} onClick={e => void test(s.name, e)}>
                  {testing === s.name ? '测试中…' : '测试'}
                </button>
                <button style={btnBase} onClick={() => { setEditing(entryToDraft(s)); setEditName(s.name); }}>编辑</button>
                <button style={{ ...btnBase, color: 'var(--color-danger)' }} onClick={() => remove(s.name)}>删除</button>
              </div>
              {s.tools > 0 && (
                <div style={{ marginTop: 6, fontSize: 11, color: 'var(--color-text-tertiary)' }}>
                  已加载 <span style={{ fontFamily: 'var(--font-mono)' }}>{s.tools}</span> 个工具
                  （<span style={{ fontFamily: 'var(--font-mono)' }}>{s.name}__…</span>）
                </div>
              )}
            </div>
          );
        })}

        {!editing && (
          <button style={{ ...btnPrimary, ...(saving ? setBtnDisabled : {}) }} disabled={saving} onClick={() => { setEditing({ ...emptyDraft }); setEditName(''); setError(''); }}>
            ＋ 添加 Server
          </button>
        )}

        {editing && (
          <div style={{ marginTop: 10, border: '1px solid var(--color-primary)', borderRadius: 8, padding: 12 }}>
            <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>
              {editName ? `编辑 ${editName}` : '添加 MCP Server'}
            </div>
            <FieldRow title="名称" hint="仅字母/数字/`-`/`_`；工具名以 `{name}__` 前缀" control={(
              <input style={{ ...inputStyle, width: 200 }} value={editing.name}
                onChange={e => setEditing({ ...editing, name: e.target.value.replace(/[^\w.-]/g, '') })} />
            )} />
            <FieldRow title="传输方式" control={(
              <select style={selectStyle} value={editing.transport}
                onChange={e => setEditing({ ...editing, transport: e.target.value })}>
                <option value="stdio">stdio（本地子进程）</option>
                <option value="streamable-http">streamable-http（远程）</option>
                <option value="sse">sse（远程）</option>
              </select>
            )} />
            {editing.transport === 'stdio' ? (
              <>
                <FieldRow title="命令" hint="MCP server 启动命令（支持 ${VAR:-default} 占位）" control={(
                  <input style={{ ...inputStyle, width: 320 }} value={editing.command}
                    onChange={e => setEditing({ ...editing, command: e.target.value })} />
                )} />
                <FieldRow title="参数" hint="空格分隔的参数列表" control={(
                  <input style={{ ...inputStyle, width: 320 }} value={editing.args}
                    onChange={e => setEditing({ ...editing, args: e.target.value })} />
                )} />
                <FieldRow title="环境变量" hint="每行 key=value（可选）" control={(
                  <textarea style={{ ...inputStyle, width: 320, height: 52, resize: 'vertical' }} value={editing.env}
                    onChange={e => setEditing({ ...editing, env: e.target.value })} />
                )} />
              </>
            ) : (
              <>
                <FieldRow title="URL" hint="MCP HTTP 端点" control={(
                  <input style={{ ...inputStyle, width: 320 }} value={editing.url}
                    onChange={e => setEditing({ ...editing, url: e.target.value })} />
                )} />
                <FieldRow title="Headers" hint="每行 key=value（可选）" control={(
                  <textarea style={{ ...inputStyle, width: 320, height: 52, resize: 'vertical' }} value={editing.headers}
                    onChange={e => setEditing({ ...editing, headers: e.target.value })} />
                )} />
              </>
            )}
            <FieldRow title="停用此 server" hint="保留配置但不加载工具" control={(
              <Toggle checked={editing.disabled} onChange={v => setEditing({ ...editing, disabled: v })} />
            )} />
            {error && <div style={{ fontSize: 12, color: 'var(--color-danger)', padding: '4px 0' }}>⚠️ {error}</div>}
            <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
              <button style={btnPrimary} onClick={submitDraft}>{editName ? '确定修改' : '加入列表'}</button>
              <button style={btnBase} onClick={() => { setEditing(null); setEditName(''); setError(''); }}>取消</button>
            </div>
          </div>
        )}
      </Section>

      <SaveBar
        onSave={() => void save()} saving={saving} error={editing ? '' : error}
        warn="保存会重写 .mcp.json 并重建工具表（可能耗时数秒）；新会话生效"
        saveLabel="保存 MCP 配置"
      />
    </div>
  );
};

export default McpPanel;