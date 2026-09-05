/**
 * WriterView.tsx — 论文写作工作区（v10 / Phase B）。
 *
 * 三栏：文档列表 | 章节树（进度徽章）| 章节 Markdown 编辑器。
 * 数据来自 /api/creation/*；写作通常由聊天里的 agent 触发（plan → creator subagent
 * 逐章写 + SSE doc_section），本工作区轮询 activeDoc 让章节树实时打勾。
 *
 * ponytail: 无外部库。inline styles + CSS 变量，与现有组件风格一致。
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { api, whenBackendReady, type CreationDoc, type CreationDocMeta } from '../api';

const toolbarStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 10,
  padding: '8px 16px', flexShrink: 0,
  borderBottom: '1px solid var(--color-border)',
  background: 'var(--color-surface)',
};

const btnStyle: React.CSSProperties = {
  padding: '4px 10px', borderRadius: 6, fontSize: 13,
  border: '1px solid var(--color-border)', background: 'var(--color-surface)',
  display: 'inline-flex', alignItems: 'center', gap: 4,
};
const primaryBtn: React.CSSProperties = {
  ...btnStyle,
  background: 'var(--color-primary)',
  borderColor: 'var(--color-primary)',
  color: '#fff',
};

export const colStyle: React.CSSProperties = {
  display: 'flex', flexDirection: 'column',
  overflow: 'hidden', borderRight: '1px solid var(--color-border)',
};
export const colTitle: React.CSSProperties = {
  padding: '8px 12px', fontSize: 12, fontWeight: 600,
  color: 'var(--color-text-secondary)', flexShrink: 0,
  borderBottom: '1px solid var(--color-border)', background: 'var(--color-inset)',
};

export const badge: (status: string) => React.CSSProperties = (status) => ({
  fontSize: 11, padding: '1px 6px', borderRadius: 10, flexShrink: 0,
  color: status === 'done' ? 'var(--color-success)' : status === 'writing' ? 'var(--color-warning)' : 'var(--color-text-tertiary)',
  background: status === 'done' ? 'rgba(74,190,110,0.15)' : status === 'writing' ? 'rgba(245,158,11,0.15)' : 'var(--color-inset)',
});

export function WriterView({ writingDir }: { writingDir?: string }) {
  const [docs, setDocs] = useState<CreationDocMeta[]>([]);
  const [activeDocId, setActiveDocId] = useState<string | null>(null);
  const [doc, setDoc] = useState<CreationDoc | null>(null);
  const [selectedSection, setSelectedSection] = useState<string | null>(null);
  const [editContent, setEditContent] = useState('');
  const [newTitle, setNewTitle] = useState('');
  const [creating, setCreating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [savedFlash, setSavedFlash] = useState('');
  const [error, setError] = useState('');
  const mounted = useRef(true);

  // ---- 文档列表 ----
  const refreshList = useCallback(async () => {
    await whenBackendReady();  // Electron dev：renderer 先于后端就绪，首请求会 proxy ECONNREFUSED
    if (!mounted.current) return;
    try {
      const r = await api.listCreationDocs();
      if (mounted.current) {
        setDocs(r.docs);
        setError('');
      }
    } catch { setError('列表加载失败（后端未就绪？）'); }
  }, []);

  // ---- 载入文档详情 ----
  const loadDoc = useCallback(async (docId: string) => {
    try {
      const d = await api.getCreationDoc(docId);
      if (!mounted.current) return;
      setDoc(d);
      setSelectedSection(prev => (prev && d.outline.some(o => o.section_id === prev) ? prev : d.outline[0]?.section_id ?? null));
      return d;
    } catch {
      if (mounted.current) setError('文档加载失败');
      return null;
    }
  }, []);

  const refreshActive = useCallback(async () => {
    if (!activeDocId) return;
    const d = await loadDoc(activeDocId);
    if (d) {
      // 章节树徽章实时来自 outline；编辑区内容不被轮询覆盖
      setEditContent(prev =>
        selectedSection && d.sections_content[selectedSection] !== undefined && prev === ''
          ? d.sections_content[selectedSection]
          : prev
      );
    }
  }, [activeDocId, loadDoc, selectedSection]);

  useEffect(() => { mounted.current = true; void refreshList(); return () => { mounted.current = false; }; }, [refreshList]);

  // 项目路径变更 → 写作文档目录跟随切换，重拉列表
  useEffect(() => {
    if (writingDir !== undefined) void refreshList();
  }, [writingDir, refreshList]);

  useEffect(() => {
    if (!activeDocId) return;
    void loadDoc(activeDocId);
  }, [activeDocId, loadDoc]);

  // 会话中 agent 正在写 → 每 5s 轻量刷新章节进度
  useEffect(() => {
    if (!activeDocId) return;
    const timer = setInterval(() => void refreshActive(), 5000);
    return () => clearInterval(timer);
  }, [activeDocId, refreshActive]);

  // 切换章节 → 载入该章内容到编辑区
  const selectSection = useCallback((sid: string) => {
    setSelectedSection(sid);
    setEditContent(doc?.sections_content?.[sid] ?? '');
    setSavedFlash('');
  }, [doc]);

  const handleCreate = useCallback(async () => {
    const title = newTitle.trim();
    if (!title) return;
    setCreating(true); setError('');
    try {
      const r = await api.createCreationDoc(title);
      await refreshList();
      setActiveDocId(r.doc_id);
      setNewTitle('');
    } catch { setError('创建失败'); }
    finally { setCreating(false); }
  }, [newTitle, refreshList]);

  const handleSave = useCallback(async () => {
    if (!activeDocId || !selectedSection) return;
    setSaving(true); setError('');
    try {
      await api.writeCreationSection(activeDocId, selectedSection, editContent);
      setSavedFlash('已保存');
      setTimeout(() => setSavedFlash(''), 1600);
      void refreshActive();
    } catch { setError('保存失败'); }
    finally { setSaving(false); }
  }, [activeDocId, selectedSection, editContent, refreshActive]);

  const handleExport = useCallback(async () => {
    if (!activeDocId) return;
    try { await api.downloadDocx(activeDocId); }
    catch { setError('导出失败'); }
  }, [activeDocId]);

  const activeOutline = doc?.outline ?? [];
  const activeSection = activeOutline.find(o => o.section_id === selectedSection) ?? null;

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* 工具条 */}
      <div style={toolbarStyle}>
        <span style={{ fontWeight: 700, fontSize: 14 }}>✍️ 论文写作</span>
        {writingDir && (
          <span style={{
            fontSize: 11, color: 'var(--color-text-tertiary)',
            fontFamily: 'var(--font-mono)', overflow: 'hidden',
            textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 260,
          }} title="写作文档保存目录">📁 {writingDir}</span>
        )}
        <input
          value={newTitle}
          onChange={e => setNewTitle(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') void handleCreate(); }}
          placeholder="新文档标题…"
          style={{ width: 200, padding: '4px 10px', fontSize: 13 }}
        />
        <button style={primaryBtn} onClick={() => void handleCreate()} disabled={creating}>
          {creating ? '创建中…' : '＋ 新文档'}
        </button>
        <button style={btnStyle} onClick={() => void refreshList()}>⟳ 刷新</button>
        <div style={{ flex: 1 }} />
        <button style={btnStyle} onClick={() => void handleExport()} disabled={!activeDocId}>⤓ 导出 docx</button>
      </div>

      {error && (
        <div style={{ padding: '4px 16px', fontSize: 12, color: 'var(--color-danger)', background: '#fdecea' }}>{error}</div>
      )}

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* 左：文档列表 */}
        <div style={{ ...colStyle, width: 220, flexShrink: 0 }}>
          <div style={colTitle}>文档</div>
          <div style={{ flex: 1, overflow: 'auto', padding: 6 }}>
            {docs.length === 0 && (
              <div style={{ padding: 12, fontSize: 12, color: 'var(--color-text-tertiary)' }}>
                暂无文档。让 agent「写一篇…」或点击新建。
              </div>
            )}
            {docs.map(d => (
              <div
                key={d.doc_id}
                onClick={() => setActiveDocId(d.doc_id)}
                style={{
                  padding: '8px 10px', borderRadius: 6, cursor: 'pointer', marginBottom: 4,
                  background: d.doc_id === activeDocId ? 'var(--color-primary-light)' : 'transparent',
                  border: '1px solid transparent',
                  borderColor: d.doc_id === activeDocId ? 'var(--color-primary)' : 'transparent',
                }}
              >
                <div style={{ fontSize: 13, fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {d.title || d.doc_id}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 3, fontSize: 11, color: 'var(--color-text-tertiary)' }}>
                  <span>{d.n_sections} 章</span>
                  <span>·</span>
                  <span style={badge(d.status)}>{d.status === 'done' ? '完成' : d.status === 'writing' ? '写作中' : '未开始'}</span>
                  <span>·</span>
                  <span>{d.updated_at?.slice(5, 16) ?? ''}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 中：章节树 */}
        <div style={{ ...colStyle, width: 230, flexShrink: 0 }}>
          <div style={colTitle}>章节（{activeOutline.length}）</div>
          <div style={{ flex: 1, overflow: 'auto', padding: 6 }}>
            {!doc && <div style={{ padding: 12, fontSize: 12, color: 'var(--color-text-tertiary)' }}>选择文档查看章节</div>}
            {doc && activeOutline.length === 0 && (
              <div style={{ padding: 12, fontSize: 12, color: 'var(--color-text-tertiary)' }}>
                大纲为空。可在聊天里让 agent 规划章节。
              </div>
            )}
            {activeOutline.map((o, i) => (
              <div
                key={o.section_id}
                onClick={() => selectSection(o.section_id)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  padding: '6px 8px', borderRadius: 6, cursor: 'pointer', marginBottom: 2,
                  background: o.section_id === selectedSection ? 'var(--color-primary-light)' : 'transparent',
                  fontSize: 13,
                }}
              >
                <span style={{ color: 'var(--color-text-tertiary)', fontSize: 12 }}>{i + 1}</span>
                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {o.title || o.section_id}
                </span>
                <span style={badge(o.status)}>
                  {o.status === 'done' ? '✓' : '…'}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* 右：章节编辑器 */}
        <div style={{ ...colStyle, flex: 1, borderRight: 'none' }}>
          <div style={{ ...colTitle, display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>{activeSection ? activeSection.title : (doc?.title ?? '')}</span>
            {activeSection && activeSection.cites.length > 0 && (
              <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>参考 {activeSection.cites.join('、')}</span>
            )}
            <div style={{ flex: 1 }} />
            {savedFlash && <span style={{ fontSize: 12, color: 'var(--color-success)' }}>{savedFlash}</span>}
            <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>{editContent.trim() ? editContent.trim().split(/\s+/).length : 0} 词</span>
          </div>

          {activeSection ? (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              <textarea
                value={editContent}
                onChange={e => setEditContent(e.target.value)}
                placeholder="在此输入或编辑章节的 Markdown 内容…（agent 写作进度每 5s 刷新）"
                spellCheck={false}
                style={{
                  flex: 1, border: 'none', resize: 'none', outline: 'none',
                  padding: 14, fontSize: 13.5, lineHeight: 1.7,
                  fontFamily: 'var(--font-mono)',
                  background: 'var(--color-surface)', color: 'var(--color-text)',
                }}
              />
              <div style={{ padding: '8px 14px', borderTop: '1px solid var(--color-border)', display: 'flex', alignItems: 'center', gap: 8 }}>
                <button style={primaryBtn} onClick={() => void handleSave()} disabled={saving}>
                  {saving ? '保存中…' : '💾 保存章节'}
                </button>
                <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>
                  Markdown 保存后同步到全局文档；导出的 docx 按 `#/##` 标题层级渲染。
                </span>
              </div>
            </div>
          ) : (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-text-tertiary)', fontSize: 13 }}>
              {doc ? (doc.outline.length ? '选择一个章节开始编辑' : '当前文档没有章节') : '← 选择或新建一个文档'}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}