/**
 * DocPanel.tsx — 右侧工作台「文档」面板（对话中心化 L2）。
 *
 * 聚焦显示当前对话绑定的文档（thread.docId，SSE doc_section 事件归因）：
 * 文档下拉（无绑定时可手动选）→ 章节列表（badge 打勾）→ 章节编辑器。
 * 单列窄面板布局，复用 /api/creation/*；全量「+新文档/大纲/导出」留在全屏
 * WriterView（未被路由引用时此处为功能入口）。
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { api, whenBackendReady, type CreationDoc, type CreationDocMeta } from '../api';
import { badge, colTitle } from './WriterView';

const btnStyle: React.CSSProperties = {
  padding: '4px 10px', borderRadius: 6, fontSize: 13,
  border: '1px solid var(--color-border)', background: 'var(--color-surface)',
  display: 'inline-flex', alignItems: 'center', gap: 4,
};
const primaryBtn: React.CSSProperties = {
  ...btnStyle, background: 'var(--color-primary)',
  borderColor: 'var(--color-primary)', color: '#fff',
};

interface Props {
  /** 当前对话绑定文档（SSE 归因）；空 → 面板回落列表视图。 */
  docId?: string;
  writingDir?: string;
}

export function DocPanel({ docId, writingDir }: Props) {
  const [docs, setDocs] = useState<CreationDocMeta[]>([]);
  const [activeDocId, setActiveDocId] = useState<string | null>(docId ?? null);
  const [doc, setDoc] = useState<CreationDoc | null>(null);
  const [selectedSection, setSelectedSection] = useState<string | null>(null);
  const [editContent, setEditContent] = useState('');
  const [saving, setSaving] = useState(false);
  const [savedFlash, setSavedFlash] = useState('');
  const [error, setError] = useState('');
  const mounted = useRef(true);

  // 跟随对话绑定：docId prop 变化 → 聚焦该文档
  useEffect(() => {
    if (docId) setActiveDocId(docId);
  }, [docId]);

  const refreshList = useCallback(async () => {
    await whenBackendReady();
    if (!mounted.current) return;
    try {
      const r = await api.listCreationDocs();
      if (mounted.current) { setDocs(r.docs); setError(''); }
    } catch { setError('文档列表加载失败'); }
  }, []);

  const loadDoc = useCallback(async (id: string) => {
    try {
      const d = await api.getCreationDoc(id);
      if (!mounted.current) return;
      setDoc(d);
      setSelectedSection(prev =>
        (prev && d.outline.some(o => o.section_id === prev) ? prev : d.outline[0]?.section_id ?? null));
      return d;
    } catch {
      if (mounted.current) setError('文档加载失败');
      return null;
    }
  }, []);

  useEffect(() => { mounted.current = true; void refreshList(); return () => { mounted.current = false; }; }, [refreshList]);
  useEffect(() => { if (writingDir !== undefined) void refreshList(); }, [writingDir, refreshList]);
  useEffect(() => { if (!activeDocId) return; void loadDoc(activeDocId); }, [activeDocId, loadDoc]);

  // 会话中 agent 正在写 → 每 5s 轻量刷新章节进度（不覆盖编辑区）
  useEffect(() => {
    if (!activeDocId) return;
    const timer = setInterval(async () => {
      const d = await api.getCreationDoc(activeDocId).catch(() => null);
      if (!mounted.current || !d) return;
      setDoc(d);
      setEditContent(prev =>
        selectedSection && d.sections_content[selectedSection] !== undefined && prev === ''
          ? d.sections_content[selectedSection]
          : prev
      );
    }, 5000);
    return () => clearInterval(timer);
  }, [activeDocId, selectedSection]);

  const selectSection = useCallback((sid: string) => {
    setSelectedSection(sid);
    setEditContent(doc?.sections_content?.[sid] ?? '');
    setSavedFlash('');
  }, [doc]);

  const handleSave = useCallback(async () => {
    if (!activeDocId || !selectedSection) return;
    setSaving(true); setError('');
    try {
      await api.writeCreationSection(activeDocId, selectedSection, editContent);
      setSavedFlash('已保存');
      setTimeout(() => setSavedFlash(''), 1600);
      void loadDoc(activeDocId);
    } catch { setError('保存失败'); }
    finally { setSaving(false); }
  }, [activeDocId, selectedSection, editContent, loadDoc]);

  const outline = doc?.outline ?? [];
  const activeSection = outline.find(o => o.section_id === selectedSection) ?? null;

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* 头：文档选择 + 状态 */}
      <div style={colTitle}>
        <span>📄 文档{docId ? ' · 对话绑定' : ''}</span>
        {docId && (
          <span style={{ marginLeft: 8, fontSize: 11, color: 'var(--color-text-tertiary)', fontFamily: 'var(--font-mono)' }}>
            {docId.slice(0, 10)}
          </span>
        )}
      </div>
      <div style={{ padding: '6px 10px', borderBottom: '1px solid var(--color-border)', display: 'flex', gap: 6, alignItems: 'center' }}>
        <select
          value={activeDocId ?? ''}
          onChange={e => setActiveDocId(e.target.value || null)}
          style={{ flex: 1, padding: '4px 8px', fontSize: 13, borderRadius: 6, border: '1px solid var(--color-border)' }}
        >
          <option value="">（未绑定 · 手动选择）</option>
          {docs.map(d => <option key={d.doc_id} value={d.doc_id}>{d.title || d.doc_id}</option>)}
        </select>
        <button style={btnStyle} onClick={() => void refreshList()} title="刷新">⟳</button>
      </div>
      {error && <div style={{ padding: '4px 12px', fontSize: 12, color: 'var(--color-danger)', background: '#fdecea' }}>{error}</div>}

      {!doc || outline.length === 0 ? (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-text-tertiary)', fontSize: 13, padding: 16, textAlign: 'center' }}>
          {docs.length === 0 ? '暂无文档。在对话里让 agent「写一篇…」即可开始。' : '该文档没有章节，或在对话里让 agent 规划大纲。'}
        </div>
      ) : (
        <>
          {/* 章节列表 */}
          <div style={{ flexShrink: 0, maxHeight: '38%', overflow: 'auto', padding: 6, borderBottom: '1px solid var(--color-border)' }}>
            {outline.map((o, i) => (
              <div
                key={o.section_id}
                onClick={() => selectSection(o.section_id)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  padding: '5px 8px', borderRadius: 6, cursor: 'pointer', marginBottom: 2, fontSize: 13,
                  background: o.section_id === selectedSection ? 'var(--color-primary-light)' : 'transparent',
                }}
              >
                <span style={{ color: 'var(--color-text-tertiary)', fontSize: 12 }}>{i + 1}</span>
                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {o.title || o.section_id}
                </span>
                <span style={badge(o.status)}>{o.status === 'done' ? '✓' : o.status === 'writing' ? '…' : ''}</span>
              </div>
            ))}
          </div>

          {/* 章节编辑器 */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ ...colTitle, display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {activeSection ? activeSection.title : doc.title}
              </span>
              <div style={{ flex: 1 }} />
              {savedFlash && <span style={{ fontSize: 12, color: 'var(--color-success)' }}>{savedFlash}</span>}
              <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>{editContent.trim().split(/\s+/).filter(Boolean).length} 词</span>
            </div>
            <textarea
              value={editContent}
              onChange={e => setEditContent(e.target.value)}
              placeholder="章节 Markdown…（agent 写作进度每 5s 刷新）"
              spellCheck={false}
              style={{
                flex: 1, border: 'none', resize: 'none', outline: 'none',
                padding: 12, fontSize: 13, lineHeight: 1.7,
                fontFamily: 'var(--font-mono)',
                background: 'var(--color-surface)', color: 'var(--color-text)',
              }}
            />
            <div style={{ padding: '6px 12px', borderTop: '1px solid var(--color-border)', display: 'flex', alignItems: 'center', gap: 8 }}>
              <button style={primaryBtn} onClick={() => void handleSave()} disabled={saving || !activeSection}>
                {saving ? '保存中…' : '💾 保存章节'}
              </button>
              <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>
                Markdown 保存后同步到 doc；导出 docx 见全屏工作区。
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}