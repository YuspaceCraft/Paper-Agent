/**
 * MessageSteps.tsx — collapsible tool-call steps for an assistant turn.
 *
 * CowAgent-style: one muted row per tool call (spinner / ✓ / ✗ + name + time),
 * click to expand the call's args and result. Running steps auto-expand.
 */

import { type FC, useState } from 'react';
import type { ToolStep } from '../state';

const TOOL_LABELS: Record<string, string> = {
  search_literature: '🔍 检索文献',
  read_paper_section: '📖 阅读章节',
  get_paper_abstract: '📄 获取摘要',
  get_chunk_context: '📋 获取上下文',
  list_papers: '📚 查找论文',
  lookup_page: '🔖 定位页面',
  paper_discovery: '📚 论文发现',
  paper_reader: '📖 精读',
  paper_ingest: '📥 入库',
  download_paper: '⬇️ 下载 PDF',
  ingest_paper: '📥 入库',
  check_paper: '🔍 本地状态检查',
  check_task_status: '📊 任务状态',
};

// Subagent boundary steps (arxiv/library/ingest) are rendered distinctly from
// the leaf tools nested under them.
const SUBAGENT_LABELS: Record<string, string> = {
  arxiv: '🌐 arXiv 检索',
  library: '📚 本地知识库',
  ingest: '📥 下载 / 入库',
};

function subagentLabel(name: string): string {
  return SUBAGENT_LABELS[name] ?? `🤖 ${name}`;
}

const rowStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  fontSize: 12,
  color: 'var(--color-text-secondary)',
  cursor: 'pointer',
  padding: '2px 0',
  userSelect: 'none',
};

const panelStyle: React.CSSProperties = {
  margin: '4px 0 6px 18px',
  padding: 8,
  borderRadius: 6,
  background: 'var(--color-inset)',
  border: '1px solid var(--color-border)',
  fontSize: 11,
  lineHeight: 1.55,
};

export function labelFor(name: string): string {
  return TOOL_LABELS[name] ?? `🔧 ${name}`;
}

const ToolStepRow: FC<{ step: ToolStep }> = ({ step }) => {
  // Default-collapsed: the row (status + name + time) always stays visible;
  // args/result only show when the user clicks to expand.
  const [expanded, setExpanded] = useState(false);
  const running = step.status === 'running';
  const failed = step.status === 'error';
  const isSubagent = step.kind === 'subagent';

  return (
    <div style={{ marginBottom: 4 }}>
      <div style={rowStyle} onClick={() => setExpanded(v => !v)}>
        {running ? (
          <span className="step-spinner" style={{ flexShrink: 0 }} />
        ) : (
          <span style={{ flexShrink: 0, color: failed ? 'var(--color-danger)' : 'var(--color-success)' }}>
            {failed ? '✗' : '✓'}
          </span>
        )}
        {isSubagent && (
          <span
            style={{
              flexShrink: 0,
              fontSize: 9,
              padding: '0 4px',
              borderRadius: 3,
              background: 'var(--color-primary)',
              color: '#fff',
              opacity: 0.85,
              fontWeight: 600,
              letterSpacing: 0.3,
            }}
          >
            子代理
          </span>
        )}
        <span style={{ fontWeight: 500, color: failed ? 'var(--color-danger)' : isSubagent ? 'var(--color-primary)' : 'inherit' }}>
          {isSubagent ? subagentLabel(step.name) : labelFor(step.name)}
        </span>
        {step.executionTime !== undefined && step.executionTime > 0 && (
          <span style={{ opacity: 0.6 }}>{step.executionTime}s</span>
        )}
        <span
          style={{
            marginLeft: 'auto',
            fontSize: 11,
            opacity: 0.5,
            transform: expanded ? 'rotate(90deg)' : 'none',
            transition: 'transform 0.12s',
          }}
        >
          ▸
        </span>
      </div>

      {expanded && (
        <div style={panelStyle}>
          {step.args && Object.keys(step.args).length > 0 && (
            <div>
              <div style={{ fontWeight: 600, opacity: 0.6, fontSize: 10, textTransform: 'uppercase', marginBottom: 2 }}>
                入参
              </div>
              <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontFamily: 'var(--font-mono)', margin: 0 }}>
                {JSON.stringify(step.args, null, 2)}
              </pre>
            </div>
          )}
          {step.result && (
            <div style={{ marginTop: step.args && Object.keys(step.args).length > 0 ? 8 : 0 }}>
              <div style={{ fontWeight: 600, opacity: 0.6, fontSize: 10, textTransform: 'uppercase', marginBottom: 2 }}>
                {failed ? '错误' : '结果'}
              </div>
              <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: failed ? 'var(--color-danger)' : 'inherit' }}>
                {step.result}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Recursively render a step and its nested children (a subagent's leaf tools).
const StepNode: FC<{ step: ToolStep }> = ({ step }) => (
  <div>
    <ToolStepRow step={step} />
    {step.children && step.children.length > 0 && (
      <div
        style={{
          marginLeft: 14,
          paddingLeft: 8,
          borderLeft: '1px solid var(--color-border)',
        }}
      >
        {step.children.map(c => <StepNode key={c.id} step={c} />)}
      </div>
    )}
  </div>
);

export const MessageSteps: FC<{ steps: ToolStep[] }> = ({ steps }) => {
  if (!steps || steps.length === 0) return null;
  return (
    <div>
      {steps.map(s => <StepNode key={s.id} step={s} />)}
    </div>
  );
};

export default MessageSteps;
