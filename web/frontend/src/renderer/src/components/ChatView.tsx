/**
 * ChatView.tsx — MainContent inner: message list + input bar.
 *
 * Wires SSE streaming from api.ts into the reducer/state. Tool calls become
 * collapsible steps (MessageSteps) rather than a single status line.
 */

import { type FC, useCallback, useRef } from 'react';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { api, streamChat, type AgentApproval } from '../api';
import type { AgentMode, ApprovalRequest, CompletionReportData, Message, PlanStep, PlanVerdictData, ToolStep, WorkNote } from '../state';

interface Props {
  threadId: string;
  messages: Message[];
  isStreaming: boolean;
  mode: AgentMode;
  onModeChange: (m: AgentMode) => void;
  onAddMessage: (msg: Message) => void;
  onAppendToken: (msgId: string, token: string) => void;
  onReplaceMessageContent: (msgId: string, content: string) => void;
  onFinishMessage: (msgId: string) => void;
  onAbortMessage: (msgId: string) => void;
  onToolStart: (msgId: string, step: ToolStep) => void;
  onToolEnd: (msgId: string, toolCallId: string, patch: Partial<ToolStep>, name?: string, threadId?: string) => void;
  onPlan: (msgId: string, plan: PlanStep[]) => void;
  onPlanStep: (msgId: string, stepId: string, patch: Partial<PlanStep>) => void;
  onPlanProgress: (msgId: string, done: number, total: number) => void;
  onPlanVerify: (msgId: string, verdict: PlanVerdictData) => void;
  onTaskStatus: (msgId: string, report: CompletionReportData) => void;
  onMode: (msgId: string, mode: 'react' | 'plan') => void;
  onSetStreaming: (v: boolean) => void;
  /** 对话中心化：写作/实验内联状态片（doc_section / experiment SSE）。 */
  onWorkNote: (msgId: string, note: WorkNote) => void;
  /** Graph interrupt: show/clear the pending tool-approval card. */
  onApproval: (msgId: string, approval: ApprovalRequest | null) => void;
  /** 对话中心化：SSE 事件 → 对话绑定（doc_id / project），供右侧工作台跟随。 */
  onThreadBinding: (threadId: string, binding: { docId?: string; project?: string }) => void;
  /** 上传 PDF（输入框内上传图标触发）。 */
  onUpload: (file: File) => void;
}

const containerStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  flex: 1,
  minHeight: 0, // 防止消息内容的最小高度把上方任务区顶出/裁切 —— 任务区是固定组件
  background: 'var(--color-bg)',
};

export const ChatView: FC<Props> = ({
  threadId, messages, isStreaming, mode, onModeChange,
  onAddMessage, onAppendToken, onFinishMessage, onAbortMessage, onToolStart, onToolEnd,
  onReplaceMessageContent,
  onPlan, onPlanStep, onPlanProgress, onPlanVerify, onTaskStatus, onMode, onSetStreaming,
  onWorkNote, onApproval, onThreadBinding, onUpload,
}) => {
  const abortRef = useRef<AbortController | null>(null);

  const handleSend = useCallback((text: string) => {
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      status: 'complete',
      timestamp: new Date().toISOString(),
    };
    onAddMessage(userMsg);

    const sysId = crypto.randomUUID();
    const sysMsg: Message = {
      id: sysId,
      role: 'system',
      content: '',
      status: 'streaming',
      timestamp: new Date().toISOString(),
    };
    onAddMessage(sysMsg);
    onSetStreaming(true);

    abortRef.current = streamChat(text, threadId, mode, {
      onToolStart(id, name, args, parentId, kind) {
        onToolStart(sysId, { id, name, args, parentId, kind, status: 'running' });
      },
      onToolEnd(id, status, result, executionTime, name) {
        onToolEnd(sysId, id, {
          status: status as ToolStep['status'],
          result,
          executionTime,
        }, name, threadId);
      },
      onMode(m, _source) {
        onMode(sysId, m);
      },
      onPlan(steps) {
        onPlan(sysId, steps);
      },
      onPlanStep(id, status, output) {
        onPlanStep(sysId, id, { status, ...(output !== undefined ? { output } : {}) });
      },
      onPlanProgress(done, total) {
        onPlanProgress(sysId, done, total);
      },
      onPlanVerify(v) {
        onPlanVerify(sysId, v as PlanVerdictData);
      },
      onTaskStatus(report) {
        onTaskStatus(sysId, report);
      },
      onApprovalRequired(approval: AgentApproval) {
        onApproval(sysId, approval);
      },
      onDocSection(docId, payload) {
        onWorkNote(sysId, {
          id: crypto.randomUUID(),
          kind: 'doc',
          text: payload.title
            ? `${payload.title}${payload.section_id ? ' · ' + payload.section_id : ''}`
            : docId,
          status: payload.status,
        });
        onThreadBinding(threadId, { docId });
      },
      onExperiment(expId, payload) {
        onWorkNote(sysId, {
          id: crypto.randomUUID(),
          kind: 'experiment',
          text: `${payload.name || expId} · ${payload.status}${payload.exit_code !== null && payload.exit_code !== undefined ? ` (exit ${payload.exit_code})` : ''}`,
          status: payload.status,
        });
        if (payload.project) onThreadBinding(threadId, { project: payload.project });
      },
      onToken(content) {
        onAppendToken(sysId, content);
      },
      onReplaceAnswer(content) {
        onReplaceMessageContent(sysId, content);
      },
      onDone() {
        onFinishMessage(sysId);
      },
      onError(message) {
        onAppendToken(sysId, `\n\n⚠️ 错误: ${message}`);
        onAbortMessage(sysId);
      },
    });
  }, [threadId, mode, onAddMessage, onAppendToken, onReplaceMessageContent, onFinishMessage, onAbortMessage, onToolStart, onToolEnd, onPlan, onPlanStep, onPlanProgress, onPlanVerify, onTaskStatus, onMode, onSetStreaming, onWorkNote, onApproval, onThreadBinding]);

  const handleApprovalDecision = useCallback(async (msgId: string, approved: boolean) => {
    onSetStreaming(true);
    onApproval(msgId, null);
    try {
      const result = await api.resumeAgent(threadId, approved, mode);
      if (result.task_status) onTaskStatus(msgId, result.task_status);
      if (result.approval) {
        onApproval(msgId, result.approval);
      } else {
        if (result.answer) onAppendToken(msgId, result.answer);
        onFinishMessage(msgId);
      }
    } catch (error) {
      onApproval(msgId, null);
      onAppendToken(msgId, `\n\n⚠️ 审批续跑失败: ${String(error)}`);
      onAbortMessage(msgId);
    } finally {
      onSetStreaming(false);
    }
  }, [threadId, mode, onAppendToken, onFinishMessage, onAbortMessage, onApproval, onSetStreaming, onTaskStatus]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    // Find the last streaming sys message and abort it
    const lastSys = [...messages].reverse().find(m => m.status === 'streaming');
    if (lastSys) onAbortMessage(lastSys.id);
    onSetStreaming(false);
  }, [messages, onAbortMessage, onSetStreaming]);

  return (
    <div style={containerStyle}>
      <MessageList
        messages={messages}
        onSuggestion={handleSend}
        onApprovalDecision={handleApprovalDecision}
      />
      <ChatInput isStreaming={isStreaming} mode={mode} onModeChange={onModeChange} onSend={handleSend} onStop={handleStop} onUpload={onUpload} />
    </div>
  );
};
