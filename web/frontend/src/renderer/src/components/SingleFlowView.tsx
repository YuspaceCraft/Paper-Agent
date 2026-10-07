/**
 * SingleFlowView - node-level visualization for one evaluation turn.
 *
 * Only executable LangGraph units are nodes. Prompts, tool surfaces, context,
 * state, observations and results are resources attached to the node that
 * consumes or produces them; they are not graph nodes themselves.
 */

import {
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from 'react';
import type { EvalSingleFlow, EvalFlowItem, EvalTraceEvent } from '../api';

type NodeStatus = 'complete' | 'active' | 'pending' | 'error' | 'skipped';

interface NodeFact {
  label: string;
  value: string;
  tone?: 'default' | 'ok' | 'warn' | 'danger' | 'muted';
}

interface NodeSection {
  label: string;
  value: string;
  tone?: 'default' | 'ok' | 'warn' | 'danger' | 'muted';
}

type ResourceKind =
  | 'query'
  | 'prompt'
  | 'tools'
  | 'context'
  | 'state'
  | 'result'
  | 'tool'
  | 'subagent'
  | 'artifact'
  | 'error';

interface NodeResource {
  id: string;
  kind: ResourceKind;
  label: string;
  role: 'input' | 'output';
  value: unknown;
  tone?: NodeFact['tone'];
}

interface PipelineNode {
  id: string;
  title: string;
  kind: string;
  description: string;
  status: NodeStatus;
  durationMs?: number | null;
  facts?: NodeFact[];
  sections?: NodeSection[];
  resources?: NodeResource[];
  children?: PipelineNode[];
}

interface GraphPlacement {
  id: string;
  node: PipelineNode;
  x: number;
  y: number;
  width: number;
  height: number;
  lane: 'main' | 'branch';
}

interface GraphEdge {
  id: string;
  from: string;
  to: string;
  kind: 'main' | 'branch' | 'loop';
  status: NodeStatus;
}

const mono: CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 10.5,
  lineHeight: 1.55,
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
};

function fmtMs(value?: number | null): string {
  if (value === null || value === undefined) return '-';
  if (value < 1000) return `${Math.round(value)} ms`;
  return `${(value / 1000).toFixed(value >= 10_000 ? 1 : 2)} s`;
}

function fmtNumber(value: unknown, digits = 2): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return '-';
  return String(Math.round(n * 10 ** digits) / 10 ** digits);
}

function short(value: unknown, limit = 900): string {
  const text = typeof value === 'string'
    ? value
    : JSON.stringify(value ?? {}, null, 2);
  return text.length > limit ? `${text.slice(0, limit)}\n... [${text.length - limit} chars omitted]` : text;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function payload(event?: EvalTraceEvent): Record<string, unknown> {
  return asRecord(event?.payload);
}

function statusColor(status: NodeStatus): string {
  switch (status) {
    case 'complete': return 'var(--color-primary)';
    case 'active': return 'var(--color-warning)';
    case 'error': return 'var(--color-danger)';
    case 'skipped': return 'var(--color-text-tertiary)';
    default: return 'var(--color-border)';
  }
}

function statusIcon(status: NodeStatus) {
  if (status === 'active') return <span className="step-spinner" />;
  return <span style={{ fontSize: 12, fontWeight: 800 }}>
    {status === 'complete' ? '✓' : status === 'error' ? '!' : status === 'skipped' ? '·' : '○'}
  </span>;
}

function edgeStatus(source: PipelineNode, target: PipelineNode): NodeStatus {
  if (target.status === 'active') return 'active';
  if (target.status === 'error' || source.status === 'error') return 'error';
  if (target.status === 'complete' || target.status === 'skipped') return 'complete';
  return 'pending';
}

function hasEvent(events: EvalTraceEvent[], type: string, node?: string): boolean {
  return events.some(event =>
    event.event_type === type && (!node || event.node === node));
}

function statusOf(
  complete: boolean,
  active: boolean,
  skipped = false,
): NodeStatus {
  if (complete) return 'complete';
  if (active) return 'active';
  if (skipped) return 'skipped';
  return 'pending';
}

function graphNodeLabel(node: PipelineNode): string {
  const labels: Record<string, string> = {
    input: '输入',
    runtime: '运行时',
    config: '配置',
    analysis: '分析',
    context: '上下文',
    resolution: '解析',
    plan: '计划',
    agent: 'Agent',
    llm: 'LLM',
    tool: '工具',
    subagent: '子 Agent',
    io: '入参',
    gate: '校验',
    execution: '执行',
    parse: '解析',
    citation: '引用',
    window: '窗口',
    answer: '回答',
    turn: '结束',
  };
  return labels[node.kind] ?? node.kind;
}

function flattenNodes(nodes: PipelineNode[]): PipelineNode[] {
  return nodes.flatMap(node => [node, ...flattenNodes(node.children ?? [])]);
}

function lastEvent(
  events: EvalTraceEvent[],
  type: string,
  node?: string,
): EvalTraceEvent | undefined {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (event.event_type === type && (!node || event.node === node)) return event;
  }
  return undefined;
}

function tokenTotal(events: EvalTraceEvent[]): number {
  return events
    .filter(event => event.event_type === 'llm_call')
    .reduce((sum, event) => {
      const tokens = asRecord(payload(event).tokens);
      return sum + Number(tokens.total_tokens ?? 0);
    }, 0);
}

function isEmptyResource(value: unknown): boolean {
  if (value === null || value === undefined || value === '') return true;
  if (Array.isArray(value)) return value.length === 0;
  if (typeof value === 'object') return Object.keys(value as Record<string, unknown>).length === 0;
  return false;
}

function makeNodeResource(
  id: string,
  kind: ResourceKind,
  label: string,
  role: NodeResource['role'],
  value: unknown,
  tone: NodeFact['tone'] = 'default',
): NodeResource | null {
  if (isEmptyResource(value)) return null;
  return { id, kind, label, role, value, tone };
}

function compactResources(resources: NodeResource[]): NodeResource[] {
  const seen = new Set<string>();
  return resources.filter(resource => {
    const key = `${resource.role}:${resource.kind}:${resource.label}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function resourceCount(resource: NodeResource): number {
  if (Array.isArray(resource.value)) return resource.value.length;
  const record = asRecord(resource.value);
  if (Array.isArray(record.steps)) return record.steps.length;
  if (Array.isArray(record.tools)) return record.tools.length;
  return 1;
}

function resourceDurationMs(resource: NodeResource): number | null {
  const value = asRecord(resource.value);
  const duration = Number(value.duration_ms);
  return Number.isFinite(duration) ? duration : null;
}

function resourceKindLabel(kind: ResourceKind): string {
  const labels: Record<ResourceKind, string> = {
    query: '问题',
    prompt: '提示词',
    tools: '工具列表',
    context: '上下文',
    state: '状态',
    result: '执行结果',
    tool: '工具执行结果',
    subagent: '子 Agent',
    artifact: '产物',
    error: '错误',
  };
  return labels[kind];
}

function buildMainGraphNodes(
  flow: EvalSingleFlow,
  isRunning: boolean,
): PipelineNode[] {
  const events = flow.events ?? [];
  const finalSeen = Boolean(lastEvent(events, 'final_answer') || lastEvent(events, 'turn_end'));
  const nodeEvents = (node: string) => events.filter(event => event.node === node);
  const nodeStart = (node: string) => nodeEvents(node).some(event => event.event_type === 'node_start');
  const nodeEnd = (node: string) => nodeEvents(node).some(event => event.event_type === 'node_end');
  const nodeError = (node: string) => nodeEvents(node).some(event =>
    event.event_type === 'node_error' || event.event_type.endsWith('_failed'),
  );

  const scopeFor = (node: string): EvalTraceEvent[] => {
    const starts = events.filter(event =>
      event.event_type === 'node_start' && event.node === node,
    );
    if (!starts.length) return nodeEvents(node);
    const first = Math.min(...starts.map(event => event.seq));
    const ends = events.filter(event =>
      event.event_type === 'node_end' && event.node === node && event.seq >= first,
    );
    const last = ends.length ? Math.max(...ends.map(event => event.seq)) : Number.POSITIVE_INFINITY;
    const direct = new Set(nodeEvents(node).map(event => event.seq));
    return events.filter(event => direct.has(event.seq) || (event.seq >= first && event.seq <= last));
  };

  const statusFor = (node: string, complete = false): NodeStatus => {
    if (nodeError(node)) return 'error';
    if (complete || nodeEnd(node)) return 'complete';
    if (nodeStart(node)) return isRunning ? 'active' : 'error';
    return 'pending';
  };

  const resourcePayload = (node: string) => payload(lastEvent(events, 'node_resources', node));
  const llmPayload = (node: string) => payload(lastEvent(scopeFor(node), 'llm_start'));

  const buildNodeResources = (node: string): NodeResource[] => {
    const scope = scopeFor(node);
    const resources = resourcePayload(node);
    const llm = llmPayload(node);
    const visibleToolCalls = scope.filter(event =>
      event.event_type === 'tool_call'
      && event.tool
      && !event.parent_id,
    );
    const scopedTools = [...new Set(visibleToolCalls.map(event => String(event.tool)))];
    const offeredTools = Array.isArray(llm.tools) && llm.tools.length
      ? llm.tools
      : Array.isArray(resources.tools) && resources.tools.length
        ? resources.tools
        : scopedTools;
    const out: Array<NodeResource | null> = [
      makeNodeResource(`${node}-prompt`, 'prompt', 'Prompt', 'input',
        llm.prompt ?? resources.prompt, 'muted'),
      makeNodeResource(`${node}-tools`, 'tools', '工具清单', 'input',
        offeredTools, Array.isArray(offeredTools) && offeredTools.length ? 'ok' : 'muted'),
      makeNodeResource(`${node}-context`, 'context', '可见上下文', 'input',
        llm.context_preview
          ?? resources.context_preview,
        'muted'),
      makeNodeResource(`${node}-state`, 'state', '节点状态', 'input',
        resources.context_decision
          ?? resources.message_count
          ?? resources.domain
          ?? llm.message_count,
        'muted'),
    ];

    if (node === 'understand') {
      out.push(makeNodeResource(`${node}-query`, 'query', '用户问题', 'input',
        flow.query, 'default'));
      out.push(makeNodeResource(`${node}-result`, 'result', '识别结果', 'output',
        payload(lastEvent(events, 'intent', node)), 'ok'));
    } else if (node === 'memory') {
      out.push(makeNodeResource(`${node}-result`, 'result', '上下文包', 'output',
        {
          snapshot: resources.context_snapshot,
          pack: payload(lastEvent(events, 'context_built', node)).pack,
        }, 'ok'));
    } else if (node === 'context') {
      out.push(makeNodeResource(`${node}-result`, 'result', '工作区上下文', 'output',
        resources.context, 'ok'));
    } else if (node === 'resolve') {
      out.push(makeNodeResource(`${node}-result`, 'result', '解析结果', 'output',
        resources.resolved ?? resources.candidates, 'ok'));
    } else if (node === 'domain') {
      out.push(makeNodeResource(`${node}-result`, 'result', '路由结果', 'output',
        { domain: resources.domain, mode: flow.metrics.mode }, 'ok'));
    } else if (node === 'plan') {
      out.push(makeNodeResource(`${node}-result`, 'result', '计划', 'output',
        payload(lastEvent(events, 'plan', node)).steps, 'ok'));
    } else if (node === 'verify') {
      out.push(makeNodeResource(`${node}-result`, 'result', '验证报告', 'output',
        payload(lastEvent(events, 'plan_verify', node)).verification, 'ok'));
    }

    scope.filter(event => event.event_type === 'plan_step').forEach(event => {
      out.push(makeNodeResource(
        `${node}-step-${event.seq}`, 'result',
        `步骤 ${String(payload(event).step_id ?? event.seq)}`, 'output',
        payload(event), String(payload(event).status) === 'failed' ? 'danger' : 'ok',
      ));
    });

    visibleToolCalls.forEach(event => {
      const data = payload(event);
      out.push(makeNodeResource(
        `${node}-tool-${event.seq}`, 'tool',
        `工具 · ${String(event.tool ?? 'tool')}`, 'output',
        {
          outcome: event.outcome,
          duration_ms: event.duration_ms,
          args: data.args,
          result: data.operation ?? data.result_summary,
          parsed: data.parsed,
        },
        event.outcome === 'succeeded' ? 'ok' : 'danger',
      ));
    });

    scope.filter(event => event.event_type === 'subagent_dispatch').forEach(event => {
      out.push(makeNodeResource(
        `${node}-subagent-${event.seq}`, 'subagent',
        `子 Agent · ${String(payload(event).role ?? 'worker')}`, 'output',
        payload(event), 'ok',
      ));
    });

    scope.filter(event => event.event_type === 'tool_timing').forEach(event => {
      const data = payload(event);
      out.push(makeNodeResource(
        `${node}-timing-${event.seq}`, 'subagent',
        `${String(event.tool ?? 'subagent')} 用时`, 'output',
        {
          duration_ms: event.duration_ms,
          outcome: event.outcome,
          kind: data.kind ?? 'tool',
          call_id: data.call_id,
        },
        event.outcome === 'succeeded' ? 'ok' : 'danger',
      ));
    });

    if (node === 'synthesize' || node === 'chat' || node === 'clarify' || node === 'task') {
      out.push(makeNodeResource(`${node}-answer`, 'result', '最终回答', 'output',
        payload(lastEvent(events, 'final_answer', node) ?? lastEvent(events, 'final_answer')).answer,
        'ok'));
    }

    nodeEvents(node)
      .filter(event => event.event_type === 'node_error' || event.error)
      .forEach(event => {
        out.push(makeNodeResource(
          `${node}-error-${event.seq}`, 'error', '错误', 'output',
          { type: event.event_type, error: event.error ?? payload(event).error },
          'danger',
        ));
      });

    return compactResources(out.filter(Boolean) as NodeResource[]);
  };

  const factsFor = (node: string): NodeFact[] => {
    const scope = scopeFor(node);
    const llmCalls = scope.filter(event => event.event_type === 'llm_call').length;
    const tools = new Set(
      scope
        .filter(event => event.event_type === 'tool_call' && !event.parent_id)
        .map(event => event.tool),
    ).size;
    const resources = buildNodeResources(node);
    return [
      { label: 'LLM', value: String(llmCalls) },
      { label: '工具', value: String(tools) },
      { label: '资源', value: String(resources.length) },
      { label: 'Token', value: String(tokenTotal(scope)) },
    ];
  };

  const node = (
    id: string,
    name: string,
    title: string,
    kind: string,
    description: string,
    complete = false,
  ): PipelineNode => ({
    id,
    title,
    kind,
    description,
    status: statusFor(name, complete),
    durationMs: eventDuration(events, 'node_end', name),
    facts: factsFor(name),
    resources: buildNodeResources(name),
  });

  const nodes: PipelineNode[] = [];
  if (nodeStart('understand') || nodeEnd('understand') || lastEvent(events, 'intent')) {
    nodes.push(node(
      'node-understand', 'understand', 'understand · 问题理解', 'analysis',
      '读取用户问题，产出意图、领域、实体和规划信号。',
      Boolean(lastEvent(events, 'intent') || nodeEnd('understand')),
    ));
  }
  if (nodeStart('memory') || nodeEnd('memory') || lastEvent(events, 'context_built')) {
    nodes.push(node(
      'node-memory', 'memory', 'memory · 上下文装配', 'context',
      '读取历史、记忆和画像，产出带预算的 Context Pack。',
      Boolean(lastEvent(events, 'context_built') || nodeEnd('memory')),
    ));
  }
  if (nodeStart('context') || nodeEnd('context')) {
    nodes.push(node(
      'node-context', 'context', 'context · 工作区绑定', 'context',
      '读取当前文档、实验项目、研究主题和近期实验状态。',
      nodeEnd('context'),
    ));
  }
  if (nodeStart('resolve') || nodeEnd('resolve')) {
    nodes.push(node(
      'node-resolve', 'resolve', 'resolve · 引用解析', 'resolution',
      '把用户措辞归一为可验证的论文和章节引用。',
      nodeEnd('resolve'),
    ));
  }
  if (nodeStart('domain') || nodeEnd('domain')) {
    nodes.push(node(
      'node-domain', 'domain', 'domain · 领域与模式路由', 'resolution',
      '选择 paper / creation / coding 领域以及 react / plan 执行模式。',
      nodeEnd('domain'),
    ));
  }

  const planPath = Boolean(
    nodeStart('plan') || nodeEnd('plan') || lastEvent(events, 'plan')
    || nodeStart('executor') || nodeEnd('executor')
    || events.some(event => event.event_type === 'plan_step'),
  );
  if (planPath) {
    if (nodeStart('plan') || nodeEnd('plan') || lastEvent(events, 'plan')) {
      nodes.push(node(
        'node-plan', 'plan', 'plan · 任务分解', 'plan',
        '读取问题、实体和引用线索，产出结构化依赖步骤。',
        Boolean(lastEvent(events, 'plan') || nodeEnd('plan')),
      ));
    }
    if (nodeStart('executor') || nodeEnd('executor') || events.some(event => event.event_type === 'plan_step')) {
      nodes.push(node(
        'node-executor', 'executor', 'executor · 步骤执行', 'agent',
        '按依赖执行结果步骤；每步读取独立 Prompt、工具、上下文和前序结果。',
        nodeEnd('executor'),
      ));
    }
    if (nodeStart('verify') || nodeEnd('verify') || lastEvent(events, 'plan_verify')) {
      nodes.push(node(
        'node-verify', 'verify', 'verify · 结果验证', 'plan',
        '读取步骤结果，判断目标是否满足并列出缺口。',
        Boolean(lastEvent(events, 'plan_verify') || nodeEnd('verify')),
      ));
    }
  } else if (
    nodeStart('agent') || nodeEnd('agent')
    || events.some(event => event.event_type === 'tool_call' && event.node === 'agent')
  ) {
    nodes.push(node(
      'node-agent', 'agent', 'search / agent · 推理循环', 'agent',
      '读取 Prompt、工具清单和上下文，执行工具并观察结果。',
      finalSeen && (nodeEnd('agent') || nodeStart('agent')),
    ));
  }

  const finalNodeName = ['synthesize', 'chat', 'clarify', 'task']
    .find(candidate => nodeEnd(candidate) || lastEvent(events, 'final_answer', candidate))
    ?? String(lastEvent(events, 'final_answer')?.node ?? 'synthesize');
  if (nodeStart(finalNodeName) || nodeEnd(finalNodeName) || lastEvent(events, 'final_answer')) {
    nodes.push(node(
      `node-${finalNodeName}`,
      finalNodeName,
      `${finalNodeName} · 回答生成`,
      'answer',
      '读取已确认的上下文、证据和工具结果，生成面向用户的回答。',
      Boolean(lastEvent(events, 'final_answer') || nodeEnd(finalNodeName)),
    ));
  }

  const subagentEvents = events.filter(event => event.event_type === 'subagent_dispatch');
  if (subagentEvents.length) {
    const parent = nodes.find(item => item.kind === 'agent')
      ?? nodes.find(item => item.id === 'node-executor');
    if (parent) {
      parent.children = subagentEvents.map(event => buildSubagentNode(event, events));
    }
  }

  return nodes;
}

function buildGraphLayout(nodes: PipelineNode[]): {
  placements: GraphPlacement[];
  edges: GraphEdge[];
  width: number;
  height: number;
} {
  const placements: GraphPlacement[] = [];
  const edges: GraphEdge[] = [];
  const mainX = 24;
  const branchX = 330;
  const mainWidth = 250;
  const mainHeight = 112;
  const branchWidth = 300;
  const branchHeight = 74;
  let y = 26;
  let previousMain: PipelineNode | undefined;

  nodes.forEach((node, index) => {
    const main: GraphPlacement = {
      id: node.id,
      node,
      x: mainX,
      y,
      width: mainWidth,
      height: mainHeight,
      lane: 'main',
    };
    placements.push(main);
    if (previousMain) {
      edges.push({
        id: `${previousMain.id}->${node.id}`,
        from: previousMain.id,
        to: node.id,
        kind: 'main',
        status: edgeStatus(previousMain, node),
      });
    }
    previousMain = node;

    const children = node.children ?? [];
    let branchY = y + mainHeight + 28;
    let previousChild: PipelineNode | undefined;
    children.forEach(child => {
      const placement: GraphPlacement = {
        id: child.id,
        node: child,
        x: branchX,
        y: branchY,
        width: branchWidth,
        height: branchHeight,
        lane: 'branch',
      };
      placements.push(placement);
      const source = previousChild ?? node;
      edges.push({
        id: `${source.id}->${child.id}`,
        from: source.id,
        to: child.id,
        kind: previousChild ? 'branch' : 'branch',
        status: edgeStatus(source, child),
      });
      previousChild = child;
      branchY += branchHeight + 14;
    });

    if (previousChild && nodes[index + 1]) {
      edges.push({
        id: `${previousChild.id}->${nodes[index + 1].id}`,
        from: previousChild.id,
        to: nodes[index + 1].id,
        kind: node.kind === 'agent' ? 'loop' : 'branch',
        status: edgeStatus(previousChild, nodes[index + 1]),
      });
    }

    const branchExtent = children.length
      ? mainHeight + 28 + children.length * (branchHeight + 14) + 18
      : mainHeight + 48;
    y += Math.max(mainHeight + 48, branchExtent);
  });

  return {
    placements,
    edges,
    width: branchX + branchWidth + 36,
    height: Math.max(360, y + 20),
  };
}

function edgePath(from: GraphPlacement, to: GraphPlacement): string {
  if (from.lane === 'main' && to.lane === 'main') {
    const sx = from.x + from.width / 2;
    const sy = from.y + from.height;
    const tx = to.x + to.width / 2;
    const ty = to.y;
    return `M ${sx} ${sy} C ${sx} ${sy + 26}, ${tx} ${ty - 26}, ${tx} ${ty}`;
  }
  const sx = from.x + from.width;
  const sy = from.y + from.height / 2;
  const tx = to.x;
  const ty = to.y + to.height / 2;
  return `M ${sx} ${sy} C ${sx + 38} ${sy}, ${tx - 38} ${ty}, ${tx} ${ty}`;
}

function eventDuration(events: EvalTraceEvent[], type: string, node?: string): number {
  return events
    .filter(event => event.event_type === type && (!node || event.node === node))
    .reduce((sum, event) => sum + Number(event.duration_ms ?? 0), 0);
}

function nodeSection(label: string, value: unknown, tone: NodeSection['tone'] = 'default'): NodeSection | null {
  if (value === null || value === undefined || value === '') return null;
  const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  if (!text.trim()) return null;
  return { label, value: text, tone };
}

function buildToolNode(toolEvent: EvalTraceEvent, scope: EvalTraceEvent[]): PipelineNode {
  const toolName = toolEvent.tool || 'tool';
  const toolPayload = payload(toolEvent);
  const audit = [...scope].reverse().find(event =>
    event.event_type === 'tool_audit'
    && event.tool === toolName
    && event.seq <= toolEvent.seq,
  ) ?? scope.find(event => event.event_type === 'tool_audit' && event.tool === toolName);
  const retrieval = scope.find(event =>
    event.event_type === 'retrieved_context'
    && event.tool === toolName
    && event.seq >= toolEvent.seq,
  );
  const parsed = asRecord(toolPayload.parsed);
  const auditPayload = payload(audit);
  const retrievalPayload = payload(retrieval);
  const failed = toolEvent.outcome !== 'succeeded'
    || (Object.keys(parsed).length > 0
      && parsed.outcome !== 'succeeded'
      && parsed.outcome !== undefined);
  const isSubagent = /(^|_)(task|subagent|delegate)/i.test(toolName);
  const artifactStored = Boolean(toolPayload.result_stored);

  const children: PipelineNode[] = [];
  if (toolPayload.args) {
    children.push({
      id: `${toolEvent.seq}-args`,
      title: '调用入参',
      kind: 'io',
      description: '记录本次模型生成的工具参数',
      status: 'complete',
      sections: [nodeSection('arguments', toolPayload.args)!],
    });
  }
  if (audit) {
    children.push({
      id: `${toolEvent.seq}-audit`,
      title: '治理与入参校验',
      kind: 'gate',
      description: '权限、schema、审批、幂等键与重试决策',
      status: auditPayload.decision === 'deny' ? 'error' : 'complete',
      facts: [
        { label: 'decision', value: String(auditPayload.decision ?? '-') },
        { label: 'attempts', value: String(auditPayload.attempts ?? '-') },
        {
          label: 'side effect',
          value: String(auditPayload.side_effect ?? '-'),
          tone: auditPayload.side_effect ? 'warn' : 'muted',
        },
      ],
      sections: [
        nodeSection('reason', auditPayload.decision_reason),
        nodeSection('idempotency', auditPayload.idempotency_key),
        nodeSection('error', auditPayload.error_type || auditPayload.error_code, 'danger'),
      ].filter(Boolean) as NodeSection[],
    });
  }
  children.push({
    id: `${toolEvent.seq}-execute`,
    title: '工具执行',
    kind: 'execution',
    description: auditPayload.tool_version ? `工具版本 ${String(auditPayload.tool_version)}` : '适配器执行',
    status: failed ? 'error' : 'complete',
    durationMs: toolEvent.duration_ms,
    sections: [
      nodeSection('operation', toolPayload.operation),
      nodeSection('transport error', toolEvent.error, 'danger'),
    ].filter(Boolean) as NodeSection[],
  });
  children.push({
    id: `${toolEvent.seq}-parse`,
    title: '结果解析与结果引用',
    kind: 'parse',
    description: parsed.is_envelope ? '统一信封解析' : '纯文本结果解析',
    status: failed ? 'error' : 'complete',
    facts: [
      { label: 'envelope', value: String(parsed.is_envelope ?? false) },
      {
        label: 'outcome',
        value: String(parsed.outcome ?? toolEvent.outcome ?? '-'),
        tone: failed ? 'danger' : 'ok',
      },
      { label: 'error type', value: String(parsed.error_type || '-'), tone: parsed.error_type ? 'danger' : 'muted' },
    ],
    sections: [
      nodeSection('failure reason', parsed.error || toolEvent.error, 'danger'),
      nodeSection(
        'artifact',
        artifactStored
          ? `${String(toolPayload.result_ref ?? '')}\nsha256=${String(toolPayload.result_sha256 ?? '')}\nbytes=${String(toolPayload.result_bytes ?? '')}`
          : '未触发长结果外部化',
        artifactStored ? 'ok' : 'muted',
      ),
    ].filter(Boolean) as NodeSection[],
  });
  if (retrieval) {
    const chunkIds = (retrievalPayload.chunk_ids as unknown[] | undefined) ?? [];
    children.push({
      id: `${toolEvent.seq}-citation`,
      title: '引用指向与检索上下文',
      kind: 'citation',
      description: `${chunkIds.length} 个检索片段写入上下文`,
      status: 'complete',
      facts: [
        { label: 'chunks', value: String(chunkIds.length) },
        { label: 'query', value: short(retrievalPayload.query ?? '-', 80) },
      ],
      sections: [
        nodeSection('chunk ids', chunkIds.join('\n')),
        nodeSection('snapshot', retrievalPayload.snapshot),
      ].filter(Boolean) as NodeSection[],
    });
  }
  children.push({
    id: `${toolEvent.seq}-context`,
    title: '写入 Observation / 去重',
    kind: 'context',
    description: '结果进入下一轮上下文；相同调用可通过缓存与幂等键避免重复执行',
    status: 'complete',
    facts: [
      { label: 'cache key', value: String(auditPayload.idempotency_key ?? 'derived') },
      { label: 'context entry', value: failed ? 'failure record' : 'observation' },
    ],
  });

  return {
    id: `${isSubagent ? 'subagent' : 'tool'}-${toolEvent.seq}`,
    title: `${isSubagent ? '子 Agent 任务' : '工具调用'} · ${toolName}`,
    kind: isSubagent ? 'subagent' : 'tool',
    description: failed ? '调用失败，已形成失败记录供恢复决策' : '调用成功，结果已解析并准备写入上下文',
    status: failed ? 'error' : 'complete',
    durationMs: toolEvent.duration_ms,
    children,
  };
}

function buildSubagentNode(
  dispatchEvent: EvalTraceEvent,
  scope: EvalTraceEvent[],
): PipelineNode {
  const data = payload(dispatchEvent);
  const taskId = String(data.task_id ?? '');
  const resultEvent = scope.find(event =>
    event.event_type === 'subagent_result'
    && String(payload(event).task_id ?? '') === taskId
    && event.seq > dispatchEvent.seq,
  );
  const result = payload(resultEvent);
  const failed = resultEvent?.error || result.status === 'failed';
  return {
    id: `subagent-${taskId || dispatchEvent.seq}`,
    title: `子 Agent · ${String(data.role ?? 'worker')}`,
    kind: 'subagent',
    description: String(data.title ?? '隔离任务'),
    status: resultEvent ? (failed ? 'error' : 'complete') : 'active',
    durationMs: resultEvent?.duration_ms,
    facts: [
      { label: 'task', value: taskId || '-' },
      { label: 'role', value: String(data.role ?? '-') },
      { label: 'status', value: String(result.status ?? 'running') },
      { label: 'tools', value: String((data.tools as unknown[] | undefined)?.length ?? 0) },
    ],
    resources: compactResources([
      makeNodeResource(`${taskId}-prompt`, 'prompt', '子 Agent Prompt', 'input',
        data.prompt_preview, 'muted'),
      makeNodeResource(`${taskId}-tools`, 'tools', '工具清单', 'input',
        data.tools, 'ok'),
      makeNodeResource(`${taskId}-task`, 'context', '任务输入', 'input',
        data.task_preview, 'muted'),
      makeNodeResource(`${taskId}-result`, 'result', '执行结果', 'output',
        result.output_preview, failed ? 'danger' : 'ok'),
      makeNodeResource(`${taskId}-error`, 'error', '错误', 'output',
        resultEvent?.error || result.error, 'danger'),
    ].filter(Boolean) as NodeResource[]),
    sections: [
      nodeSection('injected prompt', data.prompt_preview),
      nodeSection('tool configuration', data.tools),
      nodeSection('task prompt', data.task_preview),
      nodeSection('agent handoff', {
        parent_thread: data.parent_thread,
        model: data.model,
      }),
      nodeSection('result', result.output_preview),
      nodeSection('error', resultEvent?.error || result.error, 'danger'),
    ].filter(Boolean) as NodeSection[],
  };
}

function buildPipelineNodes(
  flow: EvalSingleFlow,
  isRunning: boolean,
): PipelineNode[] {
  const events = flow.events ?? [];
  const finalSeen = hasEvent(events, 'final_answer') || hasEvent(events, 'turn_end');
  const hasError = events.some(event => event.event_type === 'node_error' || Boolean(event.error));
  const nodes: PipelineNode[] = [];

  const turnStart = events.find(event => event.event_type === 'turn_start');
  nodes.push({
    id: 'input',
    title: '问题输入',
    kind: 'input',
    description: '用户问题进入运行时',
    status: statusOf(Boolean(turnStart), isRunning),
    sections: [
      nodeSection('query', turnStart ? payload(turnStart).query || flow.query : flow.query),
      nodeSection('thread', flow.thread_id),
      nodeSection('mode', flow.metrics.mode || 'auto'),
    ].filter(Boolean) as NodeSection[],
  });

  const runtimeDone = hasEvent(events, 'node_start') || hasEvent(events, 'llm_call');
  nodes.push({
    id: 'runtime',
    title: '创建 Agent / LLM 运行实例',
    kind: 'runtime',
    description: '冻结本轮运行上下文并建立根 trace',
    status: statusOf(runtimeDone, isRunning && Boolean(turnStart)),
    facts: [
      { label: 'trace', value: flow.metrics.trace_id || '-' },
      { label: 'events', value: String(flow.events?.length ?? flow.metrics.event_count ?? 0) },
    ],
  });

  const snapshotEvent = events.find(event => event.event_type === 'configuration_snapshot');
  const snapshot = payload(snapshotEvent);
  const promptVersionEvents = events.filter(event =>
    event.event_type === 'prompt_resolved'
    || event.event_type === 'prompt_canary_selected'
    || event.event_type === 'prompt_override_selected',
  );
  const promptVersions = {
    ...asRecord(snapshot.prompt_versions),
    ...Object.fromEntries(promptVersionEvents.map(event => [
      String(payload(event).prompt_id ?? 'prompt'),
      String(payload(event).version ?? '-'),
    ])),
  };
  const configEvidence = Boolean(snapshotEvent) || promptVersionEvents.length > 0
    || hasEvent(events, 'context_pack_prompt');
  nodes.push({
    id: 'config',
    title: '注入配置、Prompt、工具与 MCP / Skills',
    kind: 'config',
    description: '本轮使用的配置组合冻结为快照，后续节点只读该快照',
    status: statusOf(configEvidence, isRunning && runtimeDone && !finalSeen),
    facts: [
      { label: 'revision', value: short(snapshot.config_revision ?? flow.metrics.config_revision ?? '-', 32) },
      { label: 'model', value: short(snapshot.model_route ?? '-', 48) },
      { label: 'tool registry', value: short(snapshot.tool_registry_hash ?? '-', 24) },
      {
        label: 'disabled tools',
        value: String(
          Object.values(asRecord(snapshot.disabled_tools))
            .reduce(
              (sum: number, value: unknown) =>
                sum + (Array.isArray(value) ? value.length : 0),
              0,
            ),
        ),
      },
    ],
    sections: [
      nodeSection('prompt versions', promptVersions),
      nodeSection('feature flags', snapshot.feature_flags),
      nodeSection('limits', snapshot.runtime_limits),
    ].filter(Boolean) as NodeSection[],
  });

  const intentEvent = events.find(event => event.event_type === 'intent');
  const intentStart = events.find(event => event.event_type === 'node_start' && event.node === 'understand');
  const intentFailed = events.some(event => event.event_type === 'understand_llm_failed');
  nodes.push({
    id: 'intent',
    title: '问题分析',
    kind: 'analysis',
    description: '识别意图、领域、实体、指代与是否需要计划',
    status: intentEvent ? 'complete' : intentFailed ? 'error' : statusOf(false, Boolean(intentStart)),
    durationMs: eventDuration(events, 'node_end', 'understand'),
    facts: [
      { label: 'intent', value: String(intentEvent?.intent ?? '-') },
      { label: 'domain', value: String(payload(intentEvent).domain ?? '-') },
      { label: 'confidence', value: fmtNumber(payload(intentEvent).confidence) },
      { label: 'planning', value: String(payload(intentEvent).needs_planning ?? '-') },
    ],
    sections: [
      nodeSection('entities', (payload(intentEvent).entities as unknown[] | undefined)?.join(', ')),
      nodeSection('focus papers', (payload(intentEvent).focus_papers as unknown[] | undefined)?.join(', ')),
    ].filter(Boolean) as NodeSection[],
  });

  const contextBuilt = events.find(event => event.event_type === 'context_built');
  const memoryStart = events.find(event => event.event_type === 'node_start' && event.node === 'memory');
  const contextData = payload(contextBuilt);
  const pack = asRecord(contextData.pack);
  nodes.push({
    id: 'context',
    title: '上下文 / 记忆装配',
    kind: 'context',
    description: '组装对话、记忆、任务和检索上下文，并应用窗口预算',
    status: contextBuilt ? 'complete' : statusOf(false, Boolean(memoryStart)),
    durationMs: eventDuration(events, 'node_end', 'memory'),
    facts: [
      { label: 'estimated', value: `${String(pack.estimated_tokens ?? contextData.estimated_tokens ?? '-')} tok` },
      { label: 'budget', value: `${String(pack.max_tokens ?? contextData.max_tokens ?? '-')} tok` },
      { label: 'utilization', value: fmtNumber(pack.utilization ?? contextData.utilization, 3) },
      { label: 'summary', value: String(contextData.summary_used ?? false) },
    ],
    sections: [
      nodeSection('context decision', contextData),
      nodeSection('runtime snapshot', flow.context_snapshot),
    ].filter(Boolean) as NodeSection[],
  });

  const resolveDone = hasEvent(events, 'node_end', 'resolve') && hasEvent(events, 'node_end', 'domain');
  const resolveActive = hasEvent(events, 'node_start', 'resolve') || hasEvent(events, 'node_start', 'domain');
  nodes.push({
    id: 'resolve',
    title: '引用解析与领域路由',
    kind: 'resolution',
    description: '把用户措辞映射到可验证的论文/工作对象，并选择执行域',
    status: statusOf(resolveDone, resolveActive),
    durationMs: eventDuration(events, 'node_end', 'resolve') + eventDuration(events, 'node_end', 'domain'),
  });

  const planEvents = events.filter(event =>
    event.event_type === 'plan'
    || event.event_type === 'plan_step'
    || event.event_type === 'plan_verify'
    || (event.node === 'plan' && (event.event_type === 'node_start' || event.event_type === 'node_end')),
  );
  nodes.push({
    id: 'plan',
    title: '任务规划',
    kind: 'plan',
    description: '需要多步骤时生成计划；单步任务可跳过',
    status: planEvents.length
      ? statusOf(hasEvent(events, 'plan_verify') || hasEvent(events, 'node_end', 'plan'), isRunning && !finalSeen)
      : statusOf(false, false, finalSeen),
    facts: [
      {
        label: 'steps',
        value: String(
          events
            .filter(event => event.event_type === 'plan')
            .flatMap(event => (payload(event).steps as unknown[] | undefined) ?? []).length,
        ),
      },
      {
        label: 'verification',
        value: String(
          asRecord(payload(events.find(event => event.event_type === 'plan_verify')).verification).status
          ?? '-',
        ),
      },
    ],
    sections: [
      nodeSection('plan', payload(events.find(event => event.event_type === 'plan')).steps),
      nodeSection('verify', payload(events.find(event => event.event_type === 'plan_verify')).verification),
    ].filter(Boolean) as NodeSection[],
  });

  const agentStarts = events.filter(event => event.event_type === 'node_start' && event.node === 'agent');
  const agentGroups: Array<{ start: EvalTraceEvent; end?: EvalTraceEvent; events: EvalTraceEvent[] }> = [];
  for (const start of agentStarts) {
    const end = events.find(event =>
      event.event_type === 'node_end'
      && event.node === 'agent'
      && event.seq > start.seq
      && !agentStarts.some(other => other !== start && other.seq > start.seq && other.seq < event.seq),
    );
    const upper = end?.seq ?? Number.POSITIVE_INFINITY;
    agentGroups.push({
      start,
      end,
      events: events.filter(event => event.seq > start.seq && event.seq < upper),
    });
  }
  if (agentGroups.length === 0) {
    const loopEvents = events.filter(event =>
      event.event_type === 'llm_call' || event.event_type === 'tool_call',
    );
    if (loopEvents.length) {
      agentGroups.push({
        start: loopEvents[0],
        end: loopEvents[loopEvents.length - 1],
        events: loopEvents,
      });
    }
  }

  agentGroups.forEach((group, index) => {
    const children: PipelineNode[] = [];
    for (const event of group.events) {
      if (event.event_type === 'llm_call') {
        const data = payload(event);
        const tokens = asRecord(data.tokens);
        children.push({
          id: `llm-${event.seq}`,
          title: `LLM 思考 ${children.filter(item => item.kind === 'llm').length + 1}`,
          kind: 'llm',
          description: event.model || '模型调用',
          status: event.error ? 'error' : 'complete',
          durationMs: event.duration_ms,
          facts: [
            { label: 'model', value: event.model || '-' },
            { label: 'mode', value: String(data.mode ?? '-') },
            { label: 'tokens', value: `${String(tokens.prompt_tokens ?? 0)} / ${String(tokens.completion_tokens ?? 0)}` },
            { label: 'estimated', value: String(tokens.estimated ?? false) },
          ],
          sections: [nodeSection('error', event.error, 'danger')].filter(Boolean) as NodeSection[],
        });
      } else if (event.event_type === 'tool_call') {
        children.push(buildToolNode(event, group.events));
      } else if (event.event_type === 'subagent_dispatch') {
        children.push(buildSubagentNode(event, events));
      } else if (event.event_type === 'context_pack_prompt') {
        children.push({
          id: `context-prompt-${event.seq}`,
          title: 'Prompt / Context Pack 注入',
          kind: 'context',
          description: '把当前上下文渲染进本轮模型输入',
          status: 'complete',
          sections: [nodeSection('decision', payload(event))].filter(Boolean) as NodeSection[],
        });
      }
    }
    const groupFailed = children.some(child => child.status === 'error');
    nodes.push({
      id: `agent-${group.start.seq}`,
      title: `LLM 思考 / 工具循环 ${index + 1}`,
      kind: 'agent',
      description: '模型决策 → 工具调用 → 观察写回上下文 → 再决策',
      status: group.end ? (groupFailed ? 'error' : 'complete') : (isRunning ? 'active' : 'error'),
      durationMs: group.end?.duration_ms,
      facts: [
        { label: 'llm calls', value: String(children.filter(item => item.kind === 'llm').length) },
        { label: 'tool calls', value: String(children.filter(item => item.kind === 'tool' || item.kind === 'subagent').length) },
        { label: 'errors', value: String(children.filter(item => item.status === 'error').length) },
      ],
      children,
    });
  });

  const subagentDispatches = events.filter(event => event.event_type === 'subagent_dispatch');
  const representedSubagents = new Set(
    agentGroups.flatMap(group => group.events)
      .filter(event => event.event_type === 'subagent_dispatch')
      .map(event => event.seq),
  );
  subagentDispatches
    .filter(event => !representedSubagents.has(event.seq))
    .forEach(event => nodes.push(buildSubagentNode(event, events)));

  const contextPrompt = events.find(event => event.event_type === 'context_pack_prompt');
  if (contextPrompt) {
    nodes.push({
      id: `window-${contextPrompt.seq}`,
      title: '上下文窗口管理',
      kind: 'window',
      description: '检查预算、截断策略与是否触发摘要/窗口压缩',
      status: 'complete',
      facts: [
        { label: 'max tokens', value: String(payload(contextPrompt).max_tokens ?? '-') },
        { label: 'estimated', value: String(payload(contextPrompt).estimated_tokens ?? '-') },
        { label: 'truncated', value: String(payload(contextPrompt).truncated ?? false) },
      ],
      sections: [nodeSection('decision', contextPrompt.payload)].filter(Boolean) as NodeSection[],
    });
  }

  const finalEvent = events.find(event => event.event_type === 'final_answer');
  const synthesizeStart = events.find(event => event.event_type === 'node_start' && event.node === 'synthesize');
  nodes.push({
    id: 'answer',
    title: '最终回答',
    kind: 'answer',
    description: '综合已确认的上下文与引用生成回答',
    status: finalEvent
      ? 'complete'
      : flow.status === 'failed'
        ? 'error'
        : statusOf(false, Boolean(synthesizeStart) || isRunning),
    durationMs: eventDuration(events, 'node_end', 'synthesize'),
    sections: [
      nodeSection('answer', finalEvent ? payload(finalEvent).answer : '等待回答'),
      nodeSection('verification', payload(finalEvent).verification),
    ].filter(Boolean) as NodeSection[],
  });

  const turnEnd = events.find(event => event.event_type === 'turn_end');
  nodes.push({
    id: 'turn-end',
    title: '运行结束',
    kind: 'turn',
    description: '汇总链路、指标与错误状态',
    status: turnEnd ? (String(payload(turnEnd).status ?? 'ok') === 'ok' && !hasError ? 'complete' : 'error') : 'pending',
    durationMs: turnEnd?.duration_ms ?? flow.metrics.total_duration_ms,
    facts: [
      { label: 'status', value: String(payload(turnEnd).status ?? flow.status ?? '-') },
      { label: 'total', value: fmtMs(flow.metrics.total_duration_ms) },
    ],
  });

  return nodes;
}

function factColor(tone?: NodeFact['tone']): string {
  if (tone === 'ok') return 'var(--color-primary)';
  if (tone === 'warn') return 'var(--color-warning)';
  if (tone === 'danger') return 'var(--color-danger)';
  if (tone === 'muted') return 'var(--color-text-tertiary)';
  return 'var(--color-text)';
}

function resourceTone(resource: NodeResource): string {
  if (resource.tone === 'danger') return 'var(--color-danger)';
  if (resource.tone === 'warn') return 'var(--color-warning)';
  if (resource.tone === 'ok') return 'var(--color-primary)';
  return 'var(--color-text-tertiary)';
}

function ResourceChips({
  resources,
  limit = 5,
}: {
  resources?: NodeResource[];
  limit?: number;
}) {
  if (!resources?.length) return null;
  const visible = resources.slice(0, limit);
  const rest = resources.length - visible.length;
  return (
    <div className="eval-resource-chips">
      {visible.map(resource => (
        <span
          key={resource.id}
          className={`eval-resource-chip is-${resource.role}`}
          title={resource.label}
          style={{ color: resourceTone(resource) }}
        >
          {resource.kind === 'tool' || resource.kind === 'subagent'
            ? `${resource.label}${resourceDurationMs(resource) !== null
              ? ` · ${fmtMs(resourceDurationMs(resource))}`
              : ''}`
            : `${resourceKindLabel(resource.kind)}${resourceCount(resource) > 1
              ? ` ${resourceCount(resource)}`
              : ''}`}
        </span>
      ))}
      {rest > 0 && <span className="eval-resource-chip is-more">+{rest}</span>}
    </div>
  );
}

function ResourceGroups({ resources }: { resources?: NodeResource[] }) {
  if (!resources?.length) return null;
  return (
    <div className="eval-resource-groups">
      {(['input', 'output'] as NodeResource['role'][]).map(role => {
        const items = resources.filter(resource => resource.role === role);
        if (!items.length) return null;
        return (
          <div key={role} className="eval-resource-group">
            <div className="eval-resource-group-label">
              {role === 'input' ? '节点可见资源' : '节点产出'}
            </div>
            {items.map(resource => (
              <div key={resource.id} className="eval-resource-row">
                <span
                  className="eval-resource-row-kind"
                  style={{ color: resourceTone(resource) }}
                >
                  {resourceKindLabel(resource.kind)}
                </span>
                <div className="eval-resource-row-body">
                  <div className="eval-resource-row-label">{resource.label}</div>
                  <div style={{ ...mono, color: factColor(resource.tone) }}>
                    {short(resource.value, 1400)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
}

function ResourceKindDisclosures({
  resources,
}: {
  resources?: NodeResource[];
}) {
  if (!resources?.length) return null;
  const order: ResourceKind[] = [
    'prompt', 'tools', 'context', 'state', 'query',
    'tool', 'subagent', 'result', 'artifact', 'error',
  ];
  return (
    <>
      {order.map(kind => {
        const items = resources.filter(resource => resource.kind === kind);
        if (!items.length) return null;
        return (
          <InspectorDisclosure
            key={kind}
            label={resourceKindLabel(kind)}
            count={items.length}
          >
            <div className="eval-resource-detail-list">
              {items.map(resource => (
                <div key={resource.id} className="eval-resource-detail">
                  <div className="eval-resource-detail-label">
                    <span style={{ color: resourceTone(resource) }}>
                      {resource.role === 'input' ? '输入' : '产出'}
                    </span>
                    {resource.label}
                  </div>
                  <div
                    className="eval-resource-detail-code"
                    style={{ color: factColor(resource.tone) }}
                  >
                    {short(resource.value, 6000)}
                  </div>
                </div>
              ))}
            </div>
          </InspectorDisclosure>
        );
      })}
    </>
  );
}

function NodeCard({ node, depth = 0 }: { node: PipelineNode; depth?: number }) {
  const color = statusColor(node.status);
  return (
    <div className="eval-node-wrap" style={{ marginLeft: depth * 18 }}>
      <div className="eval-node-rail" style={{ background: color }}>
        {statusIcon(node.status)}
      </div>
      <details
        className="eval-node-card"
        open={node.status === 'active' || node.status === 'error' || depth === 0}
        style={{
          borderColor: node.status === 'active' || node.status === 'error'
            ? color
            : 'var(--color-border)',
        }}
      >
        <summary>
          <span style={{ minWidth: 0 }}>
            <span className="eval-node-title">{node.title}</span>
            <span className="eval-node-description">{node.description}</span>
          </span>
          <span className="eval-node-meta">
            {node.durationMs != null && node.durationMs > 0 && <span>{fmtMs(node.durationMs)}</span>}
            <span className={`eval-node-status eval-node-status-${node.status}`}>{node.status}</span>
          </span>
        </summary>
        <div className="eval-node-body">
          {node.facts && node.facts.length > 0 && (
            <div className="eval-node-facts">
              {node.facts.map(fact => (
                <div key={`${node.id}-${fact.label}`} className="eval-node-fact">
                  <span>{fact.label}</span>
                  <b style={{ color: factColor(fact.tone) }}>{fact.value}</b>
                </div>
              ))}
            </div>
          )}
          <ResourceGroups resources={node.resources} />
          {node.sections?.map(section => (
            <div key={`${node.id}-${section.label}`} className="eval-node-section">
              <div className="eval-node-section-label">{section.label}</div>
              <div style={{ ...mono, color: factColor(section.tone) }}>{short(section.value, 1400)}</div>
            </div>
          ))}
          {node.children && node.children.length > 0 && (
            <div className="eval-node-children">
              {node.children.map(child => <NodeCard key={child.id} node={child} depth={depth + 1} />)}
            </div>
          )}
        </div>
      </details>
    </div>
  );
}

function InspectorDisclosure({
  label,
  count,
  children,
}: {
  label: string;
  count?: number;
  children: ReactNode;
}) {
  return (
    <details className="eval-inspector-disclosure">
      <summary>
        <span>{label}</span>
        {count !== undefined && <b>{count}</b>}
        <i aria-hidden="true">›</i>
      </summary>
      <div className="eval-inspector-disclosure-body">{children}</div>
    </details>
  );
}

function GraphInspector({
  node,
  onSelect,
}: {
  node?: PipelineNode;
  onSelect: (id: string) => void;
}) {
  if (!node) {
    return (
      <aside className="eval-graph-inspector">
        <div className="eval-graph-inspector-empty">点击任意节点查看运行信息</div>
      </aside>
    );
  }
  const color = statusColor(node.status);
  return (
    <aside className="eval-graph-inspector">
      <div className="eval-graph-inspector-head">
        <span className="eval-graph-kind">{graphNodeLabel(node)}</span>
        <span className={`eval-node-status eval-node-status-${node.status}`}>{node.status}</span>
      </div>
      <div className="eval-graph-inspector-title">{node.title}</div>
      <div className="eval-graph-inspector-description">{node.description}</div>
      <div className="eval-graph-inspector-facts">
        <div>
          <span>duration</span>
          <b>{fmtMs(node.durationMs)}</b>
        </div>
        <div>
          <span>children</span>
          <b>{node.children?.length ?? 0}</b>
        </div>
        <div>
          <span>state</span>
          <b style={{ color }}>{node.status}</b>
        </div>
      </div>

      {node.resources && node.resources.length > 0 && (
        <ResourceKindDisclosures resources={node.resources} />
      )}

      {node.facts && node.facts.length > 0 && (
        <div className="eval-graph-inspector-metrics">
          {node.facts.map(fact => (
            <div key={fact.label}>
              <span>{fact.label}</span>
              <b style={{ color: factColor(fact.tone) }}>{fact.value}</b>
            </div>
          ))}
        </div>
      )}

      {node.sections?.map(section => (
        <InspectorDisclosure key={section.label} label={section.label}>
          <div
            className="eval-graph-inspector-code"
            style={{ color: factColor(section.tone) }}
          >
            {short(section.value, 6000)}
          </div>
        </InspectorDisclosure>
      ))}

      {node.children && node.children.length > 0 && (
        <InspectorDisclosure label="子节点" count={node.children.length}>
          <div className="eval-graph-child-list">
            {node.children.map(child => (
              <button key={child.id} onClick={() => onSelect(child.id)}>
                <span style={{ background: statusColor(child.status) }} />
                {child.title}
              </button>
            ))}
          </div>
        </InspectorDisclosure>
      )}
    </aside>
  );
}

function GraphView({
  nodes,
  selectedId,
  onSelect,
  isRunning,
}: {
  nodes: PipelineNode[];
  selectedId: string;
  onSelect: (id: string) => void;
  isRunning: boolean;
}) {
  const layout = buildGraphLayout(nodes);
  const storageKey = 'eval.graph.inspector-width';
  const defaultInspectorWidth = 300;
  const clampInspectorWidth = (width: number) => {
    const viewportMax = typeof window === 'undefined'
      ? 560
      : Math.max(220, Math.min(560, window.innerWidth - 420));
    return Math.max(220, Math.min(viewportMax, width));
  };
  const [inspectorWidth, setInspectorWidth] = useState(() => {
    try {
      const stored = Number(window.localStorage.getItem(storageKey));
      return clampInspectorWidth(Number.isFinite(stored) && stored > 0
        ? stored
        : defaultInspectorWidth);
    } catch {
      return defaultInspectorWidth;
    }
  });
  const [resizingInspector, setResizingInspector] = useState(false);
  const dragState = useRef<{
    startX: number;
    startWidth: number;
    currentWidth: number;
  } | null>(null);
  const updateInspectorWidth = (width: number, persist = false) => {
    const next = clampInspectorWidth(width);
    setInspectorWidth(next);
    if (persist) {
      try {
        window.localStorage.setItem(storageKey, String(Math.round(next)));
      } catch {
        // localStorage may be unavailable in hardened browser contexts.
      }
    }
  };
  const onResizeStart = (event: ReactPointerEvent<HTMLDivElement>) => {
    dragState.current = {
      startX: event.clientX,
      startWidth: inspectorWidth,
      currentWidth: inspectorWidth,
    };
    setResizingInspector(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const onResizeMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const state = dragState.current;
    if (!state) return;
    const next = clampInspectorWidth(
      state.startWidth + state.startX - event.clientX,
    );
    state.currentWidth = next;
    updateInspectorWidth(next);
  };
  const onResizeEnd = (event: ReactPointerEvent<HTMLDivElement>) => {
    const state = dragState.current;
    if (!state) return;
    const currentWidth = state.currentWidth;
    dragState.current = null;
    setResizingInspector(false);
    updateInspectorWidth(currentWidth, true);
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch {
      // Pointer capture may already be released by the browser.
    }
  };
  const onResizeKeyDown = (
    event: ReactKeyboardEvent<HTMLDivElement>,
  ) => {
    if (event.key === 'ArrowLeft') {
      event.preventDefault();
      updateInspectorWidth(inspectorWidth + 16, true);
    } else if (event.key === 'ArrowRight') {
      event.preventDefault();
      updateInspectorWidth(inspectorWidth - 16, true);
    } else if (event.key === 'Home') {
      event.preventDefault();
      updateInspectorWidth(defaultInspectorWidth, true);
    }
  };
  const placementById = new Map(layout.placements.map(item => [item.id, item]));
  const selected = flattenNodes(nodes)
    .find(node => node.id === selectedId)
    ?? nodes.find(node => node.status === 'active')
    ?? nodes[0];

  return (
    <div
      className={`eval-graph-layout ${resizingInspector ? 'is-resizing' : ''}`}
      style={{
        '--eval-inspector-width': `${inspectorWidth}px`,
      } as CSSProperties}
    >
      <section className="eval-graph-stage">
        <div className="eval-graph-stage-head">
          <div>
            <span className="eval-graph-stage-title">LangGraph 执行节点</span>
            <span className="eval-graph-stage-sub">
              {isRunning ? '执行状态与资源绑定正在更新' : '节点、输入资源和执行产出'}
            </span>
          </div>
          <div className="eval-graph-legend">
            <span><i className="is-complete" />完成</span>
            <span><i className="is-active" />运行中</span>
            <span><i className="is-error" />失败</span>
            <span><i className="is-pending" />未到达</span>
          </div>
        </div>
        <div className="eval-graph-scroll">
          <div
            className="eval-graph-canvas"
            style={{ width: layout.width, height: layout.height }}
          >
            <svg
              className="eval-graph-edges"
              width={layout.width}
              height={layout.height}
              aria-hidden="true"
            >
              <defs>
                {(['complete', 'active', 'error', 'pending'] as NodeStatus[]).map(status => (
                  <marker
                    key={status}
                    id={`eval-arrow-${status}`}
                    viewBox="0 0 10 10"
                    refX="8"
                    refY="5"
                    markerWidth="7"
                    markerHeight="7"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 0 L 10 5 L 0 10 z" fill={statusColor(status)} />
                  </marker>
                ))}
              </defs>
              {layout.edges.map(edge => {
                const from = placementById.get(edge.from);
                const to = placementById.get(edge.to);
                if (!from || !to) return null;
                const color = statusColor(edge.status);
                return (
                  <path
                    key={edge.id}
                    d={edgePath(from, to)}
                    className={`eval-graph-edge is-${edge.status} is-${edge.kind}`}
                    style={{ color }}
                    stroke={color}
                    fill="none"
                    markerEnd={`url(#eval-arrow-${edge.status})`}
                  />
                );
              })}
            </svg>

            {layout.placements.map(({ node, x, y, width, height, lane }) => {
              const active = node.status === 'active';
              const selectedNode = node.id === selected?.id;
              return (
                <button
                  key={node.id}
                  onClick={() => onSelect(node.id)}
                  className={[
                    'eval-graph-node',
                    `is-${node.status}`,
                    `is-${lane}`,
                    active ? 'is-live' : '',
                    selectedNode ? 'is-selected' : '',
                  ].filter(Boolean).join(' ')}
                  style={{
                    left: x,
                    top: y,
                    width,
                    height,
                    borderColor: selectedNode ? statusColor(node.status) : undefined,
                  }}
                >
                  <span
                    className="eval-graph-node-rail"
                    style={{ background: statusColor(node.status) }}
                  >
                    {statusIcon(node.status)}
                  </span>
                  <span className="eval-graph-node-copy">
                    <span className="eval-graph-node-kind">{graphNodeLabel(node)}</span>
                    <span className="eval-graph-node-title">{node.title}</span>
                    <span className="eval-graph-node-sub">
                      {node.children?.length
                        ? `${node.children.length} 个子节点`
                        : node.description}
                    </span>
                    <ResourceChips resources={node.resources} />
                  </span>
                  {node.durationMs != null && node.durationMs > 0 && (
                    <span className="eval-graph-node-time">{fmtMs(node.durationMs)}</span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      </section>
      <div
        className="eval-graph-resizer"
        role="separator"
        aria-orientation="vertical"
        aria-label="调整右侧详情宽度"
        tabIndex={0}
        onPointerDown={onResizeStart}
        onPointerMove={onResizeMove}
        onPointerUp={onResizeEnd}
        onPointerCancel={onResizeEnd}
        onDoubleClick={() => updateInspectorWidth(defaultInspectorWidth, true)}
        onKeyDown={onResizeKeyDown}
      >
        <span />
      </div>
      <GraphInspector node={selected} onSelect={onSelect} />
    </div>
  );
}

function TimelineView({
  nodes,
  selectedId,
  onSelect,
  isRunning,
  totalDurationMs,
}: {
  nodes: PipelineNode[];
  selectedId: string;
  onSelect: (id: string) => void;
  isRunning: boolean;
  totalDurationMs: number;
}) {
  const selected = flattenNodes(nodes)
    .find(node => node.id === selectedId)
    ?? nodes.find(node => node.status === 'active')
    ?? nodes[0];
  let cursor = 0;
  const rows = nodes.map(node => {
    const duration = Math.max(0, Number(node.durationMs ?? 0));
    const start = cursor;
    cursor += duration;
    return { node, duration, start };
  });
  const measuredTotal = rows.reduce((sum, row) => sum + row.duration, 0);
  const basis = Math.max(1, measuredTotal, Number(totalDurationMs) || 0);

  return (
    <div className="eval-timeline-layout">
      <section className="eval-timeline-stage">
        <div className="eval-timeline-head">
          <div>
            <b>执行时间线</b>
            <span>{isRunning ? '节点完成后持续写入' : '每个阶段共享统一时间标尺'}</span>
          </div>
          <div className="eval-timeline-legend">
            <span><i className="is-complete" />完成</span>
            <span><i className="is-active" />运行中</span>
            <span><i className="is-error" />失败</span>
          </div>
        </div>
        <div className="eval-timeline-axis" aria-hidden="true">
          <span>0</span>
          <span>{fmtMs(totalDurationMs * 0.25)}</span>
          <span>{fmtMs(totalDurationMs * 0.5)}</span>
          <span>{fmtMs(totalDurationMs * 0.75)}</span>
          <span>{fmtMs(totalDurationMs)}</span>
        </div>
        <div className="eval-timeline-rows">
          {rows.map(({ node, duration, start }) => {
            const active = node.status === 'active';
            const selectedNode = node.id === selected?.id;
            const left = (start / basis) * 100;
            const width = Math.max(2.5, (duration / basis) * 100);
            return (
              <button
                type="button"
                key={node.id}
                className={[
                  'eval-timeline-row',
                  `is-${node.status}`,
                  active ? 'is-live' : '',
                  selectedNode ? 'is-selected' : '',
                ].filter(Boolean).join(' ')}
                onClick={() => onSelect(node.id)}
              >
                <span
                  className="eval-timeline-node"
                  style={{ borderColor: statusColor(node.status) }}
                >
                  {statusIcon(node.status)}
                </span>
                <span className="eval-timeline-copy">
                  <span className="eval-timeline-kind">{graphNodeLabel(node)}</span>
                  <b>{node.title}</b>
                  <span>{node.description}</span>
                </span>
                <span className="eval-timeline-track">
                  <i
                    className={`is-${node.status}`}
                    style={{
                      left: `${left}%`,
                      width: `${width}%`,
                      background: statusColor(node.status),
                    }}
                  />
                </span>
                <span className="eval-timeline-duration">
                  {duration > 0 ? fmtMs(duration) : '—'}
                </span>
                <span className="eval-timeline-facts">
                  {(node.facts ?? []).slice(0, 3).map(fact => (
                    <span key={`${node.id}-${fact.label}`}>
                      {fact.label} <b style={{ color: factColor(fact.tone) }}>{fact.value}</b>
                    </span>
                  ))}
                  {(node.children?.length ?? 0) > 0 && (
                    <span>子节点 <b>{node.children?.length}</b></span>
                  )}
                </span>
                <ResourceChips resources={node.resources} limit={4} />
              </button>
            );
          })}
        </div>
      </section>
      <GraphInspector node={selected} onSelect={onSelect} />
    </div>
  );
}

function legacyStagesToNodes(flow: EvalSingleFlow): PipelineNode[] {
  const nodes: PipelineNode[] = [];
  flow.stages.forEach((stage, index) => {
    nodes.push({
      id: `stage-${stage.key}-${index}`,
      title: stage.label,
      kind: stage.key,
      description: `${stage.items.length} 个事件`,
      status: 'complete',
      durationMs: stage.duration_ms,
      children: stage.items.map((item, itemIndex) => {
        const typed = item as EvalFlowItem;
        return {
          id: `stage-${stage.key}-${index}-${itemIndex}`,
          title: typed.type,
          kind: typed.type,
          description: typed.tool || typed.node || typed.model || '',
          status: (
            (typed.outcome !== undefined && typed.outcome !== 'succeeded')
            || typed.error
          ) ? 'error' : 'complete',
          durationMs: typed.duration_ms,
          sections: [nodeSection('payload', typed)].filter(Boolean) as NodeSection[],
        };
      }),
    });
  });
  return nodes;
}

export function SingleFlowView({
  flow,
  isRunning = false,
}: {
  flow: EvalSingleFlow;
  isRunning?: boolean;
}) {
  const [viewMode, setViewMode] = useState<'timeline' | 'graph' | 'trace'>('timeline');
  const [selectedNodeId, setSelectedNodeId] = useState('');
  const executionNodes = flow.events?.length ? buildMainGraphNodes(flow, isRunning) : [];
  const nodes = executionNodes.length
    ? executionNodes
    : flow.events?.length
      ? buildPipelineNodes(flow, isRunning)
      : legacyStagesToNodes(flow);
  const visibleNodes = nodes;
  const completed = visibleNodes.filter(node => node.status === 'complete' || node.status === 'skipped').length;
  const active = visibleNodes.find(node => node.status === 'active');
  const toolEvents = (flow.events ?? []).filter(event => event.event_type === 'tool_call');
  const failedTools = toolEvents.filter(
    event => event.outcome !== undefined && event.outcome !== 'succeeded',
  ).length;
  const storedArtifacts = (flow.events ?? []).filter(event => Boolean(payload(event).result_stored)).length;
  const contextEvent = (flow.events ?? []).find(event => event.event_type === 'context_built');
  const contextData = payload(contextEvent);
  const contextPack = asRecord(contextData.pack);
  const snapshotData = payload(lastEvent(flow.events ?? [], 'configuration_snapshot'));
  const progress = Math.round((completed / Math.max(1, visibleNodes.length)) * 100);
  const finalFailure = !isRunning && (!flow.metrics.task_success || failedTools > 0);
  const badgeLabel = isRunning ? '运行中' : finalFailure ? '存在失败' : '链路成功';
  const badgeClass = isRunning ? 'is-running' : finalFailure ? 'is-bad' : 'is-ok';
  const taskState = isRunning
    ? (failedTools ? '运行中，已有失败' : '运行中')
    : finalFailure ? '失败 / 未确认' : '成功';

  return (
    <div className="eval-flow">
      <section className="eval-flow-summary">
        <div className="eval-flow-summary-head">
          <div style={{ minWidth: 0 }}>
            <div className="eval-flow-kicker">
              <span className={`eval-live-dot ${isRunning ? 'is-running' : ''}`} />
              {isRunning ? 'RUNNING' : flow.status === 'failed' ? 'FAILED' : 'COMPLETE'}
            </div>
            <div className="eval-flow-question">{flow.query}</div>
          </div>
          <span className={`eval-result-badge ${badgeClass}`}>
            {badgeLabel}
          </span>
        </div>
        <div className="eval-flow-progress">
          <div style={{ width: `${progress}%` }} />
        </div>
        <div className="eval-flow-progress-copy">
          <span>{completed}/{visibleNodes.length} 个节点</span>
          <span>{active ? `当前：${active.title}` : isRunning ? '等待下一节点' : '运行结束'}</span>
          <span>{progress}%</span>
        </div>
        <div className="eval-metric-grid">
          <div><span>总耗时</span><b>{fmtMs(flow.metrics.total_duration_ms)}</b></div>
          <div><span>Token</span><b>{String(flow.metrics.tokens.total || 0)}</b></div>
          <div><span>LLM 调用</span><b>{String(flow.metrics.llm_calls || 0)}</b></div>
          <div><span>工具调用</span><b>{String(toolEvents.length || flow.metrics.tool_calls || 0)}</b></div>
          <div><span>失败</span><b className={failedTools ? 'is-danger' : ''}>{String(failedTools || flow.metrics.tool_failures || 0)}</b></div>
          <div><span>长结果落盘</span><b>{String(storedArtifacts)}</b></div>
        </div>
      </section>

      <section className="eval-run-resource-bar">
        <div>
          <span>问题</span>
          <b>{short(flow.query, 120)}</b>
        </div>
        <div>
          <span>模型</span>
          <b>{String(snapshotData.model_route ?? '-')}</b>
        </div>
        <div>
          <span>执行模式</span>
          <b>{flow.metrics.mode || 'auto'}</b>
        </div>
        <div>
          <span>Prompt 绑定</span>
          <b>{String(Object.keys(asRecord(snapshotData.prompt_versions)).length)}</b>
        </div>
        <div>
          <span>工具注册表</span>
          <b>{short(snapshotData.tool_registry_hash ?? '-', 20)}</b>
        </div>
        <div>
          <span>上下文预算</span>
          <b>{String(contextPack.max_tokens ?? contextData.max_tokens ?? '-')}</b>
        </div>
      </section>

      <div className="eval-view-toolbar">
        <div className="eval-view-switch">
          <button
            className={viewMode === 'timeline' ? 'is-active' : ''}
            onClick={() => setViewMode('timeline')}
          >
            <span>≋</span> 时间线
          </button>
          <button
            className={viewMode === 'graph' ? 'is-active' : ''}
            onClick={() => setViewMode('graph')}
          >
            <span>◇</span> 流程图
          </button>
          <button
            className={viewMode === 'trace' ? 'is-active' : ''}
            onClick={() => setViewMode('trace')}
          >
            <span>≡</span> 轨迹列表
          </button>
        </div>
        <span className="eval-view-hint">
          {viewMode === 'timeline'
            ? '共享时间标尺 · 点击节点查看证据'
            : viewMode === 'graph'
              ? '关系视图 · 点击节点查看资源'
              : '展开节点可见完整资源与执行产出'}
        </span>
      </div>

      {viewMode === 'timeline' && (
        <TimelineView
          nodes={nodes}
          selectedId={selectedNodeId}
          onSelect={setSelectedNodeId}
          isRunning={isRunning}
          totalDurationMs={flow.metrics.total_duration_ms}
        />
      )}

      {viewMode === 'graph' && (
        <GraphView
          nodes={nodes}
          selectedId={selectedNodeId}
          onSelect={setSelectedNodeId}
          isRunning={isRunning}
        />
      )}

      {viewMode === 'trace' && (
        <div className="eval-flow-layout">
          <section className="eval-trace-column">
            <div className="eval-section-heading">
              <span>执行节点</span>
              <span>{flow.metrics.config_revision ? `config ${flow.metrics.config_revision}` : 'trace ordered by seq'}</span>
            </div>
            <div className="eval-node-list">
              {nodes.map(node => <NodeCard key={node.id} node={node} />)}
            </div>
          </section>

          <aside className="eval-inspector-column">
            <div className="eval-inspector-card">
              <div className="eval-inspector-title">实时指标</div>
              <dl className="eval-inspector-list">
                <div><dt>任务状态</dt><dd>{taskState}</dd></div>
                <div><dt>意图</dt><dd>{flow.metrics.intent || '-'}</dd></div>
                <div><dt>执行模式</dt><dd>{flow.metrics.mode || 'auto'}</dd></div>
                <div><dt>Prompt tokens</dt><dd>{flow.metrics.tokens.prompt || 0}</dd></div>
                <div><dt>Completion</dt><dd>{flow.metrics.tokens.completion || 0}</dd></div>
                <div><dt>估算调用</dt><dd>{flow.metrics.estimated_calls}/{flow.metrics.llm_calls}</dd></div>
                <div><dt>成本</dt><dd>{flow.metrics.cost_usd == null ? '-' : `$${fmtNumber(flow.metrics.cost_usd, 4)}`}</dd></div>
              </dl>
            </div>

            <div className="eval-inspector-card">
              <div className="eval-inspector-title">上下文窗口</div>
              <dl className="eval-inspector-list">
                <div><dt>已用</dt><dd>{String(contextPack.estimated_tokens ?? contextData.estimated_tokens ?? '-')}</dd></div>
                <div><dt>预算</dt><dd>{String(contextPack.max_tokens ?? contextData.max_tokens ?? '-')}</dd></div>
                <div><dt>利用率</dt><dd>{fmtNumber(contextPack.utilization ?? contextData.utilization, 3)}</dd></div>
                <div><dt>摘要压缩</dt><dd>{String(contextData.summary_used ?? false)}</dd></div>
                <div><dt>已截断</dt><dd>{String(contextPack.truncated ?? contextData.truncated ?? false)}</dd></div>
              </dl>
            </div>

            <div className="eval-inspector-card">
              <div className="eval-inspector-title">配置快照</div>
              <dl className="eval-inspector-list">
                <div><dt>revision</dt><dd>{flow.metrics.config_revision || '-'}</dd></div>
                <div><dt>hash</dt><dd>{flow.metrics.config_hash || '-'}</dd></div>
                {Object.entries(flow.metrics.prompt_versions ?? {}).slice(0, 8).map(([key, value]) => (
                  <div key={key}><dt>{key}</dt><dd>{value}</dd></div>
                ))}
              </dl>
            </div>

            {flow.context_snapshot && (
              <div className="eval-inspector-card">
                <div className="eval-inspector-title">中间上下文</div>
                <div style={{ ...mono, maxHeight: 180, overflow: 'auto', color: 'var(--color-text-secondary)' }}>
                  {short(flow.context_snapshot, 1600)}
                </div>
              </div>
            )}
          </aside>
        </div>
      )}
    </div>
  );
}

export default SingleFlowView;
