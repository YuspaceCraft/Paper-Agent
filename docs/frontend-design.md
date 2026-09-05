# 前端界面设计稿

> **对话中心化更新（2026-09-05，对应 docs/conversation-centric-workspace.md）**：
> 右侧 420px 面板从「独立 PDF 阅读器 / 全屏工作区」改为 **WorkspacePanel（右侧工作台）**，
> 三 Tab：📄 文档（DocPanel，绑定对话 active doc）/ 🧪 实验（ExperimentPanel，绑定对话
> project）/ 📁 文件（FileExplorer）。**TopBar 的「文献问答/论文写作/实验」切换已删除**
> （冗余——Panel 自身 Tab 承担）；Upload PDF 移入聊天输入框（ChatInput 上传图标）。
> **对话主区永不被替换**。
> 对话·工件绑定由 SSE 事件归因（`doc_section` / `experiment` 事件 → ThreadMeta.docId /
> project，localStorage 持久化）。写作/实验内联状态片渲染在 assistant 气泡内
> （Message.workNotes）。本文件下方旧设计（独立 PDF 阅读器 RightPanel）保留作参考。

## 一、页面布局

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TopBar                                                                  │
│  [☰]  [📚 Demo]                              [⬆ Upload] [🟢 API/Qdrant] │
├──────────────┬─────────────────────────────────┬─────────────────────────┤
│              │                                 │                         │
│  LeftPanel   │        MainContent              │  RightPanel             │
│  260px       │        flex: 1                  │  420px                  │
│  可折叠      │                                 │  可折叠                  │
│              │  ┌───────────────────────────┐  │                         │
│  ┌────────┐  │  │                           │  │  ┌───────────────────┐  │
│  │+ 新对话 │  │  │  ┌─────────────────────┐  │  │  │ PDF 下拉选择器    │  │
│  └────────┘  │  │  │ 系统消息 (左侧)      │  │  │  │ [RMNet.pdf ▾][×]  │  │
│              │  │  │ RMNet proposes a...  │  │  │  └───────────────────┘  │
│  ┌────────┐  │  │  │ [RMNet, page 5]     │  │  │                         │
│  │ 对话 1  │  │  │  └─────────────────────┘  │  │  ┌─── PDF Tabs ──────┐  │
│  │ RMNet   │  │  │                           │  │  │ RMNet | BLIP | +  │  │
│  │ 7/21    │  │  │        ┌───────────────┐  │  │  ├───────────────────┤  │
│  └────────┘  │  │        │ 用户消息 (右侧) │  │  │  │                   │  │
│              │  │        │ What is the... │  │  │  │   PDF Page 5/12   │  │
│  ┌────────┐  │  │        └───────────────┘  │  │  │                   │  │
│  │ 对话 2  │  │  │                           │  │  │  ┌─────────────┐  │  │
│  │ BLIP-CC │  │  │  ┌─────────────────────┐  │  │  │  │highlighted  │  │  │
│  │ 7/20    │  │  │  │ 系统消息 (左侧)      │  │  │  │  │chunk region │  │  │
│  └────────┘  │  │  │ Thinking... ✨       │  │  │  │  └─────────────┘  │  │
│              │  │  │ 正在搜索 RMNet loss  │  │  │  │                   │  │
│  ┌────────┐  │  │  └─────────────────────┘  │  │  └───────────────────┘  │
│  │ 对话 3  │  │  │                           │  │                         │
│  │ MV-CC   │  │  │  ─────────────────────── │  │                         │
│  │ 7/18    │  │  │                           │  │                         │
│  └────────┘  │  │  ┌───────────────────────┐│  │                         │
│              │  │  │ 输入框        [⏹ 停止] ││  │                         │
│              │  │  │ (streaming时禁用发送)   ││  │                         │
│              │  │  └───────────────────────┘│  │                         │
├──────────────┴─────────────────────────────────┴─────────────────────────┤
│  StatusBar                                                                │
│  Qdrant: 123 chunks | 5 papers | Agent: qwen-plus                         │
└──────────────────────────────────────────────────────────────────────────┘
```

### 三栏职责

| 区域 | 宽度 | 核心职责 | 折叠行为 |
|------|------|---------|---------|
| **LeftPanel** | 260px | 对话线程管理（历史列表 + 新建） | 点击 ☰ 或拖拽左边界折叠 |
| **MainContent** | flex:1 | 聊天消息展示 + 输入 | 始终可见 |
| **RightPanel** | 420px | PDF 阅读器（下拉选文 + 标签切换） | 点击右边界按钮折叠 |

### LeftPanel 详细

```
┌──────────────────────┐
│  🔍 搜索对话...       │  ← 可选：过滤历史对话
│                      │
│  [+ 新建对话]         │  ← 始终在顶部，醒目按钮
│  ─────────────────── │
│                      │
│  ┌──────────────────┐ │
│  │ 📄 RMNet Loss    │ │  ← 对话标题（自动取首条消息摘要）
│  │ 3 条消息   7/21  │ │
│  └──────────────────┘ │
│                      │
│  ┌──────────────────┐ │
│  │ 📄 Change Det... │ │
│  │ 5 条消息   7/20  │ │
│  └──────────────────┘ │
│                      │
│  ┌──────────────────┐ │
│  │ 📄 (空对话)       │ │  ← 新建但未发送消息
│  │ 0 条消息   7/21  │ │
│  └──────────────────┘ │
│                      │
└──────────────────────┘
```

- 对话按最近活跃时间倒序排列
- 点击某条对话 → MainContent 切换到该 thread_id 的消息历史
- 点击「新建对话」→ 生成新 thread_id → 清空 MainContent → 聚焦输入框
- 空对话（未发送消息）7 天自动清理
- 对话数据存储：`localStorage`，key 为 `thread_id`，value 为 `{title, messages[], createdAt, updatedAt}`

### MainContent 详细

```
┌─────────────────────────────────────────┐
│  (对话标题，可选)                         │
│  ─────────────────────────────────────── │
│                                         │
│  ┌─────────────────────┐                │
│  │ 🤖 系统              │                │
│  │                     │                │
│  │ RMNet proposes a    │                │
│  │ dual-stream remote  │                │
│  │ sensing change      │                │
│  │ detection framework │                │
│  │                     │                │
│  │ [RMNet, page 5] ←──→│  可点击引用     │
│  └─────────────────────┘                │
│                                         │
│              ┌───────────────────┐      │
│              │        用户 👤     │      │
│              │ What is the loss  │      │
│              │ function of RMNet?│      │
│              └───────────────────┘      │
│                                         │
│  ┌─────────────────────┐                │
│  │ 🤖 系统 (streaming)  │                │
│  │                     │                │
│  │ The loss function   │                │
│  │ combines cross-     │  ← 逐词追加     │
│  │ entropy and a       │                │
│  │ contrastive... ▌    │  ← 光标闪烁     │
│  └─────────────────────┘                │
│                                         │
│  ─────────────────────────────────────── │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ 输入你的问题...          [⏹ 停止]│    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

**消息对齐规则：**
- 用户消息：右对齐，浅蓝/浅灰背景
- 系统消息：左对齐，白色/浅绿背景，带 Agent 头像标识
- 流式消息：左对齐，带闪烁光标动画，实时逐词追加

**输入框行为：**
- 默认状态：输入框可编辑，右侧为发送按钮（➤）
- Agent 流式输出中（`isStreaming === true`）：
  - 输入框 + 发送按钮 **禁用**（灰色 + `disabled`）
  - 右侧显示红色 **⏹ 停止按钮**，点击 → abort SSE 连接
  - 按 Enter 不触发发送
  - 停止后当前回答保留不完整内容，用户可以重新发送
- 停止后输入框恢复可用，停止按钮变回发送按钮

**系统消息内 Markdown 渲染：**
- 代码块（```）→ 语法高亮 + 复制按钮
- 行内公式/公式块 → KaTeX 渲染（react-katex）
- 引用标记 `[PaperName, page N]` → 可点击链接，点击触发 PDF 定位
- 表格 → 自适应表格样式

### RightPanel 详细

```
┌──────────────────────────────────────┐
│  PDF 论文浏览                         │
│  ┌──────────────────────────────────┐│
│  │ 选择 PDF: [RMNet.pdf      ▾] [×] ││  ← 下拉列表 + 关闭当前
│  └──────────────────────────────────┘│
│                                      │
│  ┌─── Tabs ─────────────────────────┐│
│  │ RMNet.pdf │ BLIP-CC.pdf │ [+ 打开]││  ← 标签页切换
│  ├──────────────────────────────────┤│
│  │                                  ││
│  │  ⬆ ⬇  Page 5 / 12    [🔍 缩放]  ││  ← 工具栏
│  │                                  ││
│  │  ┌──────────────────────────┐   ││
│  │  │                          │   ││
│  │  │     PDF 页面渲染          │   ││
│  │  │                          │   ││
│  │  │  ┌──────────────────┐    │   ││  ← chunk 高亮区域
│  │  │  │ highlight overlay │    │   ││     (黄色半透明)
│  │  │  └──────────────────┘    │   ││
│  │  │                          │   ││
│  │  └──────────────────────────┘   ││
│  │                                  ││
│  └──────────────────────────────────┘│
└──────────────────────────────────────┘
```

**PDF 加载流程：**
1. GET `/api/pdf/outputs` → 获取所有已处理论文列表（paper_name）
2. 下拉选择器列出所有 paper_name，选中后：
   - 新建/切换到对应 tab
   - 通过 paper_name 定位 PDF 文件路径（约定：`pdf_pipeline/output/{paper_name}/` 目录下的源 PDF 或 `data/uploads/{paper_name}.pdf`）
   - 若 PDF 不直接可访问，后端需提供 `/api/pdf/outputs/{paper_name}/file` 端点返回 PDF 二进制

**PDF 标签页：**
- 每个打开的 PDF 一个 tab，显示 paper_name
- 右侧 `[+ 打开]` 等价于下拉选择器
- 可关闭 tab（×），关闭后自动切换到相邻 tab
- 至少保留 0 个 tab（全部关闭时显示空白占位提示 "选择一篇论文开始阅读"）
- 最多同时打开 5 个 tab（超出时提示关闭其他 tab）

**引用联动 → PDF：**
当用户在聊天中点击引用 `[RMNet, page 5]`：
1. RightPanel 切换到 RMNet 的 tab（如未打开则自动打开）
2. GET `/api/reader/position/RMNet/{chunk_id}` → `{page_no, bbox}`
3. PDF 滚动到 page 5
4. 在 page 5 上按 bbox 坐标渲染高亮覆盖层（黄色半透明矩形）
5. 高亮 3 秒后渐隐（或手动关闭）

**对比阅读：**
- 两个 tab 并排显示时（通过拖拽 tab 到分屏区域），左右并排渲染两份 PDF
- 适用于 agent 回答中同时引用两篇论文的场景

---

## 二、功能实现

### 2.1 组件树

```
<App>
  <TopBar>
    <ToggleLeftPanel />       // ☰ 按钮，折叠/展开 LeftPanel
    <Logo />                  // "Demo" 品牌标识
    <UploadButton />          // 上传 PDF
    <ServiceIndicator />      // API / Qdrant 状态灯
    <ToggleRightPanel />      // 折叠/展开 RightPanel
  </TopBar>

  <Layout>                    // CSS Grid: left main right
    <LeftPanel>
      <SearchThreads />       // 可选：搜索过滤对话
      <NewThreadButton />     // "+ 新建对话"
      <ThreadList>            // 对话列表
        <ThreadItem>          // 单条对话摘要（标题 + 消息数 + 时间）
      </ThreadList>
    </LeftPanel>

    <MainContent>
      <MessageList>           // 消息滚动区
        <SystemMessage>       // 左对齐，markdown + 引用
          <MarkdownRenderer />
          <CitationLink />    // 可点击引用跳转
        </SystemMessage>
        <UserMessage>         // 右对齐
        </UserMessage>
        <StreamingMessage>    // 流式输出中（闪烁光标）
          <ThinkingStatus />  // 工具调用进度条
        </StreamingMessage>
      </MessageList>
      <ChatInput>             // 底部固定
        <TextArea />          // 输入框
        <SendButton />        // 发送 (idle) / 停止 (streaming)
      </ChatInput>
    </MainContent>

    <RightPanel>
      <PDFSelector />         // 下拉选择论文
      <PDFTabs>               // 标签页容器
        <PDFTab />            // 单个 tab → PDF 渲染
          <PDFToolbar />      // 翻页 + 缩放
          <PDFCanvas />       // pdfjs render
          <HighlightOverlay />// chunk bbox 高亮层
      </PDFTabs>
    </RightPanel>
  </Layout>

  <StatusBar>
    <IndexInfo />             // Qdrant chunks / papers 统计
    <AgentInfo />             // 当前模型名
    <UploadTask />            // 进行中的上传任务
  </StatusBar>
</App>
```

### 2.2 核心功能模块

#### A. 对话线程管理 (LeftPanel)

**数据存储：**
```typescript
// localStorage key: "demo_threads"
interface ThreadStore {
  threads: Record<string, ThreadMeta>;  // thread_id → meta
  order: string[];                       // thread_id 排序列表
}

interface ThreadMeta {
  title: string;           // 自动截取首条用户消息前 30 字符
  messageCount: number;
  createdAt: string;       // ISO
  updatedAt: string;       // ISO
}
```

**行为：**
- 新建对话 → `thread_id = crypto.randomUUID()` → 写入 localStorage → 追加到列表顶部 → 切换到新对话
- 发送首条消息后 → 更新 title（取用户消息前 30 字符）
- 切换对话 → 从 localStorage 加载 `messages_{thread_id}` → 渲染 MessageList
- 消息持久化：每条消息存为 `messages_{thread_id}` key，value 为 `Message[]` JSON

#### B. 聊天界面 (MainContent)

**消息数据结构：**
```typescript
interface Message {
  id: string;
  role: 'user' | 'system';
  content: string;
  citations?: Citation[];      // 仅 system 消息
  status?: 'complete' | 'streaming' | 'aborted';
  timestamp: string;
}

interface Citation {
  paper_name: string;
  page_no: number;
  chunk_id: string;
  relevant_text: string;
}
```

**流式输出处理（SSE）：**
```
POST /api/agent/chat/stream { query, thread_id }
  │
  ├─ status/understand  → MessageList 追加占位系统消息，status=streaming
  ├─ status/agent       → 显示工具调用状态条 "正在搜索: RMNet loss..."
  ├─ status/tools       → 状态条更新 "搜索完成"
  ├─ status/synthesize  → 状态条消失
  ├─ token × N          → 逐词追加到系统消息 content 末尾
  ├─ done               → 标记 status=complete，渲染 citations
  └─ error              → 标记 status=aborted，显示错误提示
```

**输入限制规则：**
| 状态 | 输入框 | 发送按钮 | 停止按钮 | Enter 键 |
|------|--------|---------|---------|----------|
| idle（无流式输出） | 可编辑 | 显示 ➤，可用 | 隐藏 | 发送 |
| streaming（流式输出中） | disabled + 灰色 | 隐藏 | 显示 ⏹，红色 | 无操作 |
| stream 刚结束 | 可编辑 | 显示 ➤，可用 | 隐藏 | 发送 |

**停止处理：**
```typescript
function handleStop() {
  abortController.abort();       // 断开 SSE
  // 当前系统消息标记为 status='aborted'
  // 保留已输出的部分内容
  // 输入框恢复可用
}
```

#### C. PDF 阅读器 (RightPanel)

**PDF 来源：**
需要后端新增一个端点返回 PDF 文件：

```
GET /api/pdf/outputs/{paper_name}/file
→ Response: application/pdf (binary)
```

或前端通过约定路径直接访问：
```
GET /api/pdf/file/{paper_name}
→ 查找 pdf_pipeline/output/{paper_name}/ 下的 .pdf 或 data/uploads/{paper_name}.pdf
```

建议方案：后端新增端点，由 API 层处理文件定位逻辑。

**PDF 渲染：**
- 使用 `pdfjs-dist` 按页渲染到 `<canvas>`
- 虚拟滚动只渲染当前页 ± 1 页（减少内存）
- 缩放：预设 75% / 100% / 125% / 150% / 适合宽度

**引用高亮覆盖层：**
```typescript
interface HighlightRegion {
  page_no: number;
  bbox: { x0: number; y0: number; x1: number; y1: number };
  chunk_id: string;
}

// bbox 坐标需从 PDF 坐标转换为 canvas 坐标
// PDF 坐标系原点在左下角，canvas 原点在左上角，需翻转 y
function pdfBboxToCanvas(bbox: PDFBbox, pageHeight: number, scale: number): CanvasRect {
  return {
    left: bbox.x0 * scale,
    top: (pageHeight - bbox.y1) * scale,  // y 轴翻转
    width: (bbox.x1 - bbox.x0) * scale,
    height: (bbox.y1 - bbox.y0) * scale,
  };
}
```

**标签页状态：**
```typescript
interface PDFTabState {
  paper_name: string;
  currentPage: number;
  scale: number;
  highlights: HighlightRegion[];  // 当前活跃的高亮区域
}
```

#### D. PDF 上传

- TopBar 的 Upload 按钮 → 文件选择器（accept=".pdf"）
- 也支持拖拽到 MainContent 区域（全局拖拽监听）
- 上传中显示进度条（StatusBar 或独立 toast）
- 完成后刷新 PDF 下拉选择器列表
- 去重提示（status=duplicate）

### 2.3 状态管理

```typescript
// AppState — 顶层状态
interface AppState {
  // 对话
  threads: Record<string, ThreadMeta>;
  threadOrder: string[];
  activeThreadId: string | null;
  messages: Message[];
  isStreaming: boolean;

  // PDF
  pdfTabs: PDFTabState[];
  activeTabIndex: number;
  availablePapers: string[];     // GET /api/pdf/outputs

  // UI
  leftPanelOpen: boolean;
  rightPanelOpen: boolean;

  // 系统
  indexStats: IndexStats | null;  // GET /api/index/stats
  agentHealth: AgentHealth | null; // GET /api/agent/health
  uploadTasks: string[];           // 进行中的 task_id 列表
}

// 动作
type Action =
  | { type: 'CREATE_THREAD'; threadId: string }
  | { type: 'SWITCH_THREAD'; threadId: string }
  | { type: 'DELETE_THREAD'; threadId: string }
  | { type: 'ADD_MESSAGE'; threadId: string; message: Message }
  | { type: 'APPEND_TOKEN'; threadId: string; messageId: string; token: string }
  | { type: 'SET_STREAMING'; isStreaming: boolean }
  | { type: 'OPEN_PDF'; paperName: string }
  | { type: 'CLOSE_PDF'; tabIndex: number }
  | { type: 'SET_PDF_PAGE'; tabIndex: number; page: number }
  | { type: 'ADD_HIGHLIGHT'; tabIndex: number; highlight: HighlightRegion }
  | { type: 'CLEAR_HIGHLIGHTS'; tabIndex: number }
  | { type: 'TOGGLE_LEFT_PANEL' }
  | { type: 'TOGGLE_RIGHT_PANEL' }
  | { type: 'SET_INDEX_STATS'; stats: IndexStats }
  | { type: 'SET_AGENT_HEALTH'; health: AgentHealth };
```

### 2.4 技术选型

| 层 | 选型 | 理由 |
|---|------|------|
| 框架 | React 18 + TypeScript | SSE + canvas 生态成熟 |
| 构建 | Vite | 零配置 HMR |
| 状态 | Context + useReducer | 无额外依赖，状态结构清晰 |
| 样式 | CSS Modules | 按组件隔离，无运行时开销 |
| HTTP | fetch + ReadableStream | 原生 SSE 消费，无需 EventSource（EventSource 不支持 POST） |
| PDF 渲染 | pdfjs-dist | 标准方案 |
| Markdown | react-markdown + rehype-katex | GFM + 公式 |
| 语法高亮 | react-syntax-highlighter | 代码块渲染 |
| 持久化 | localStorage | 对话历史本地存储，无服务端依赖 |

---

## 三、前后端交互协议

### 3.1 现有端点（无需修改）

| 端点 | 用途 | 调用时机 |
|------|------|---------|
| `POST /api/agent/chat/stream` | SSE 流式问答 | 用户发送消息 |
| `GET /api/agent/health` | agent 状态 | 页面加载 + 定时心跳 |
| `GET /api/pdf/outputs` | 论文列表 | 页面加载 + 上传完成后刷新 |
| `POST /api/pdf/process` | 上传 PDF | 用户点击上传 |
| `GET /api/pdf/status/{id}` | 上传进度 | 上传后轮询 |
| `GET /api/reader/{paper}/abstract` | 论文元数据 | 可选：tab hover 预览 |
| `GET /api/reader/position/{paper}/{chunk_id}` | chunk→PDF 位置 | 点击引用时查询 |
| `GET /api/index/stats` | 索引统计 | 页面加载 + StatusBar |

### 3.2 需要新增的端点

**1. 获取 PDF 文件二进制**

```
GET /api/pdf/file/{paper_name}
→ Content-Type: application/pdf
→ 响应体: PDF 二进制流
```

实现思路（在 `routers/pdf.py` 新增）：

```python
@router.get("/file/{paper_name}")
async def get_pdf_file(paper_name: str):
    """Return the original PDF file for a processed paper."""
    # 优先找 output 目录下的源文件
    # 其次找 uploads 目录
    # 返回 FileResponse
```

优先级查找路径：
1. `pdf_pipeline/output/{paper_name}/` 目录下的 `.pdf` 文件
2. `data/uploads/{paper_name}.pdf`
3. 都不存在 → 404

**2. 获取单篇论文的所有 chunk 位置（批量）**

```
GET /api/reader/{paper_name}/positions
→ { paper_name, positions: { chunk_id: { page_no, bbox }, ... } }
```

用途：打开 PDF 时一次性加载所有 chunk 位置，后续引用点击直接查本地缓存，无需逐个调 API。

实现思路（在 `routers/reader.py` 新增）：

```python
@router.get("/{paper_name}/positions")
async def get_all_positions(paper_name: str):
    """Return all chunk→page positions for a paper."""
    r = _get_redis()
    if r:
        all_pos = r.hgetall(f"pos:{paper_name}")
        if all_pos:
            return {
                "paper_name": paper_name,
                "positions": {k: json.loads(v) for k, v in all_pos.items()},
            }
    # fallback to chunk_positions.json
    path = OUTPUT_DIR / paper_name / "chunk_positions.json"
    if path.exists():
        return {
            "paper_name": paper_name,
            "positions": json.loads(path.read_text("utf-8")),
        }
    raise HTTPException(404, f"No positions for '{paper_name}'")
```

### 3.3 API 调用映射总表

```
前端操作                              → HTTP 调用                               SSE
──────────────────────────────────────────────────────────────────────────────────
页面初始化                             → GET /api/pdf/outputs
                                      → GET /api/index/stats
                                      → GET /api/agent/health
选择/新建对话                          → localStorage 读取 messages_{thread_id}
发送消息                               → POST /api/agent/chat/stream             ✓
停止流式输出                           → abortController.abort()
切换对话                               → localStorage 读取/写入
上传 PDF                               → POST /api/pdf/process (FormData)
轮询上传进度                           → GET /api/pdf/status/{task_id}
打开 PDF 标签页                        → GET /api/pdf/file/{paper_name}
                                      → GET /api/reader/{paper_name}/positions
点击引用 [RMNet, page 5]              → 查本地 positions 缓存 → scrollToPage + highlight
                                      → (缓存未命中时) GET /api/reader/position/RMNet/{chunk_id}
```

### 3.4 关键交互流程

#### 流程 A：发送消息 → 流式回答 → 引用联动

```
1. 用户输入 "What is RMNet's loss function?" → Enter
2. isStreaming = true → 输入框禁用，显示停止按钮
3. POST /api/agent/chat/stream { query, thread_id }

4. SSE 事件流：
   data: {"type":"status","node":"understand","data":{"intent":"paper_qa","entities":["RMNet","loss function"]}}
   → 消息区追加系统消息气泡，status=streaming，内容为空

   data: {"type":"status","node":"agent","data":{"tool_calls":[{"name":"search_literature","args":{...}}]}}
   → 气泡内显示 "🔍 正在检索相关文献..."

   data: {"type":"status","node":"tools","data":{"completed":true}}
   → 状态文字更新为 "检索完成"

   data: {"type":"status","node":"agent","data":{"tool_calls":[{"name":"read_paper_section","args":{...}}]}}
   → "📖 正在阅读 RMNet Methodology 章节..."

   data: {"type":"status","node":"tools","data":{"completed":true}}
   → "阅读完成"

   data: {"type":"status","node":"synthesize"}
   → 状态文字消失，开始显示 token

   data: {"type":"token","content":" RMNet"}
   data: {"type":"token","content":" uses"}
   data: {"type":"token","content":" a"}
   ...
   → 逐词追加到气泡内容区，带闪烁光标

   data: {"type":"done","citations":[{...}]}
   → isStreaming = false，输入框恢复，渲染引用链接

5. 用户点击 [RMNet, page 5] 引用链接：
   a. 从本地 positions 缓存查 chunk_id → { page_no: 5, bbox: {...} }
   b. RightPanel 切换到 RMNet tab（未打开则自动打开）
   c. pdfjs 滚动到 page 5
   d. 渲染 bbox 高亮覆盖层，3 秒后渐隐
```

#### 流程 B：上传新论文

```
1. TopBar 点击 Upload → 文件选择器
2. 用户选择 new_paper.pdf
3. POST /api/pdf/process (FormData)
   → { task_id: "abc123", paper_name: "new_paper", status: "pending" }
4. StatusBar 显示 "上传中: new_paper.pdf..."
5. 轮询 GET /api/pdf/status/abc123 (每 2s)
6. status: done → toast "处理完成: 42 chunks"
   status: duplicate → toast "论文已存在"
   status: failed → toast "处理失败: {error}"
7. 刷新 availablePapers 列表
```

#### 流程 C：多 PDF 对比阅读

```
1. Agent 回答中同时引用 [RMNet, page 5] 和 [BLIP-CC, page 3]
2. 用户先点击 [RMNet, page 5] → RightPanel 打开 RMNet tab
3. 用户再点击 [BLIP-CC, page 3] → 新增 BLIP-CC tab，自动激活
4. 用户拖动 BLIP-CC tab 到右侧分屏区域 → 并排显示两张 PDF
5. 各自高亮对应的 chunk 区域
```

### 3.5 错误处理

| 场景 | 处理 |
|------|------|
| API 不可达 | TopBar 状态灯变红，toast "后端连接失败"，不影响本地对话历史查看 |
| SSE 连接中断 | 保留已输出内容，消息标记 aborted，输入框恢复 |
| SSE 中途用户停止 | abort → 消息标记 aborted，保留部分内容 |
| PDF 文件 404 | tab 内显示 "PDF 文件不可用" 占位图 |
| localStorage 满 | toast "本地存储空间不足，请清理历史对话"（QuotaExceededError） |
| 上传超大 PDF (>50MB) | 前端校验文件大小，超限阻止上传 |
| 轮询超时（5 分钟仍 pending） | 停止轮询 → toast "处理超时，请检查后端日志" |

---

## 四、实现优先级

| 优先级 | 模块 | 核心文件 | 说明 |
|--------|------|---------|------|
| **P0** | Vite + React 脚手架 | `web/frontend/` | 项目初始化，三栏 CSS Grid 布局骨架 |
| **P0** | MainContent + SSE | `ChatView.tsx`, `useSSE.ts` | 核心：发送消息 → 流式接收 → 逐词渲染 |
| **P0** | ChatInput | `ChatInput.tsx` | 输入框 + 发送/停止按钮状态切换 |
| **P1** | LeftPanel + 对话管理 | `LeftPanel.tsx`, `useThreads.ts` | localStorage 持久化，新建/切换/删除对话 |
| **P1** | RightPanel + PDF 选择 | `RightPanel.tsx`, `PDFSelector.tsx` | 下拉选择器 + 后端 PDF file 端点 |
| **P1** | PDF 标签页 + 渲染 | `PDFTabs.tsx`, `PDFViewer.tsx` | pdfjs-dist 渲染 + 翻页/缩放 |
| **P2** | 引用 → PDF 联动 | `CitationLink.tsx`, `HighlightOverlay.tsx` | 点击引用 → 打开 PDF → 滚动 + 高亮 |
| **P2** | 后端: PDF file 端点 | `web/api/routers/pdf.py` | `GET /api/pdf/file/{paper_name}` |
| **P2** | 后端: 批量 positions 端点 | `web/api/routers/reader.py` | `GET /api/reader/{paper_name}/positions` |
| **P2** | PDF 上传 + 进度 | `UploadButton.tsx` | 文件选择 + 轮询进度 |
| **P3** | 对比阅读 (分屏) | `CompareView.tsx` | 双 PDF 并排渲染 |
| **P3** | Markdown + 公式渲染 | `MarkdownRenderer.tsx` | KaTeX + 语法高亮 |
| **P3** | TopBar + StatusBar | `TopBar.tsx`, `StatusBar.tsx` | Logo、状态灯、统计信息 |

---

## 五、后端待补端点

前端开发前需在 `web/api/` 补两个端点：

### 5.1 GET `/api/pdf/file/{paper_name}`

`routers/pdf.py` 新增：

```python
from fastapi.responses import FileResponse

@router.get("/file/{paper_name}")
async def get_pdf_file(paper_name: str):
    """返回原始 PDF 文件"""
    # 1. output 目录下查找
    output_dir = OUTPUT_DIR / paper_name
    if output_dir.is_dir():
        for f in output_dir.iterdir():
            if f.suffix.lower() == ".pdf":
                return FileResponse(str(f), media_type="application/pdf")

    # 2. uploads 目录查找
    upload_pdf = UPLOAD_DIR / f"{paper_name}.pdf"
    if upload_pdf.exists():
        return FileResponse(str(upload_pdf), media_type="application/pdf")

    raise HTTPException(404, f"PDF file not found for '{paper_name}'")
```

### 5.2 GET `/api/reader/{paper_name}/positions`

`routers/reader.py` 新增：

```python
@router.get("/{paper_name}/positions")
async def get_all_positions(paper_name: str):
    """返回论文所有 chunk 的 PDF 位置映射"""
    r = _get_redis()
    if r:
        raw = r.hgetall(f"pos:{paper_name}")
        if raw:
            return {
                "paper_name": paper_name,
                "positions": {k: json.loads(v) for k, v in raw.items()},
            }

    path = OUTPUT_DIR / paper_name / "chunk_positions.json"
    if path.exists():
        return {
            "paper_name": paper_name,
            "positions": json.loads(path.read_text("utf-8")),
        }

    raise HTTPException(404, f"No positions for '{paper_name}'")
```
