# Demo Agent — Electron 桌面客户端

把原来的 React 网页 agent UI 改造成**本地启动的桌面客户端**。主进程拉起 Python 后端（uvicorn）作为子进程，渲染进程（React）通过 HTTP + SSE 和后端通信。本地文件/系统权限由后端通用工具集承载，不再受浏览器沙箱限制。

## 架构

```
Electron 主进程 (src/main/)
  ├── index.ts    # 窗口 + 单实例锁 + IPC + before-quit→stop
  ├── backend.ts  # spawn `python -m uvicorn web.api.main:app` + /api/health 探活
  └── preload.ts  # contextBridge 暴露 window.electronAPI（端口/状态/日志）

Python 后端（子进程，uvicorn @ 127.0.0.1:8001）
  └── web.api.main:app  （FastAPI + SSE 流式聊天）

渲染进程 (src/renderer/)  —— React + Vite
  └── 通过 HTTP API 与后端通信，SSE 流式返回聊天/日志
```

渲染进程不直接 `import` 后端模块，全程走 HTTP（符合 CLAUDE.md FastAPI 封装原则）。

## 快速开始

前置：目标机已装 `demo` conda 环境（Python 后端**不打包**，从 PATH 启动）。

```bash
# 1. 安装依赖（China 网络用镜像，见 TROUBLESHOOTING）
npm install

# 2. 开发模式：Vite(5173) + Electron（主进程拉起 uvicorn@8001）
npm run dev
```

> 桌面 app 从快捷方式启动时不继承 `conda activate`。若 `python` 不是 demo 环境，设置
> `DEMO_PYTHON=C:/Users/30811/miniconda3/envs/demo/python.exe` 再启动，或把 demo 环境加进 PATH。

## 脚本

| 命令 | 说明 |
|------|------|
| `npm run dev` | 开发：并发跑 Vite + Electron |
| `npm run dev:renderer` | 只跑 Vite（纯浏览器调试，需手动 `uvicorn ... --port 8000`） |
| `npm run dev:main` | 编译主进程 + 启动 Electron |
| `npm run build` | 构建 renderer(Vite) + main(tsc) → `dist/` |
| `npm run dist:win` | electron-builder 打 Windows nsis 安装器 → `release/` |
| `npm run dist:mac` | electron-builder 打 macOS dmg/zip |

## 目录

```
web/frontend/
  package.json        # scripts + main + electron-builder build 字段
  vite.config.ts      # root: src/renderer, base: './', outDir: dist/renderer
  tsconfig.json       # 渲染进程（app + node 引用）
  tsconfig.main.json  # 主进程 tsc（CommonJS → dist/main）
  src/main/           # Electron 主进程
  src/renderer/       # React 渲染进程（index.html + src/）
```

## 聊天区

聊天区采用 CowAgent 风格浅色主题（绿色强调 `#4abe6e`），通过 `index.css` 的 `--color-*` 变量级联，三栏布局自动跟随换色。助理消息带头像 + 圆角气泡，正文用 `react-markdown` + `react-syntax-highlighter` 渲染。

工具执行过程在对话层可见：后端 `/api/agent/chat/stream` 发 `tool_start`/`tool_end` 结构化事件（见 [web/api/README.md](../api/README.md)），前端解析为**默认折叠**的步骤卡片（`components/MessageSteps.tsx`）——每行只显示状态（spinner/✓/✗）+ 工具名 + 耗时，点击才展开入参/结果；子代理步骤带「子代理」徽标。

**任务区域（`components/TaskCenter.tsx`）**：聊天区**上方固定组件**（独立于消息滚动区，`ChatView` 为 `flex:1` 布局不干扰其高度），**默认折叠**、点击标题展开；**有新的运行中任务时自动展开**，折叠态也始终显示运行/历史摘要行。所有后台任务 —— agent 入库（`/api/agent/ingest`）、Web 上传解析（`/api/pdf/process`）、索引（`/api/index/run`）统一在此展示，底部状态栏不再重复显示进度。展示分 3 态：`运行中`（pending/running）/ `完成` / `失败`，两区纯派生：
- **当前任务**：本对话 运行中/排队中（含系统级运行中任务），恒显示进度文案 + 不确定进度条（`taskbar-slide`）；运行中超 20 分钟无更新提示「可能已中断」。
- **历史任务**：本对话 已完成/失败，最新在前，可关闭 ×，随对话删除清空。

任务消息来源：`/api/agent/tasks/stream`（SSE 实时推送，主通道）+ 30s poll（`/api/agent/tasks` 兜底合并，`TASK_SYNC` 不回删已有条目）。任务**跟随对话绑定**：`threadId` 由前端归因（上传发起时、`ingest_paper` 工具返回里提取 task_id 时记录权威归属，SSE first-seen 兜底），归因 + 历史持久化于 localStorage（`demo_bg_tasks`）；删除对话时已完成任务随之清除，运行中任务解绑为系统级继续展示。任务完成通知（`streamNotify()` → `/api/agent/notify/stream`）落到**任务所属对话**（非完成时激活的对话），归属对话已删除则跳过。新增组件：`TaskCenter.tsx`；重写 `ChatView.tsx`、`StatusBar.tsx`、`state.ts`、`api.ts`、`App.tsx`。

## 关键决策

- **后端入口**：`python -m uvicorn web.api.main:app`（非 `app.py`，后者只在已弃用 `v1/`）。cwd = 项目根，因为 `checkpoints.db` / `pdf_pipeline/output` 都是相对路径。
- **固定端口 8001**：避开 dev 网页的 8000。碰撞再上动态选口。
- **Python 不打包**：安装包只含前端 + 主进程 JS，目标机预装 demo 环境。升级方向见 TROUBLESHOOTING 的 PyInstaller 打包风险。
- **就绪回填**：渲染进程首次 fetch 时后端可能还没 ready（docling 导入需数秒），通过 `onBackendStatus` 订阅 `ready` 事件 + 拉取回放重跑 init fetch。**启动竞态已收敛为单点 gate**：`api.ts` 的 `whenBackendReady()`（Electron 下 `ready` 事件 / 浏览器 dev 下立即 resolve）——`App.tsx` 与 `FileExplorer.tsx` 的初始拉取都 `await` 它，杜绝空 base URL 打到 Vite 代理造成 ECONNREFUSED、且一次性 fetch 卡死在错误态。
- **后端启动失败可见**：`backend-status` 事件非 `ready` 时（如 8001 端口被残留进程占用的 Errno 10048），顶部出红色 banner + 「重试启动后端」按钮（IPC `backend-restart` → `PythonBackend.start()` 复用重拉，事件照常回流），不再静默离线。

## 分发

```bash
npm run dist:win    # Windows nsis 安装器
npm run dist:mac    # macOS dmg/zip（需在 macOS 上构建）
```

产物在 `release/`。安装器不含 Python，目标机需预装 demo conda 环境。
