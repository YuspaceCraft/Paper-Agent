# Troubleshooting

项目开发中遇到的错误及解决方案，避免重复踩坑。

## 环境

### conda activate 在 bash 中无效

**现象**: `conda activate demo` 报错或无效。

**原因**: conda 未在 bash 中 init。

**解决**: 所有 bash 命令使用 Python 解释器直接路径：

```bash
C:/Users/30811/miniconda3/envs/demo/python.exe -m pip install ...
C:/Users/30811/miniconda3/envs/demo/python.exe script.py
```

终端中手动 `conda activate demo` 后可正常使用 `python` / `pip`。

---

### docling 导入导致 segfault (exit 139)

**现象**: `from docling import ...` 或 `from agent import ...`（触发 agent/__init__.py 导入链）导致 Segmentation fault (exit 139)。

**原因**: docling 的 onnxruntime DLL 与环境中其他库的 DLL 加载顺序冲突。

**解决**:
1. CLI 脚本直接用路径运行，不要用 `python -m` 导入（避免导入 `agent/__init__.py`）
2. 如果 segfault 出现在 bash terminal 而非 IDE，切换到 IDE 终端运行

```bash
# ❌ 不要这样
C:/Users/30811/miniconda3/envs/demo/python.exe -m pdf_pipeline.cli all paper.pdf

# ✅ 这样
C:/Users/30811/miniconda3/envs/demo/python.exe docling_cli.py chunk paper.pdf
```

---

### sentence_transformers 导入 segfault (exit 139 / 0xC0000005)

**现象**: `from sentence_transformers import SentenceTransformer` 直接 segfault，无 Python traceback。

**原因**: 与 docling segfault 同根——onnxruntime DLL 与 torch/CUDA DLL 冲突。bash terminal 环境特定，IDE 终端通常不受影响。

**解决**:
1. 在 IDE 终端（VSCode/PyCharm）中运行，而非 bash
2. 加载模型时使用 `local_files_only=True` 防止网络请求触发额外 DLL 加载
3. 无法避免时考虑将 embedding 放在独立子进程中执行

---

## pdf_pipeline

### onnxruntime 导入链冲突

**现象**: 任何涉及 `import agent` 或 `from agent import ...` 的脚本在特定环境下 segfault。

**原因**: `agent/__init__.py` 导入 docling → 触发 onnxruntime 加载，与 torch/CUDA 冲突。

**解决**:
- `pdf_pipeline/` 模块零依赖 `agent/`，可直接导入
- CLI 脚本 `docling_cli.py` 在项目根目录直接运行
- 永远不要在 pdf_pipeline 或 indexer 中导入 agent 模块

---

### enhancer 阶段无限挂起（缺少请求超时）

**现象**: 入库任务在 `parse` 阶段停在 "Enhancing formulas & images..." 长达 10 分钟，任务状态无更新、输出目录为空。

**原因**: `enhancer.py::_get_client` 创建 OpenAI 客户端时**未设置 `timeout`**，默认 600s；上游 DashScope 无响应时单次调用挂死（`agent/nodes.py` 已用 `request_timeout=120.0` 避免了同样问题）。

**解决**: `enhancer.py` 客户端补 `timeout=120.0, max_retries=1`（与 nodes.py 对齐）。每次 LLM 调用 2 分钟超时，不会无限等待。

---

### docling 解析大 PDF 截断（std::bad_alloc）/ 偶发 segfault(139) / 挂起

**现象**: 26 页论文入库时，docling 只在 page 1~11 成功，page 12~26 报 `Stage preprocess failed for run 1, pages [n]: std::bad_alloc`，markdown 确定性截断到 ~32k 字符（差不多正好丢参考文献区）；另观察到新起后端进程偶发 segfault(139)（onnxruntime 原生崩溃，连带整个 uvicorn 死掉）、低并发组合下偶发挂起（进程 CPU≈0，无 bad_alloc 也无产出）。

**原因**: docling threaded pipeline 默认按 4 页一批做 layout/table 推理，多页页面平面累积会顶穿本机空闲内存上限（已实测空闲 1.4GB 时必现，3.2GB 时改善但不确定）；onnxruntime CPU 推理在本机仍有原生层不稳。

**解决**（已内置 `pdf_pipeline/parser.py::_make_pipeline_options`）:
- `layout_batch_size / table_batch_size / ocr_batch_size = 1` —— 逐页处理，大幅降低峰值内存（实测 19→15 页失败、正文区完整保住）
- 仍失败时可 `DOCLING_TABLE_STRUCTURE=0` 环境变量关掉最重的 table 模型
- 启动后端进程时加 `OMP_NUM_THREADS=1` 降低 onnxruntime 线程 arena
- 已确认抛弃的页是尾页参考文献时，可接受并继续入库；待机器空闲后可重试完整解析

---

### 入库报 FileNotFoundError: final_enriched.md

**现象**: 入库（`/api/agent/ingest`）在 stage=parse 末尾失败：`[Errno 2] No such file or directory: .../pdf_pipeline/output/{paper}/final_enriched.md`。**仅含点号的文件名（如 arXiv ID `2003.12462v2.pdf`）必现**；纯下划线命名（`Attention.pdf` 等）正常。

**原因**: 目录名推导不一致——
- `build_bindings`（bindings.py:574）按 PDF stem 把非字母数字转为 `_`：`2003.12462v2` → `2003_12462v2`
- `_run_pipeline`（pdf.py）的 `output_dir` 按 API `paper_name`（保留点号）拼：`output/2003.12462v2`

解析/增强/enrich 全部写到 assets_dir（`2003_12462v2`），而 `_run_pipeline` 从 `output_dir`（`2003.12462v2`）读 `final_enriched.md` → 必然找不到。无点号文件名两目录恰好一致才没露馅。

**解决**: `_run_pipeline` 显式写入 output_dir——`raw.md` 不存在则先落盘，`enrich_markdown(..., output_path=str(output_dir/"final_enriched.md"))`，不再依赖 enricher 默认的 assets_dir。

---

## indexer

### sentence_transformers 导入 segfault（核心问题）

**现象**: `from sentence_transformers import SentenceTransformer` 导致 Segmentation fault (exit 139 / 0xC0000005)，即使先 `import torch` 也无法解决。bash terminal 环境下必现，IDE 终端环境下导入不 segfault 但可能 hang。

**原因**: `sentence_transformers` 包的编译扩展 DLL 与当前 conda 环境的 CUDA toolkit (torch 2.5.1+cu121) 存在二进制冲突。与 onnxruntime 无关（即使进程中没有 onnxruntime 也 crash）。

**解决**: 绕过 sentence_transformers，改用 `transformers.AutoModel` 直接加载模型 + 手动实现 pooling。`SentenceTransformerAdapter` 已内置此逻辑：

```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained(path, trust_remote_code=True)
# mean pooling (BERT/BGE) 或 last-token pooling (Qwen3-Embedding)
```

模型切换无缝 — 只需修改 `config.yaml` 的 `model_name`，adapter 自动检测架构并选择正确的 pooling 策略。

---

### embedding 阶段卡在模型加载

**现象**: 日志停在 `[EMBED-LOCAL] Loading model weights...` 或之后无输出。

**原因**:
1. 首次加载 0.6B 模型 + CUDA kernel 编译需要较长时间（30s-2min，取决于磁盘和 GPU）
2. 若日志停在 `Importing sentence_transformers...`，是缺 `import torch` 前置导入（见上方条目）

**解决**:
1. 确保 `config.yaml` 中 `model_name` 指向本地路径（如 `Qwen3_model/Qwen3-Embedding-0.6B`），adapter 会自动解析为绝对路径
2. 检查每个阶段的耗时输出，定位瓶颈（import / 模型加载 / FP16 转换 / warmup）
3. 首次加载需耐心等待；`sentence_transformers` 导入前必须先 `import torch`

---

### Chroma 文件锁导致 Windows 清理失败

**现象**: `PermissionError: [WinError 32]` 在删除 Chroma 持久化目录时，提示 `data_level0.bin` 被占用。

**原因**: Chroma PersistentClient 持有文件句柄，`shutil.rmtree` 无法删除。

**解决**:
```python
# 清理前释放 Chroma 引用
store._client = None
store._collection = None
shutil.rmtree(path, ignore_errors=True)
```

---

## agent / 信息注入架构错误导致 synthesize 失败

**现象**: Agent 成功通过 list_papers → get_paper_abstract → read_paper_section 完整链路获取论文内容，但 synthesize 节点返回空，最终输出通用道歉「抱歉，未能生成回答」。LangSmith trace 显示所有工具调用均返回 `ok: true`，但 synthesize LLM 未产出任何有效回答。

**原因**: 两个信息注入层面的设计错误叠加——

1. **内容工具返回 JSON 包裹格式**：`read_paper_section` / `get_paper_abstract` / `get_chunk_context` 返回 `{"ok": true, "data": {"chunks": [{"content": "...", ...}]}}`。LLM 需要先解析 JSON 结构，再遍历 chunks 数组提取 content 字段，最后拼接成可读文本。每个 chunk 的 content 被 `_chunk_summary()` 硬截断到 800 字符（学术方法论段落通常 2000-4000 字符），导致 JSON 内嵌的正文被腰斩。LLM 拿到的是一堆 JSON 碎片，既难解析又内容不全。

2. **系统指令伪装为 AIMessage**：`agent_node` 把 pre-flight 警告、error classifier 反馈、failure threshold 告警包装成 `AIMessage(content="[System note] ...")` 混入对话历史。`[System note]` 前缀靠 prompt 约束让 LLM 遵守，不如真正的 SystemMessage 或系统提示词嵌入权威。LLM 可选择忽略这些「AI 说的话」。

根因链：800 字符截断 → JSON 包裹增加认知开销 → LLM 收到碎片化数据 → synthesize 返回空 → fallback 无有效抢救逻辑 → 用户看到空白道歉。

**解决**:

1. `_chunk_summary()` 截断上限提升至 3000 字符（`web/api/routers/reader.py:534`）
2. 内容工具（`read_paper_section`、`get_paper_abstract`、`get_chunk_context`）**成功时返回纯文本**（markdown 格式），不再用 JSON 包裹正文；结构化工具（`list_papers`、`search_literature`、`lookup_page`）和所有错误保持 JSON 格式
3. 系统指令（pre-flight、error feedback、failure threshold）**嵌入 SystemMessage 而非伪装 AIMessage**
4. `synthesize_node` 新增 `_salvage_tool_content()` 抢救逻辑：LLM 返回空时扫描对话历史中的工具成功响应，拼接正文直接输出

关联修改：
- `agent/tools.py` — `read_paper_section` / `get_paper_abstract` / `get_chunk_context` 返回纯文本
- `agent/nodes.py` — `agent_node` 用 SystemMessage 注入系统指令；`_salvage_tool_content` 适配新格式
- `agent/prompts.py` — AGENT_SYSTEM 更新工具响应格式描述
- `web/api/routers/reader.py` — 截断上限 800→3000

---

## 通用

### Python subprocess 在 Windows bash 中异常

**现象**: `subprocess.run()` 或 `subprocess.Popen()` 返回 `3221225477` (0xC0000005) 或 `subprocess` 本身的 WinError 87。

**原因**: Windows Git Bash 的进程创建机制与 Python subprocess 不完全兼容。

**解决**: 在 IDE 终端或 PowerShell 中运行涉及 subprocess 的脚本。

---

### Windows Git Bash + curl 传单引号 JSON body → FastAPI "There was an error parsing the body"

**现象**: `curl -X POST ... -d '{"title":"..."}'` 返回 400 `{"detail":"There was an error parsing the body"}`；同样的命令在 Linux 下正常。

**原因**: Windows 上 curl 是原生 exe，Git Bash 单引号参数内的 `{`/`"` 处理与 POSIX shell 不一致，body 被破坏。

**解决**: 把 JSON 写入临时文件后用 `-d @file.json`（文件可含中文转义 `\uXXXX`）：
```bash
printf '{"title":"路径"}' > body.json
curl -s -X POST http://127.0.0.1:8000/api/... -H 'Content-Type: application/json' -d @body.json
```

---

### HuggingFace Hub 下载卡住

**现象**: 模型加载或 `from_pretrained()` 长时间无响应。

**解决**:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或将模型预先下载到本地，使用本地路径加载。

---

## web/frontend / npm install 网络 ECONNRESET

**现象**: `cd web/frontend && npm install` 报 `network read ECONNRESET`，`electron` / `electron-builder` / `concurrently` 均未安装，`node_modules/.bin/` 下无 electron 可执行文件。

**原因**: 默认 npm registry（registry.npmjs.org）与 electron 二进制（GitHub releases）在 China 网络下被重置/阻断。electron 的 postinstall 从 GitHub 下载 ~100MB 二进制，即使 registry 走通，二进制下载也会 ECONNRESET。

**解决**: 用 npmmirror 镜像同时覆盖 registry 与 electron 二进制下载：

```bash
cd web/frontend
ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/ \
ELECTRON_BUILDER_BINARIES_MIRROR=https://npmmirror.com/mirrors/electron-builder-binaries/ \
npm install --registry=https://registry.npmmirror.com
```

`npm run dist:win/mac` 时 electron-builder 还会下载 nsis/winCodeSign 等二进制，同样走 `ELECTRON_BUILDER_BINARIES_MIRROR`。

---

### electron 二进制解压卡住（dist/ 只剩 LICENSES.chromium.html）

**现象**: `npm install` 报 `info ok` 但 `node_modules/electron/dist/` 只有 `LICENSES.chromium.html` 一个文件、无 `electron.exe`；`node node_modules/electron/install.js` 返回 exit 0 但 dist 仍不完整、无 `path.txt`。

**原因**: electron 的 `install.js` 用 `extract-zip`(yauzl) 解压 115MB zip 时在 China 网络/Windows 下偶发卡死，只解出第一个文件就停，且不报错、进程正常退出。

**解决**: 用 Python 的 `zipfile` 手动解压缓存里的 zip + 补写 `path.txt`：

```bash
cd web/frontend
ZIP=/c/Users/30811/AppData/Local/electron/Cache/*/electron-v33.4.11-win32-x64.zip
rm -rf node_modules/electron/dist
C:/Users/30811/miniconda3/envs/demo/python.exe -c \
  "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
  "$ZIP" node_modules/electron/dist
printf 'electron.exe' > node_modules/electron/path.txt
ls node_modules/electron/dist/electron.exe   # 验证
```

（zip 完整度可用 `zipfile.ZipFile(z).testzip()` 校验，返回 None 即未损坏。）

---

## web/api / library 检索不到最新论文（all_rag_chunks.json 过期）

**现象**: `search_papers(query)` 只能检索到部分论文，最近解析/上传的论文搜不到；`eval_output/all_rag_chunks.json` 的 `paper_count` 小于 `pdf_pipeline/output/` 下的论文目录数。

**原因**: 稀疏检索（`optimal_retrieval_config.yaml` 中 `method: sparse`）的索引来自 `eval_output/all_rag_chunks.json`，它由 `indexer.pipeline.merge_all_chunks()` 聚合所有论文的 `rag_chunks.json` 生成。该合并原本只在 `/api/index/run`（索引任务）里触发，而 `/api/pdf/process` / `process-local`（解析任务）完成后不合并，导致只解析未索引的论文不出现在检索结果里。

**解决**: 已在 `web/api/routers/pdf.py::_run_pipeline` 末尾（`_register_paper` 之后）追加 `merge_all_chunks()` + `invalidate_retrieval_service()`，解析完成即同步稀疏索引。存量数据可手动补救：

```bash
C:/Users/30811/miniconda3/envs/demo/python.exe -c "from indexer.pipeline import merge_all_chunks; merge_all_chunks()"
```

---

## web/api / main.py .env 路径 off-by-one，agent 之外的 LLM 调用缺 key

**现象**: `/api/agent/notify/stream` 等不走 `agent.graph` 的 LLM 调用返回 `OpenAIError: Missing credentials`，报错指向 `OPENAI_API_KEY` 未设置；但 `agent.graph` 路径（正常聊天）却能用上 `DASHSCOPE_API_KEY`。

**原因**: `web/api/main.py` 的 `load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")` 多算了一级 —— `main.py` 在 `web/api/`，`parent×3` 才是项目根（`pre/Demo/.env`），`parent×4` 指向 `pre/.env`（不存在），于是 load_dotenv 静默空操作。`DASHSCOPE_API_KEY` 只有在 `agent.graph` 被懒加载（其内部 `load_dotenv(agent/../.env)` 路径正确）时才进入进程环境 —— 所以聊天正常、notifier 缺 key。

**解决**: `main.py` 改为 `load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")`，在 app 启动时确定性加载 `.env` 到所有端点。

---

## web/frontend / agent 内置工具连接失败（AGENT_API_BASE 端口不匹配）

**现象**: `library` subagent 调用 `search_papers` / `fetch_content` 报 connection error，提示「connectivity issues」；`list all papers` 也失败。直接 `uvicorn ... --port 8000` 运行时正常，但 Electron 桌面端复现。

**原因**: 内置工具（`agent/providers/builtin_provider.py`）通过 `AGENT_API_BASE` 访问本地 API，默认 `http://127.0.0.1:8000`。Electron 主进程（`web/frontend/src/main/backend.ts`）把 uvicorn 拉起在 `127.0.0.1:8001`，却没有给子进程注入 `AGENT_API_BASE`，导致工具一直打到 8000 端口。

**解决**: 已在 `backend.ts` spawn env 中注入 `AGENT_API_BASE: http://127.0.0.1:${port}`。若独立运行（非 Electron）且端口非 8000，需手动 `export AGENT_API_BASE=http://127.0.0.1:<port>`。

---

## agent / 库工具反复超时拖垮整轮（后端不可达无快速失败）

**现象**: agent 逐篇 `fetch_content` 验证论文时，某一篇之后整轮卡住直到返回「（回答超时，请重试或简化问题。）」。日志里多条 `tool_call ... error=timeout`，每条间隔约 10s；整轮时长逼近 300s 上限。

**原因**: 库工具（`fetch_content`/`search_papers` 等）经 httpx 访问 `AGENT_API_BASE`；后端不可达（端口错配 / 服务未起 / 丢包）时，每次调用要等满默认 10s 超时，返回的却是 `transient`（"Server busy. Retry once."）——LLM 按反馈反复重试，甚至 fallback 到 `search_papers`（同一个死端口又等 10s），把整轮 `TURN_TIMEOUT`(300s) 烧光。缺的是快速失败：始终打本机端口，connect 2s 未通即视为不可达；且连续失败后应进入熔断窗口，不再空等网络。

**解决**: 新增 `agent/library_api.py`（熔断 + 短 connect 超时），`builtin_provider` 所有库工具与 `resolution.fetch_papers` 接入：网络失败标记熔断（默认 45s，env `AGENT_API_BREAKER_TTL`），熔断期内库调用直接速断返回 `error_type="backend_down"`；`nodes._format_error_feedback` 识别 backend_down 后指示 agent 停止重试、向用户报告启动命令/端口检查。`download_paper`（外网 arXiv）不受影响。

---

## agent / 多工具任务被 token 预算提前截断（第 N 次 fetch_content 该来不来）

**现象**: 逐篇 `fetch_content` 验证多篇论文（如 13 篇里筛 NLP）时，验证到 3~5 篇左右 agent 不再继续调用工具，直接给出不完整的、像被切掉的回答；顺带可能演变成整轮接近 TURN_TIMEOUT 表现为「第 N 次调用卡住直到超时」。(用户观察到的「上下文和系统回复超出截断字数」即此现象。)

**原因**: `agent_node` 的 token 记账把每轮**全量历史重新统计并累加**进 `tokens_used`（`tokens_used += in_tokens + out_tokens`，in_tokens 每轮都重算整个 messages）——超线性膨胀。`token_budget` 默认 20000，实测 `AGENT_SYSTEM`≈2.5k、每次 fetch_content overview≈1.5~2k，3~4 轮后累计即撞线，[nodes.py] 的「Token budget exhausted → 强制 final answer（不带工具）」把正常的多工具流程在第 5 次等调用处掐断。

**解决**: `tokens_used` 改为**当次调用的实际输入规模**（`in_tokens + out_tokens`，不累加），预算语义从「累计消耗」变成「当前喂给模型的上下文上限」；默认预算 20000 → 60000（env `AGENT_TOKEN_BUDGET` 可覆盖，qwen-plus 窗口 128k，留足输出空间）。正常多轮任务不再被提前截断，超限兜底保留。

---

## agent / 执行粒度错位：step 上限 5 掐断复杂工具编排（第 N 次 fetch_content 被丢弃）

**现象**: 逐篇 `fetch_content` 验证多篇论文（如 13 篇里筛 NLP）时，第 5 次工具调用发出后「直接回复中断」——第 5 张工具卡片只有 `tool_start` 没有 `tool_end`，无最终文本。trace 停在 agent 第 5 次调用：iteration=5、消息末尾是待执行的 tool call，无工具结果。

**原因**: 当初把 `max_iterations` 当**单 turn 内 step 上限=5** 用，粒度错位：
1. `after_agent` 在 `iteration >= max_iterations(5)` 时把**已发出未执行**的 tool call 直接丢弃路由到 synthesize —— 第 5 个请求从未真正执行，SSE 只见 start 不见 end，前端卡片悬挂。
2. 复杂工具编排（逐篇验证 5~30 篇）是合法需求，5 步上限远不够；而此时嵌套的超时更糟——5 次 LLM 调用每次都把全量历史（含逐字保存在 messages 里的工具正文，一次 fetch_content ≈9KB）重发一遍，最后 synthesize 再全量喂一次，累计冲破 `AGENT_TURN_TIMEOUT=300s`，前端看到「回答超时」。
3. 会话级 turn 上限完全缺失（`iteration` 每回合 reset），只有 step 限制没有 turn 限制。

**解决**（v11，执行粒度拆分 turn/step）:
- **state.py**：`max_iterations` → `max_steps`（单 turn 内最多执行的**工具往返数**，默认 30，env `AGENT_MAX_STEPS`，兼容旧名 `AGENT_MAX_ITERATIONS`）；新增 `turn_count` / `max_turns`（会话级轮次上限，默认 50，env `AGENT_MAX_TURNS`）。turn_count 由 understand_node 每回合 +1。
- **after_agent**：语义改为「已执行轮数 = iteration - 1」；`已执行轮数 >= max_steps` 才转 synthesize，**撞上限那一步也放行执行完**，不再中途丢弃已发请求（前端不再有悬挂卡片）。
- **agent_node**：新增 turn 守卫（`turn_count > max_turns` 时强制无工具 final answer）。
- **工具结果截断**：`build_tools_node`（nodes.py，替代 prebuilt.ToolNode）把工具返回截断到 `AGENT_TOOL_RESULT_MAX`（默认 8000 字符）再进 state，多步循环每轮重发的输入不再爆炸；单工具异常转为 `{"ok": false, "error_type": "tool_crash"}` 信封供 LLM 恢复，不抛穿 graph。
- **prompts.py AGENT_SYSTEM**：新增「并行优先」强制节——独立工具请求（如逐篇 fetch_content 多个候选）必须在同一条消息内一次发出，一轮并行 ≈ 串行 5~10 轮，从源头减少步数与超时压力。
- **graph.py + web/api/routers/agent.py**：`AGENT_TURN_TIMEOUT` 默认 300 → 900。
- subagent 沿用同样的 after_agent/执行器，`build_subagent`/`SubagentSpec` 的 `max_iterations` 改名 `max_steps`（单任务窄工具面默认保持 5）。

---

## web/frontend / 端口 8001 被占用（uvicorn Errno 10048）

**现象**: `npm run dev` 时主进程拉起后端报 `[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8001)`，后端启动即退出；渲染进程轮询 `/api/health` 等全部 ECONNREFUSED。旧版前端对 `backend-status: error` 无任何处理，UI 静默离线。

**原因**: 上一次 Electron 会话退出时 uvicorn 子进程没被杀干净，残留 python.exe 仍占着 8001。`python -m uvicorn` 默认单进程内退出旧实例也不会主动释放端口。另有频繁复现诱因：`concurrently -k` 只是杀掉主进程，SIGTERM 后 5s 的 SIGKILL 兜底（`backend.ts::stop`）在进程已退出但端口未释放的窗口内重启。

**解决**: 
1. 杀掉占用者再重启：`netstat -ano | grep ':8001.*LISTEN'` → `taskkill /F /PID <pid>` → 重新 `npm run dev`。
2. 代码已加固（2026-08）：`api.ts` 的 `whenBackendReady()` gate 初始拉取，后端 `error` 状态顶部出 banner + 「重试启动后端」（IPC `backend-restart` → 复用 `PythonBackend.start()`），不再静默离线。
3. 排障时避免同时开多个 `npm run dev` 实例互相抢端口。

---

## agent / subagent 返回 "subagent produced no final answer"

**现象**: `arxiv` / `library` / `ingest` subagent 被调用后返回 `{"ok": false, "error": "subagent produced no final answer", "error_type": "unknown"}`，尽管底层工具（如 arxiv MCP）本身正常。

**原因**: subagent 子图（`agent/subagents.py::build_subagent`）只有 `agent` + `tools` 两个节点，没有 `synthesize` 节点。`after_agent` 在 max_iterations 耗尽但仍有待执行 tool call 时返回 `"synthesize"`，而子图把 `"synthesize"` 直接映射到 `END` —— 没有节点产出最终 AI 消息，`as_tool` 便回退到该错误。父图有 synthesize 安全网，子图缺失，导致 subagent 比父 agent 更脆弱（循环一多就静默失败）。

**解决**: 已在 `build_subagent` 增加 `synthesize` 节点（`_subagent_synthesize`），max_iterations 耗尽时对已累积的工具结果做一次最终 LLM 合成，LLM 失败则回退 `_salvage_tool_content` 提取原始内容，保证 subagent 总能产出答案而非报错。

---

## agent / plan 模式全断：`plan_empty_result` × 2 → 空 fallback → 「抱歉，未能生成回答」

**现象**: 对比/多论文类查询（「对比一下 RMNet 和 SRN」）稳定失败。trace 显示 `node: plan` `plan_empty_result`（attempt 0/1）×2 → `plan_fallback n_steps: 0` → `executor` 0ms no-op → `synthesize` 无上下文 → 返回「抱歉，未能生成回答」。较早期另见整轮直接崩 `'NoneType' object has no attribute 'steps'`（plan_node 曾无守卫访问 `result.steps`）。

**原因**: 三个根因叠加（逐一实测确认，非猜测）：

1. **输出契约与解析机制矛盾（真正根因）**：`plan_node` 用 `with_structured_output(method="function_calling")`，**只认 OpenAI tool call**；而 `PLAN_SYSTEM` 的输出契约是「Output ONLY a JSON object」。qwen-plus 忠实地输出 JSON **文本**（实测 `tool_calls: []`，内容为合法 `{"steps":[...]}`）→ langchain 返回 `None` → 合法 plan 被静默丢弃，「重试一次」是治标。
2. **understand 把论文名放进 `entities` 而非 `focus_papers`**（实测："compare RMNet and Attention" → `entities: ['RMNet','Attention']`, `focus_papers: []`），而 `resolve_node` 只消解 `focus_papers` → `resolved.papers` 恒空 → `_fallback_plan` 恒 0 步 → executor 空转。改 prompt 无效（qwen 不遵守该条指令），必须代码层兜底。
3. **`decide_mode` 用 entities 词袋 + 黑名单做 multi-target 判定**（`_NON_TARGET_TERMS`），是「单论文单动作指令被误判进 plan」后的特殊化补丁，事件里反复出现一边倒的碎片式修复。

**解决**（`agent/plan.py` + `agent/resolution.py` + `agent/prompts.py`）：

1. **plan 结构化输出直接解析模型 JSON 文本**：去掉 `with_structured_output(method="function_calling")`。`_ask_for_plan` 用普通 `model.ainvoke` → `_extract_json_text`（剥 ```json 围栏/前后缀）→ `_parse_steps`（pydantic 校验，非法步骤丢弃）；兼容 provider 偶尔发出的 tool call。契约 = 实现：prompt 要 JSON 文本，代码就解析 JSON 文本。
2. **resolve_node 同时消解 `entities`**：`focus_papers` 与 `entities` 一并作为论文候选去匹配库内论文，按 match 名去重、保留候选顺序；匹配不到的空丢（不污染 hints）。-- 论文名在不在 focus_papers 都能被解析到。
3. **decide_mode 只认 resolve 确证的目标**：删 `_NON_TARGET_TERMS`/`_is_plan_target` 黑名单启发式。`plan` 触发条件 = 对比关键词 / ≥2 子问题 / `resolved.papers` 中 level∈{EXACT,HIGH,MEDIUM} 的去重 match ≥2。单论文单动作指令（入库/下载）无对比词、rule 下 resolve 至多 1 篇 → 天然 react。
4. `UNDERSTAND_SYSTEM` 补 focus_papers 正反例、`PLAN_SYSTEM` 补「no code fences」——与解析器契约闭环。

保留纵深：`_ask_for_plan` 仍重试 2 次 → 空才 `_fallback_plan`（只读 library 步骤）；`understand_node` 仍重试 + 默认 literature_search。原则：**plan/understand 节点绝不 raise**。

排查标签：`plan_fallback` / `plan_empty_result` / `plan_llm_failed`（观测日志）。同类风险点：所有 `with_structured_output` 调用点都要确认模型的输出通道与所选 method 匹配（文本 JSON 走直接解析，不要依赖 tool call）。自检：`python agent/tests/test_plan.py`（`test_parse_plan_json` / resolved 化的 `test_decide_mode_*`）。

---

## agent / system prompt `.format()` 模板里未转义的字面花括号 → 任意对话崩溃 KeyError

**现象**: 用户问任意问题（如「./data/中有哪些论文」）直接失败，流式回复为「⚠️ 错误: KeyError: '路径' 已停止」。排查时 Python 源码里搜不到 `'路径'` 键——因为报错根本不在业务字典，而在**模板渲染**。

**原因**: `agents/prompts.py` 的 `AGENT_SYSTEM` 是 str 模板，`agent_node` 用 `.format(intent=…, entities=…, …)` 渲染。新增文案里写了字面量「"PDF 已保存到 {路径}"」，`{路径}` 被 str.format 当作占位符，找不到名为 `路径` 的形参 → `KeyError: '路径'`。且任何一轮对话都命中（format 在每轮 agent 节点都执行），表现为"端到端全崩"而非局部错误。

**解决**: 模板中的**字面花括号必须写双份** `{{路径}}`（渲染后变回 `{路径}`）。已修复并加断言（`assert '{路径}' in AGENT_SYSTEM.format(...)`）。自查法：凡经 `.format`（或 f-string 之外的 str.format / 显式 `%`）渲染的 prompt，新增 `{中文}`/`{word}` 字面量时先跑一次渲染自检。非 `.format` 模板（`PLAN_SYSTEM`/`ARXIV_SYSTEM` 等直接传 LLM，内含 JSON 示例花括号）不受影响。

---

## indexer/catalog / 论文状态与向量库脱节（已入库却显示未入库）

**现象**: 知识库列表/`check_paper` 空 term 快照显示大量论文 `parsed`，但其中部分实际已在 Qdrant 可检索。实测：`Mask_Approximation_Net` 有 35 chunk 在 `rag_chunks` 集合、目录 entry 却**缺失 `indexed` 字段** → `bool(None)=False` → 永远显示 parsed。

**原因**: 目录 `indexed` 标志只在显式写路径置位（旧版 `mark_indexed`；现在 `register_indexed`）。两条历史脱节路径：① 旧 schema 的 `register_paper` 不写 `indexed` 键（本仓库 3 篇遥感论文 entry 都缺键）；② 由未接 `mark_indexed` 的旧入库代码写入，进库成功但标志永远 False。「对外显示」读标志（reader.py `bool(meta.get("indexed"))`），「能否检索」看向量库与 all_rag_chunks.json —— 两者从不交叉对账，desync 属于结构性漏洞。

**解决**: ① 新增 `indexer/reconcile.py`：滚动向量库按 `chunk_id` 前缀（`{paper}__chunk_NNNN`）计数，与目录 `indexed` 标志对齐并 `patch_paper()` 回填（对正确调用：`python -m web.cli reconcile` 或 `POST /api/index/reconcile`）。② 入库路径改为原子收尾：`register_indexed()` 一次写完（注册+置真），`_run_ingest` 解析阶段 `register=False` 不写中间态，失败不残留「已解析未入库」。③ 对外状态收敛为两类（indexed/not_indexed），parsed/raw 由 `detail` 字段从文件系统派生，不作为持久终态维护。全新索引后运行一次 reconcile 即可彻底清除存量 desync。

## web/api / agent 入库假完成（dedup 短路）—「已入库」与 check_paper 矛盾

**现象**: 对一篇 `parsed`（仅解析未入库）的论文发起入库，任务**秒完成**并通知「已在库中（重复）」，agent 说「已完成入库」；但 check_paper 依旧 `not_indexed`，向量库检索不到，`indexed` 始终 False。

**原因**: 仅解析流（旧 `/api/pdf/process`）`register_paper()` 写目录时会**同时写 `dedup:hash:{sha256}→paper_name` 内容去重键**。`_run_ingest` 的 `_locate_and_dedup()` 对同一 PDF 算回同 hash → `catalog.is_duplicate()` 命中 → 旧逻辑 `if existing:` **无条件短路**成 `done`（`status:"indexed"`）。对已索引论文这是对的，对「已注册未索引」论文则是假完成：向量库什么都没写，状态前（未入库）后（已入库）互相矛盾。

**解决**: `web/api/routers/background.py::_run_ingest` 去重快判收紧为 `if existing and existing.get("indexed")` 才短路；非 indexed 命中时若 output 已有 `rag_chunks.json` 则跳过 docling 重解析、直接跑 `_run_indexing`（meta 沿用 catalog 现有元数据，收尾 `register_indexed()` 置真），否则走完整 `_run_pipeline(register=False)`。同时 `_run_pipeline` 在 `register=False`（复合入库解析阶段）不再置 `status="done"`，避免向量化前任务闪现「完成」被 `check_task_status` 误读。

---

## indexer/embedding / 本地 jina-embeddings-v5 入库报 `type object 'JinaEmbeddingsV5Model' has no attribute 'config_class'`

**现象**: 入库任务（如 CCExpert）~50s 后失败，前端显示 `type object 'JinaEmbeddingsV5Model' has no attribute 'config_class'`。模型是本地下载的 `jina/jina-embeddings-v5-text-nano`（`trust_remote_code` 加载，内置 `modeling_jina_embeddings_v5.py`，类继承 peft 的 `PeftMixedModel`）。

**原因**: 此错误特征为「旧 transformers 运行时 + 新 remote-code 模型类」的组合错位：`JinaEmbeddingsV5Model` 不在 `PreTrainedModel` 继承链上（MRO: PeftMixedModel→PushToHubMixin→Module），自身不定义 `config_class`；只有当进程中导入的 transformers 处于旧版本的 in-memory 状态（长驻后端进程在包升级前启动，磁盘是 5.12.0 但进程内存里是旧版模块缓存）时，`from_pretrained`/`__init_subclass__` 才会去访问 `cls.config_class` 并对该类抛 AttributeError。**与模型文件、代码、当前依赖版本都无关**——同一 demo 环境下适配器加载、完整 IndexerPipeline（上述失败代码路径）、带完整 web.api.main 导入栈的进程内加载+编码均验证通过（transformers 5.12.0 / peft 0.19.1）。

**解决**: 重启后端进程（end-to-end 新进程加载最新 transformers）后重试入库即可。若 Qdrant 里已插入 chunk 但 catalog 未登记（绕过 API 直接跑 pipeline 的情况），用 `C:/Users/30811/miniconda3/envs/demo/python.exe -m web.cli reconcile` 或 `POST /api/index/reconcile` 补齐。验证命令：

```bash
# 1) 适配器端到端（等价入库模型加载路径）
C:/Users/30811/miniconda3/envs/demo/python.exe -c "import sys; sys.path.insert(0,'.'); from indexer.config import load_config; from indexer.embedding_adapters import create_embedding_adapter; a=create_embedding_adapter(load_config().embedding); v=a.embed_batch(['warmup']); print('dim', len(v[0]))"
# 2) 完整 IndexerPipeline（CCExpert 可通过）
C:/Users/30811/miniconda3/envs/demo/python.exe -c "import sys; sys.path.insert(0,'.'); from indexer.pipeline import IndexerPipeline; print(IndexerPipeline('indexer/config.yaml').run('pdf_pipeline/output/CCExpert/rag_chunks.json'))"
```

排查口诀：报错类名不在 `PreTrainedModel` 继承链 （`peft import PeftMixedModel; PeftMixedModel.__mro__`） → 基本可判定是进程态/版本错位，先重启进程再查别的。

---

## LangSmith / agent 工作不被追踪（TRACING 变量失效）

**现象**: LangSmith 面板看不到当前项目的 agent 工作（graph 各节点 run / LLM 调用 / 工具调用全部缺失）。旧版 v1 agent 能正常追踪，重构后消失。

**原因**: 追踪开关判定在 `langsmith/utils.py::tracing_is_enabled()`：
`get_env_var("TRACING_V2", default=get_env_var("TRACING", default="")) == "true"`，`get_env_var` 按 `LANGSMITH_*` → `LANGCHAIN_*` 顺序取值。`.env` 里写的是 `LANGSMITH_TRACING_V2=false`，SDK 读到的字面量是 `"false"` → `== "true"` 失败，且 `LANGSMITH_TRACING` 未设置，开关恒关（`false` 不会被特殊处理，必须拿到 `true` 字面量）。此前 v1 `config.py::_init_langsmith` 显式 `os.environ["LANGCHAIN_TRACING_V2"]="true"` 覆盖了 .env，v2 重构移除该初始化后 .env 的 `false` 生效。另：2026-08 期间 DNS 解析 `api.smith.langchain.com` 失败曾被当作关闭追踪的理由，现已恢复可达（解析 34.8.121.39），该约束已过期。

**解决**: `.env` LangSmith 区块改为
```
LANGSMITH_TRACING=true
LANGSMITH_TRACING_SAMPLING_RATE=1.0
LANGSMITH_PROJECT=paper-agent
```
删除/不再依赖 `LANGSMITH_TRACING_V2`（值非 `true` 时反而抢占判定、导致 `LANGSMITH_TRACING=true` 也不生效）。注意 `.env` 必须早于任何 langchain/langsmith import 被加载（`graph.py`/`web/api/main.py` 已有 `load_dotenv` 且置于 import 链最前，保持即可）。验证：
```python
from langsmith.utils import get_env_var, tracing_is_enabled
get_env_var("TRACING_V2", default=get_env_var("TRACING", default=""))  # 期望 'true'
tracing_is_enabled()  # 期望 True
```

---

## agent / 写作链路「聊天回全文，doc 只落最后一章」+ LangSmith 缺 creation

**现象**: 「写综述/报告」请求跑完爆料：章节树里只有第三章 ✓，聊天却输出了完整三章正文；LangSmith 面板看不到 creator/creation 内部调用（前端 SSE 却能收到 `tool_start(creator)` + `doc_section` 事件）。

**原因**: 三个独立缺陷叠加（2026-09 实测，doc 证据见 `web/workspace/docs/c41dd6e66ce5`：outline `abstract(pending)/introduction(pending)/foundations(done)`，sections 下只有 `foundations.md`）：
1. creator `max_steps=5` 难以覆盖「读多篇论文 + 写一章」的负担，中途打满 → 子图 `_subagent_synthesize` 用泛化 prompt 把整章正文拼成 final answer，**doc_write_section 从未被调用**；
2. executor 对 creator 步骤**无落盘校验**——subagent 以纯文本作答也被当作成功，正文原样进入 `subagent_results`；
3. `as_tool._call` 调 `subgraph.ainvoke(init)` **不带 config**，subagent run 脱离父 trace（LangSmith 看不到）；父层 `_synthesize_plan` 又把正文合并进 context 交给 LLM 回给用户。

**解决**（agent/plan.py、agent/subagents.py、agent/domains/creation.py、agent/nodes.py、agent/config.*）：
1. creator `max_steps` 5→12（值统一放 `agent/config.yaml` 的 `subagents.creator.max_steps`，`state.py`/`build_subagents` 从 `get_limits()` 读取，env `AGENT_MAX_STEPS` 优先于文件）；executor 对 creator 步骤做确定性校验（`_verify_creator_step` → `creation.verify_section_written`，outline 该 section `status==done` 才算产出），失败判错且不转发正文，并自动补一次显式「必须 doc_write_section」重试；
2. `_creation_plan` 强制章节串行（每章 `depends_on` 链前章），消除并行写 `doc.json` 的 read-modify-write 竞争；
3. creation 域 synthesize 输出确定性「写作进度报告」（每章 status/字数 + doc_id），不再拼接 subagent 正文；
4. `as_tool._call` 声明 `config: RunnableConfig` 参数并透传给 `subgraph.ainvoke(init, config=config)`，`executor` 的 `_run_step` 也把图配置传入 `tool.ainvoke(..., config=config)`——subagent 运行作为子 run 挂到父 trace。

回归：`agent/tests/test_creation.py` 新增 `test_creator_step_fails_when_section_not_written` / `test_creation_plan_serializes_steps`；`test_subagents.py` fake 签名加 `config`。

---

## agent / 回复里出现 `[FINAL_ANSWER]` 噪声（marker 泄漏 + 持久化污染）

**现象**: 用户观察到回答前缀/正文里出现 `[FINAL_ANSWER]`（或 `【FINAL_ANSWER】`）字面量；更隐蔽的是跨回合复现——第一回合正常，后续回合的「Recent/Prior Conversation」摘要里仍混入 marker 噪声。

**原因**: 两处叠加（2026-09 实证）：
1. `_stream_llm` 的前缀缓冲只 `prefix.startswith("[FINAL_ANSWER]")` 精确匹配字面量。首 chunk 是 `"\n"` 时缓冲判定失败，下一 chunk `[FINAL_ANSWER]` 原样 emit → 前端渲染 marker；全角括号 / 大小写 / 中间位置 / `[FinalAnswer]` 变体更无一能剥离。
2. 剥离点只覆盖流式层；agent_node 返回的 AIMessage **带着 marker 回写 `state.messages`** → `checkpoints.db` → memory 摘要下轮把 marker 当正文摘要 → 反复污染后续回合上下文。

**解决**（INFO_FLOW_REVIEW P1）：根因 = prompt 里的 marker 协议被删（`prompts.py` Response Protocol 改为「回答 YES 直接写最终答案」），路由 `after_agent` 本就不依赖 marker（无 tool_calls + 文本 = 终局），行为零变化。剩余防御：
- 统一正则 `[\[【]\s*final\s*[-_ ]?\s*answer\s*[\]】]`（大小写不敏感）作为唯一兜底，`nodes._stream_llm`（前缀缓冲 → 任意位置行过滤，容忍首 chunk `\n`/全角括号/分隔符变体）与 `routers/agent._strip_marker` 共用；
- `/chat/stream` 的 token 事件在 `_ev_pump` 统一过 `_strip_marker_segment`（不做 `.strip()`，模型 token 常以空格开头）。

关联清理（同一次排查，见 `docs/INFO_FLOW_REVIEW.md`）：P3 subagent 返回提取只取「无 tool_calls 的 AI 消息」；P2 删除 `_resolved_ctx` contextvar 隐藏通道；P4 plan executor 对直接工具步骤做确定性错误恢复（transient 原参数重试 / param_error 按 `available_papers|sections` 修正）；P6 工具结果信封统一收敛到 `agent/tool_contract.py`（`ok/err/parse_tool_result/truncate_tool_result`，`_salvage_tool_content`/`_classify_tool_error`/`plan._ingest_guard` 三个解析点共用）。

回归：`agent/tests/test_info_flow.py`（P1 marker 正则/剥离/前缀判定 + P6 envelope 生成/解析/截断保解析性）。

---

## agent/supervisor / AsyncSqliteStore 事务嵌套（cannot start a transaction）

**现象**: `agent/supervisor.py::_get_store` 初始化 `AsyncSqliteStore` 后，`aput/aget/asearch` 全部抛
`OperationalError: cannot start a transaction within a transaction`（探针定位：`setup()` 正常，首次读写即炸；进程不退出还伴随
`Task was destroyed but it is pending` 告警）。

**原因**: langgraph `AsyncSqliteStore` 自身管理 `BEGIN/COMMIT`，而 `aiosqlite.connect(db)` 默认
`isolation_level` 也自动开启事务——两层事务重叠。`AsyncSqliteSaver`（checkpoints.db）没这个问题，
说明 saver 事务模型不同，不能沿用同样的 connect 写法。

**解决**: 连接用 autocommit 模式，把事务完全交给 store 自己管理：
`_store_conn = await aiosqlite.connect(str(TASK_STORE_DB), isolation_level=None)`。
同仓库两个 SQLite 后端——saver 用默认 connect、store 必须 `isolation_level=None`——别复制错了。
（另注：该环境 `AsyncSqliteStore.aclose()` 会挂起（与 `__del__` 报 `no attribute '_task'` 同源兼容问题），
清理直接关 `_store_conn` 即可。）

---

## agent/supervisor / interrupt 抛 "Called get_config outside of a runnable context"

**现象**: worker 调 `request_review` 后路由到 `gate` 节点，`interrupt()` 抛
`RuntimeError: Called get_config outside of a runnable context`；`tasks`/`Command(resume)` 链路全部失效。

**原因**: `interrupt()` 依赖 `get_config()`（读 `var_child_runnable_config` contextvar）。
**Python 3.10 + async 节点下 LangGraph 不注入该 contextvar**（langgraph.config.get_config 的
`sys.version_info < (3,11)` 守卫 + `asyncio.current_task()` 分支），与仓库 stream.py 记录的
`get_stream_writer` 限制同根（Py<3.11 async 节点 contextvar 传播缺失）。

**解决**: gate 节点内手动把节点拿到的 `config` 桥进 contextvar（绕过硬限制，与 stream.py 自建队列同思路）：

```python
from langchain_core.runnables.config import var_child_runnable_config
_tok = var_child_runnable_config.set(config if isinstance(config, dict) else {})
try:
    reply = interrupt({"question": question})
finally:
    var_child_runnable_config.reset(_tok)
```

验证：`agent/tests/_probe`（已删）与 `test_supervisor.py::test_interrupt_resume`——interrupt 后
`aget_state().tasks` 非空、`next==('gate',)`、`Command(resume=...)` 续跑产出终答。

## agent / SubAgent 重复调用工具 & 客户端树混乱（plan 模式）

**现象**: plan 模式提问「搜索一篇 agent 的论文并存入本地知识库」时，subagent（arxiv/ingest）
对相同 query / 相同入库请求反复调工具；同一叶子工具（如 `arxiv__search_papers`）在聊天区
渲染出两张卡片——一张顶层孤儿卡 + 一张挂在 subagent 边界下的嵌套卡，卡片 id 撞车导致
永久转圈 / React 同 key 冲突。

**原因**:

1. **无调用去重**: 父 react 循环与各 subagent 共用 `build_tools_node`，对每条 graph 级
   AI message 的 `tool_call` **无条件执行**，无 (name, args) 级去重；`agent/config.yaml`
   曾把所有 subagent 的 `max_steps` 都放宽到 30。于是 arxiv subagent 可拿同一 query 重搜
   约 30 轮；`_run_step_agent`（预算 `plan_step_max_steps`=10）可重复调 `ingest` 子代理，
   每次触发 `POST /api/agent/ingest` **新建一个后台入库任务**，任务堆叠。
2. **双发射（树混乱主因）**: subagent 子图与父 react 搜索循环的 agent/tools 节点**同名
   （"agent"/"tools"）**。`_msg_pump` 按 `langgraph_node=="agent"` 发 `tool_start`，把
   subagent 内部的叶子工具（工具名不在 `SUBAGENT_NAMES`）也当成父层调用发成**顶层孤儿卡**；
   同一调用又被 `ToolDispatcher` 按 scope 发一张 `parent_id=run_id` 的**嵌套卡**。一条调用
   两次完整 `tool_start`/`tool_end` → 树里出现重复、错层节点。
3. **id 碰撞 + 前端无去重**: `_run_step_agent` 用 `tc.id or tc.name` 做卡片 id（同名工具多次
   调用 → id 撞车，`patchStep` 只补第一张）；executor 确定式重试复用 `step_id` 重发
   `tool_start`；前端 `insertStep` 无按 id 去重，且父卡未到达时子步骤被静默挂到顶层。

**解决**:

1. subagent 子图节点改名为 `subagent_agent`/`subagent_tools`（`agent/subagents.py` 的
   `_build_subgraph`，含 entry/conditional edges/gate 同步），`_msg_pump` 的
   `node=="agent"/"tools"` 过滤只命中父层 react 循环；`web/api/routers/agent.py` ToolMessage
   分支补 `node != "tools"` 守卫 —— 恢复「`as_tool._call` 是 subagent 边界唯一权威发出者」。
2. `build_tools_node` 按 (name, canonical_args) 建 `tool_result_cache`（`AgentState`
   新增 `tool_result_cache` 字段）:单轮内相同调用直接复用上次**成功**结果、不重新执行副作用
   （重搜/重复入库）；错误信封不缓存、保留恢复路径。`_run_step_agent` 再加步骤级同名缓存。
3. `_run_step_agent` 卡片 id 随机兜底（`tc.id or f"{name}-{hex6}"`）消除碰撞;subagent 工具
   不再手动 emit（边界卡唯一,与 `_run_step` 对齐）。
4. config.yaml 按角色收紧 `max_steps`: arxiv 8 / ingest 6 / coder 15 / creator 30。
5. `agent_ingest` 派发前复用同 paper_name 的 running/pending 任务,不再堆叠任务行。
6. 前端 `insertStep` 按 id 幂等去重 + 乱序重挂（父卡到达时吸收顶层待定子步骤）。

验证: `pytest agent/tests` 不回归; POST `/api/agent/chat/stream` 复现原问题,断言任一叶子
工具只出现一次且带 `parent_id`; 前端 dev 模式无 React duplicate key 警告。
