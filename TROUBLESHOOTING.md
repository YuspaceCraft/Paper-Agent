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
