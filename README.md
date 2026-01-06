# Web3 智能体攻防演示（ChatGPT 风格）

前端基于 Streamlit，结合 LLM（LangChain + OpenAI 兼容接口）、可插拔 RAG（本地 Chroma / 远程服务）、视觉校验，以及 MCP 工具（模拟链上交互）构建的攻防演示台。

## 功能亮点
- ChatGPT 风格深色对话界面，支持图片上传触发视觉一致性校验。
- 双模式：对话模式（更贴近闲聊/问答）、顾问模式（更偏投资建议/风控提示）；可在侧边栏切换。
- 防御链路：链上快照（MCP 工具）、RAG 检索、视觉核验，全部可开/关；关闭后仍可正常对话和工具调用。
- 攻击演示：一键内存注入、RAG 投毒，观察智能体的防御响应。
- 双答案视图：同一问题同时展示“无防御/被攻击”与“防御开启”两种回复，方便对比。
- MCP 工具：通过 MCP Server 提供链上模拟工具，LLM 通过 function calling 自动选择工具（需先启动 MCP）。
- 文本账本：`data/ledger/ledger.json` 保存账户、代币、授权、交易、ENS、价格、声誉、流动池等，不在 Python 中硬编码，可换不同场景的账本文件。
- 工具解耦：每个 MCP 工具位于 `src/simulation/tools/` 独立文件，方便增删；`src/simulation/server.py` 动态注册。

## 实现现状与已知限制
- 前端：无流式输出，移动端适配和交互动效基础；调试视图为文本列表，未结构化分区。
- Agent/RAG：本地 Chroma 检索，未做重排/去重/可信度过滤；投毒检测仅依赖安全/不安全分库，无自动风险标注。
- 视觉：本地 Florence Caption + 远程文本判定已通路；远程多模态默认关闭且缺少频控/回退策略。
- 运维：无容器化/CI/健康探针；日志和指标仅基础打印，未对接监控。

## 快速开始
1) 安装依赖  
```bash
pip install -r requirements.txt
```
2) 配置环境变量（已内置 `.env` 示例，可直接修改）  
- 关键项：`OPENAI_API_KEY` 或 `LLM_API_KEY`；如使用本地/自建 OpenAI 兼容接口，设置 `LLM_API_BASE`。  
- 模块可选：`RAG_PROVIDER=off` 可关闭 RAG；`VISION_ENABLED=false` 可关闭视觉；`DEFENSE_DEFAULT_ON=false` 可默认关闭防御。  
- MCP（默认推荐常驻）：先启动 MCP 服务（见下方“MCP 服务模式”），前端/.env 设 `MCP_SERVER_URL=http://127.0.0.1:8001/sse`（可配 `MCP_SERVER_HEADERS` 传鉴权头）。如需回退本地进程模式再设 `MCP_SERVER_CMD`。  
- 其他见下方「.env 配置说明」。  
3) 运行前端（默认 http://localhost:8501）  
```bash
streamlit run app.py
```
或一键启动（常驻 MCP + 前端）：`bash scripts/quickstart.sh`

## .env 配置说明
`.env`/`.env.example` 中包含常用项，`python-dotenv` 会自动加载：
- 大模型 / Embedding（OpenAI 兼容）
  - `LLM_API_KEY`（可兼容 OPENAI_API_KEY）、`LLM_API_BASE`、`LLM_MODEL`
  - `EMBEDDING_API_KEY`、`EMBEDDING_API_BASE`、`EMBEDDING_MODEL`
  - `EMBEDDING_LOCAL_MODEL`（可选，单个本地模型名/路径，优先于远程接口，默认 all-MiniLM-L6-v2）
  - `EMBEDDING_LOCAL_MODELS`（可选，逗号分隔的模型候选列表，按顺序尝试加载）
  - `EMBEDDING_USE_LOCAL`（默认 true），`EMBEDDING_USE_REMOTE`（默认 false）
- RAG
  - `RAG_PROVIDER=local|remote|off`（off 时完全跳过 RAG）  
  - 本地：默认 Chroma；可指定 `CHROMA_PATH` 持久化；默认启用 `EMBEDDING_USE_LOCAL`（sentence-transformers，本地模型优先，无需 API Key）；若开启 `EMBEDDING_USE_REMOTE` 则回退使用 OpenAI 兼容接口（需 Embedding Key）。
  - 远程：`RAG_REMOTE_URL`（POST {query, top_k}，返回 documents/results）、`RAG_REMOTE_API_KEY`
  - `RAG_TWEET_FILE`（可选，默认 data/tweets.json，作为舆情/推特语料注入本地 RAG，可用于投毒/权威文档模拟）
  - `RAG_AUTO_INGEST`（默认 true，quickstart 启动时自动执行 `scripts/ingest_rag.py`）
  - `RAG_RESET_COLLECTIONS`（默认 false，若设为 true 则 ingest 前清空集合）
  - `RAG_RESET_STORAGE`（默认 false，设为 true 时 quickstart 会清空 `CHROMA_PATH` 目录后再 ingest）
- 视觉
  - `VISION_ENABLED=true|false` 控制是否启用视觉检测。
  - 本地 Caption/VLM：`VISION_LOCAL_CAPTION_ENABLED`（默认 true），`VISION_LOCAL_CAPTION_MODEL`（默认 Florence-2-base，本地路径或 HF 模型）。生成图片描述。
  - 远程文本判定：`VISION_REMOTE_TEXT_API_KEY`、`VISION_REMOTE_TEXT_API_BASE`、`VISION_REMOTE_TEXT_MODEL`（默认 gpt-4o-mini）。将“用户文本+本地描述”发给远程 LLM 判定一致/不一致。
  - 远程多模态判定：`VISION_REMOTE_MM_ENABLED`（默认 false）、`VISION_REMOTE_MM_API_KEY`、`VISION_REMOTE_MM_API_BASE`、`VISION_REMOTE_MM_MODEL`。直接将图+文发给远程多模态模型判定（不依赖本地 Caption）。
- 防御
  - `DEFENSE_DEFAULT_ON=true|false` 控制默认是否启用防御（UI 可再次切换）
- 调试/多轮工具
  - `LOG_LEVEL`（默认 WARNING，需更详细日志可设 INFO）
  - `TOOL_CALL_MAX_ROUNDS`（默认 3，控制多轮 tool_call 的上限，防止长循环）
  - `TOOL_CALL_TIMEOUT`（默认 15 秒），`TOOL_CALL_RETRIES`（默认 1 次）
- 账本
  - `LEDGER_FILE`（初始化种子 JSON，可指定其他账本种子）
  - `LEDGER_DB`（SQLite 账本文件路径，默认 data/ledger/ledger.db）
  - `LEDGER_SNAPSHOT_RETENTION`（默认 5，每次写快照后仅保留最新 N 份）
- MCP
  - `MCP_SERVER_URL`（默认推荐：常驻 MCP 服务 URL，SSE/HTTP 兼容），`MCP_SERVER_HEADERS`（JSON 头部，可放鉴权）  
  - `MCP_SERVER_CMD`（可选回退：如 `python -m src.simulation.server`，每次调用临时拉起 MCP 进程）
  - `MCP_TRANSPORT`（server 侧，默认 stdio，可设 sse/http/streamable-http）、`MCP_HOST`、`MCP_PORT`、`MCP_SSE_PATH`
- RAG 集合（可选覆盖默认集合名）
  - `RAG_COLLECTION_SAFE`（默认 web3-rag-safe）
  - `RAG_COLLECTION_UNSAFE`（默认 web3-rag-unsafe）

## RAG / 视觉兼容策略
- RAG：`RAG_PROVIDER=local` 时使用本地 Chroma；设为 `remote` 时请求 `RAG_REMOTE_URL`（预期返回 JSON 中的 `documents` 或 `results.text`）。  
- 视觉：两条路径均走 OpenAI 兼容协议，可将 `VISION_REMOTE_TEXT_API_BASE` 或 `VISION_REMOTE_MM_API_BASE` 指向本地服务或厂商网关；未配置对应 Key/Base 时跳过该路径。

## 链上工具清单（MCP，文本账本驱动）
- 目录：`src/simulation/tools/`（单文件单工具，可自由增删）
- **基础查询**：`get_eth_balance`、`get_token_balance`、`get_transaction_history`、`get_contract_bytecode`、`resolve_ens_domain`、`get_token_price`
- **安全防御**：`check_address_reputation`、`simulate_transaction`、`verify_contract_owner`、`check_token_approval`、`verify_signature`
- **资产操作**：`transfer_eth`、`swap_tokens`、`approve_token`、`revoke_approval`
- **高级/DeFi**：`get_liquidity_pool_info`、`bridge_asset`、`stake_tokens`
- 兼容旧名：`get_balance`、`transfer`

## 账本说明
- 底层：SQLite 账本（默认 `data/ledger/ledger.db`），首次启动会从 `data/ledger/ledger.json` 迁移初始数据（可用 `LEDGER_FILE` 指向其他种子；`LEDGER_DB` 指定 DB 路径）。
- 内容：账户、代币余额、授权、交易、ENS、价格、声誉、流动池等，已扩充多用户与多代币。
- 写操作：支持可选幂等键（idempotency_key）；每次写前自动生成快照到 `data/ledger/snapshots/` 并记录审计表。
- 默认操作人 `DEFAULT_ACTOR`（缺省 treasury）。
- 舆情/投毒：本地 RAG 会自动加载 `data/tweets.json`（或 `RAG_TWEET_FILE` 指定的文件）作为舆情语料，可用于投毒或权威文档模拟。

## 演示路径
- 正常对话：直接问答，上传截图可触发视觉校验；防御模式可在侧边栏开关。
- 模拟攻击：点击「内存注入攻击」或「RAG 投毒」，观察回复中的链上快照 / 检索情报变化。
- 链上操作：侧边栏填写转出方/接收方/金额点击「执行转账」，结果写回 `ledger.json` 并实时展示快照。

## RAG 语料导入
准备你的文档到 `data/rag/`（支持 .md/.txt），可选 tweets 语料 `data/tweets.json`。为了区分攻击/干净语料，推荐放在：
- `data/rag/clean/` 正常文档
- `data/rag/poison/` 投毒文档

默认 ingest 会合并以上目录（含 `data/rag/` 根目录）一起写入集合。运行：

```bash
python scripts/ingest_rag.py --src data/rag --src-clean data/rag/clean --src-poison data/rag/poison --tweets data/tweets.json
```

需要设置的环境变量：
- `EMBEDDING_API_KEY`（或复用 `LLM_API_KEY`）
- `EMBEDDING_API_BASE`（默认同 LLM_API_BASE）
- `EMBEDDING_MODEL`（默认 text-embedding-3-small）
- `CHROMA_PATH`（设置后持久化向量库）
- `RAG_COLLECTION_SAFE` / `RAG_COLLECTION_UNSAFE`（可选，默认 web3-rag-safe/unsafe）

导入完成后，开启防御模式（或显式开启 RAG），即可在对话中自动检索命中文档。

## 视觉功能使用
- 开关：`.env` 设置 `VISION_ENABLED=true`；false 时完全跳过视觉一致性检测。
- 本地 Caption/VLM（默认 Florence-2-base）：`VISION_LOCAL_CAPTION_ENABLED` 控制是否生成图片描述；`VISION_LOCAL_CAPTION_MODEL` 可填 HF 路径或本地目录。
- 远程文本判定：`VISION_REMOTE_TEXT_API_KEY`、`VISION_REMOTE_TEXT_API_BASE`、`VISION_REMOTE_TEXT_MODEL`，用于将「用户文本 + 本地描述」发送到远程 LLM 判定一致/不一致（纯文本模型即可）。
- 远程多模态判定：`VISION_REMOTE_MM_ENABLED`，`VISION_REMOTE_MM_API_KEY`、`VISION_REMOTE_MM_API_BASE`、`VISION_REMOTE_MM_MODEL`，直接把图+文送入远程多模态模型判定（不依赖本地 Caption）。
- 使用方法：前端开启防御，上传图片并输入描述。防御列回复会附带 `[Vision]` 提示：`✅`=一致，`⚠️`=不一致，`❌`=调用失败/出错；勾选 “Show debug messages” 可查看完整 trace。
- 不上传图片时，视觉模块不会影响纯文本功能；视觉只在防御列执行。
- 若判定为不一致（⚠️），会直接拦截回答并提示修改图片/描述，不再返回链上/RAG 结果。
- 若本地模型不存在会尝试联网拉取，建议提前用 `huggingface-cli` 下载到本地并在 `.env` 填本地路径。

## MCP 服务模式
- 常驻 SSE/HTTP（默认推荐）：先启动 MCP 服务  
  ```bash
  MCP_TRANSPORT=sse MCP_HOST=0.0.0.0 MCP_PORT=8001 MCP_SSE_PATH=/sse python -m src.simulation.server
  ```  
  前端环境设置 `MCP_SERVER_URL=http://127.0.0.1:8001/sse`（如需鉴权头，设 `MCP_SERVER_HEADERS='{"Authorization":"Bearer xxx"}'`）。此模式连接复用、无需频繁拉起子进程。
- 本地 stdio（回退/快速体验）：设置 `MCP_SERVER_CMD="python -m src.simulation.server"`，前端每次调用会临时拉起 MCP 进程。
- 一键：`bash scripts/quickstart.sh`（如有 `.env` 会自动加载，将 MCP SSE 常驻并启动 Streamlit，Ctrl+C 结束时顺便退出 MCP）。
 - 健康/就绪探针：启动后可访问 `http://<HEALTH_HOST>:<HEALTH_PORT>/healthz`（存活）和 `/readyz`（就绪），默认 0.0.0.0:8081。

## 工具一键回归
运行 `python scripts/test_mcp_tools.py`（会复制临时账本，避免污染原始数据），可快速回归所有 MCP 工具。

## 调试 / 决策链路可视化
- 侧边栏可勾选 “Show debug messages” 查看发送给 LLM 的原始消息。
- 每条回复下的“决策过程”折叠面板会展示工具调用 trace 和完整 LLM 调用链（每轮输入/输出、tool_calls）。多轮工具调用上限由 `TOOL_CALL_MAX_ROUNDS` 控制。
- UI 体验：消息输入区支持草稿保存（表单文本框）、生成时有 spinner 状态提示；工具调用支持超时/重试（见 env 配置）。

## 发布/部署
- 本地或服务器直接运行 `streamlit run app.py`，无需额外后端。
- 容器化：以 `python:3.10` 为基底安装依赖，复制项目，暴露 8501 后运行同样命令。
