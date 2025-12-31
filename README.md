# Web3 智能体攻防演示（ChatGPT 风格）

前端基于 Streamlit，结合 LLM（LangChain + OpenAI 兼容接口）、可插拔 RAG（本地 Chroma / 远程服务）、视觉校验，以及 MCP 工具（模拟链上交互）构建的攻防演示台。

## 功能亮点
- ChatGPT 风格深色对话界面，支持图片上传触发视觉一致性校验。
- 双模式：对话模式（更贴近闲聊/问答）、顾问模式（更偏投资建议/风控提示）；可在侧边栏切换。
- 防御链路：链上快照（MCP 工具）、RAG 检索、视觉核验，全部可开/关；关闭后仍可正常对话和工具调用。
- 攻击演示：一键内存注入、RAG 投毒，观察智能体的防御响应。
- 双答案视图：同一问题同时展示“无防御/被攻击”与“防御开启”两种回复，方便对比。
- 文本账本：`data/ledger/ledger.json` 保存账户、代币、授权、交易、ENS、价格、声誉、流动池等，不在 Python 中硬编码，可换不同场景的账本文件。
- 工具解耦：每个 MCP 工具位于 `src/simulation/tools/` 独立文件，方便增删；`src/simulation/server.py` 动态注册。

## 快速开始
1) 安装依赖  
```bash
pip install -r requirements.txt
```
2) 配置环境变量（已内置 `.env` 示例，可直接修改）  
- 关键项：`OPENAI_API_KEY` 或 `LLM_API_KEY`；如使用本地/自建 OpenAI 兼容接口，设置 `LLM_API_BASE`。  
- 模块可选：`RAG_PROVIDER=off` 可关闭 RAG；`VISION_ENABLED=false` 可关闭视觉；`DEFENSE_DEFAULT_ON=false` 可默认关闭防御。  
- 其他见下方「.env 配置说明」。
3) 运行前端（默认 http://localhost:8501）  
```bash
streamlit run app.py
```

## .env 配置说明
`.env`/`.env.example` 中包含常用项，`python-dotenv` 会自动加载：
- 大模型 / Embedding（OpenAI 兼容）
  - `LLM_API_KEY`（可兼容 OPENAI_API_KEY）、`LLM_API_BASE`、`LLM_MODEL`
  - `EMBEDDING_API_KEY`、`EMBEDDING_API_BASE`、`EMBEDDING_MODEL`
- RAG
  - `RAG_PROVIDER=local|remote|off`（off 时完全跳过 RAG）  
  - 本地：默认 Chroma；可指定 `CHROMA_PATH` 持久化；无 Embedding Key 时自动跳过。
  - 远程：`RAG_REMOTE_URL`（POST {query, top_k}，返回 documents/results）、`RAG_REMOTE_API_KEY`
- 视觉
  - `VISION_ENABLED=true|false`；`VISION_API_KEY`（可兼容 LLM/OPENAI）、`VISION_API_BASE`、`VISION_MODEL`（OpenAI 兼容接口即可接入本地或厂商视觉模型）
- 防御
  - `DEFENSE_DEFAULT_ON=true|false` 控制默认是否启用防御（UI 可再次切换）

## RAG / 视觉兼容策略
- RAG：`RAG_PROVIDER=local` 时使用本地 Chroma；设为 `remote` 时请求 `RAG_REMOTE_URL`（预期返回 JSON 中的 `documents` 或 `results.text`）。  
- 视觉：走 OpenAI 兼容协议，可将 `VISION_API_BASE` 指向本地服务或厂商网关；无键或错误将返回“未通过”以避免盲信图片。

## 链上工具清单（MCP，文本账本驱动）
- 目录：`src/simulation/tools/`（单文件单工具，可自由增删）
- **基础查询**：`get_eth_balance`、`get_token_balance`、`get_transaction_history`、`get_contract_bytecode`、`resolve_ens_domain`、`get_token_price`
- **安全防御**：`check_address_reputation`、`simulate_transaction`、`verify_contract_owner`、`check_token_approval`、`verify_signature`
- **资产操作**：`transfer_eth`、`swap_tokens`、`approve_token`、`revoke_approval`
- **高级/DeFi**：`get_liquidity_pool_info`、`bridge_asset`、`stake_tokens`
- 兼容旧名：`get_balance`、`transfer`

## 账本说明
- 文件：`data/ledger/ledger.json`（账户、代币余额、授权、交易、ENS、价格、声誉、流动池等，已扩充多用户与多代币）。
- 修改即可自定义初始状态；默认操作人 `meta.default_actor` = `treasury`；可通过环境变量 `LEDGER_FILE` 指定其他账本。

## 演示路径
- 正常对话：直接问答，上传截图可触发视觉校验；防御模式可在侧边栏开关。
- 模拟攻击：点击「内存注入攻击」或「RAG 投毒」，观察回复中的链上快照 / 检索情报变化。
- 链上操作：侧边栏填写转出方/接收方/金额点击「执行转账」，结果写回 `ledger.json` 并实时展示快照。

## 发布/部署
- 本地或服务器直接运行 `streamlit run app.py`，无需额外后端。
- 容器化：以 `python:3.10` 为基底安装依赖，复制项目，暴露 8501 后运行同样命令。
