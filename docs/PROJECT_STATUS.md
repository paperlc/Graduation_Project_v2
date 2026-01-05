# 项目现状与改进建议

本文件详细梳理当前各组件的实现程度、功能点，以及已知不足与改进方向，便于后续迭代。

## 整体概览
- 形态：Streamlit 前端 + MCP Server（模拟链上工具）+ SQLite 账本 + 可选 RAG/视觉防御。双通道（安全/无防御）并行，方便对照。
- 流程：前端发起对话 → Web3Agent 构造系统提示与上下文 → LLM function calling → MCP 工具执行 → 汇总回复。防御态自动加链上快照/RAG/视觉；无防御态可被攻击（内存注入、RAG 投毒）。

## 前端（app.py）
- 实现：Streamlit UI，双列展示安全/无防御回复；悬浮输入条，消息样式（用户气泡右侧，AI 文本左对齐无气泡）；侧边栏模式/防御开关、攻击面板、手动工具调用、链上快照、调试开关；SAFE/UNSAFE 两个 MCP 客户端绑定不同账本。
- 不足：UI 动效与体验基础（无流式输出、发送状态有限）；滚动/固定用 CSS hack，移动端适配不足；调试视图可再结构化；安全/无防御仍共用同一 LLM 配置（如需彻底隔离可分配不同模型/Key）。

## Web3Agent（src/agent/core.py）
- 实现：LangChain ChatOpenAI；系统提示强调“先答案后依据，链上快照视作真实主网”；工具多轮调用，`TOOL_CALL_MAX_ROUNDS` 控制；写工具支持 `idempotency_key` 防双写；防御态自动链上快照/RAG/视觉；顾问/对话模式提示差异；本地 SimpleChatMemory；本地 Chroma RAG（集合 safe/unsafe 分离，自动加载 tweets）；视觉校验占位；trace_id/span/JSON 日志。
- 不足：安全策略粗（缺少工具白名单/风险评分等）；记忆未持久化/未摘要；RAG 检索策略简单（无重排/去重/可信度过滤）；视觉校验仅占位；无流式输出，工具并发/批量未做。

## MCP 客户端（src/mcp_client/client.py）
- 实现：SSE/stdio，两种 transport；超时与重试可配；每次调用独立 session；日志含 trace_id。
- 不足：无连接池复用；错误分类粗，用户提示有限；未支持鉴权刷新/失败重连。

## MCP Server（src/simulation/server.py）
- 实现：FastMCP 注册工具列表；健康/就绪探针；多 transport 支持。
- 不足：工具注册静态；无限流/鉴权；日志与追踪配置简化；未做版本/命名空间管理。

## 账本与数据库（src/simulation/ledger.py, db.py）
- 实现：SQLite 账本，从 JSON 种子初始化；账户/代币/授权/交易/ENS/价格/声誉/流动池等；写操作快照与审计表，幂等键支持；双账本 safe/unsafe，quickstart 启动前重建。
- 不足：业务规则简化（滑点、手续费、nonce、跨链安全缺失）；并发控制粗；审计/快照未防篡改且未清理；缺少多网络分层；工具逻辑与单一服务实例仍耦合。

## 工具层（src/simulation/tools/）
- 实现：单文件单工具注册，参数校验由 FastMCP 提供；覆盖余额/授权/交易/ENS/声誉/swap/bridge/stake 等，写工具支持 `idempotency_key`；兼容旧接口（compat_*）。
- 不足：参数校验与错误提示简化；模拟逻辑粗（定价、滑点、gas 等）；缺少更多安全工具（风险扫描、细颗粒交易模拟）。

## RAG（本地/远程）
- 实现：本地 Chroma 集成；tweets.json 自动注入；`scripts/ingest_rag.py` 支持批量导入 .md/.txt + tweets，同步到 safe/unsafe 集合；远程模式 POST {query, top_k}。
- 不足：单一向量召回，无重排/去重/源可信度过滤；未做投毒检测；缺少元数据过滤与 chunk 配置；无前端上传/在线增量导入。

## 视觉（src/agent/vision.py）
- 实现：OpenAI 兼容接口占位，用于一致性校验。
- 不足：缺少真实检测逻辑、对抗样本防护、置信度/错误反馈；UI 未呈现视觉细节。

## 攻击/防御演示（src/attacks/）
- 实现：内存注入、RAG 投毒按钮；无防御侧向量库独立。
- 不足：攻击样本有限；无自动化剧本/压测；未覆盖更多场景（提示绕过、签名伪造等）。

## Telemetry（src/utils/telemetry.py）
- 实现：结构化日志（JSON 可配）、trace_id 生成/注入、简单 span（可接 OTel）。
- 不足：未对接 OTLP/Prometheus；缺少指标（工具分布、成功率、耗时）落盘；无采样策略。

## 运维/启动（scripts/quickstart.sh, .env*）
- 实现：quickstart 启两套 MCP（safe/unsafe）并启动前端，启动前重置账本；.env/.env.example 覆盖 LLM、RAG、视觉、MCP、双账本配置。
- 不足：无容器化/docker-compose；缺少进程守护/日志轮转；前端未暴露健康探针；无鉴权/限流。

## 测试（scripts/test_mcp_tools.py）
- 实现：基础工具回归脚本（用临时账本避免污染）。
- 不足：覆盖有限，未集成 CI；无安全/对抗测试；无前端端到端测试。

## 当前可用性
- 核心链路可跑通：双通道前端、MCP 工具调用、账本写入（快照/审计）、RAG 本地/远程切换、视觉占位。
- 适合演示“安全 vs 无防御”行为差异，但尚未生产级：安全策略浅、模拟简化、缺少观测/限流/鉴权/部署最佳实践。若要提升工业化程度，优先级建议：加强安全策略与观测、完善 RAG 检索与防投毒、引入流式输出与更好 UI 体验、容器化和 CI 测试体系。
