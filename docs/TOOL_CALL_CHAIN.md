# 工具调用链详解：以 transfer_eth 为例

## 完整调用流程图

```
用户在浏览器输入 → Streamlit 前端 → Agent → MCP Client → MCP Server → 工具实现 → SQLite
```

---

## 第 1 步：工具定义

### 文件：`src/simulation/tools/transfer_eth.py`

```python
from typing import Any, Dict
from mcp.server.fastmcp import FastMCP
from src.simulation.services.types import LedgerService

def register(mcp: FastMCP, service: LedgerService):
    @mcp.tool()
    async def transfer_eth(to_address: str, amount: float, sender: str | None = None):
        """发送 ETH（写入文本账本）。"""
        return await service.transfer_eth(to_address, amount, from_address=sender)
```

**作用**：
- 将 `service.transfer_eth` 方法注册为 MCP 工具
- 定义工具的参数：`to_address`, `amount`, `sender`
- 这是工具的"外壳"，只负责注册，不包含业务逻辑

---

## 第 2 步：服务接口定义

### 文件：`src/simulation/services/types.py`

```python
class LedgerService(Protocol):
    async def transfer_eth(
        self,
        to_address: str,
        amount: float,
        from_address: str | None = None,
        idempotency_key: str | None = None
    ) -> Dict[str, Any]: ...
```

**作用**：
- 定义 `LedgerService` 接口（Protocol）
- 规定所有服务方法必须实现的签名
- 类型检查工具可以确保实现符合接口

---

## 第 3 步：MCP 服务器注册

### 文件：`src/simulation/server.py`

```python
# 工具模块列表
TOOL_MODULES: List[str] = [
    "src.simulation.tools.transfer_eth",  # ← 这里包含 transfer_eth
    # ... 其他工具
]

def build_server(host: str | None = None, port: int | None = None) -> FastMCP:
    # 创建 MCP 服务器
    mcp = FastMCP("web3-ledger", host=host, port=port)

    # 创建服务实例（实际业务逻辑）
    service: LedgerService = Ledger()  # ← Ledger 实现了 LedgerService

    # 遍历所有工具模块并注册
    for module_path in TOOL_MODULES:
        module = import_module(module_path)
        if hasattr(module, "register"):
            module.register(mcp, service)  # ← 调用 transfer_eth.py 的 register 函数

    return mcp
```

**执行流程**：
1. 创建 `FastMCP` 服务器实例
2. 创建 `Ledger()` 实例作为服务层
3. 导入 `transfer_eth.py` 模块
4. 调用 `transfer_eth.register(mcp, service)`
5. 在 `mcp` 上注册 `transfer_eth` 工具，关联到 `service.transfer_eth` 方法

---

## 第 4 步：工具实现（核心业务逻辑）

### 文件：`src/simulation/ledger.py`

```python
class Ledger:
    """SQLite-backed ledger with simple simulation helpers."""

    async def transfer_eth(
        self,
        to_address: str,
        amount: float,
        from_address: str | None = None,
        idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. 参数验证
        if amount <= 0:
            raise ValueError("Amount must be positive.")

        # 2. 幂等性检查（防止重复执行）
        cached = self._idempotent_result(conn, idempotency_key)
        if cached is not None:
            return cached

        # 3. 获取发送者地址
        sender = from_address or self._default_actor(conn)

        # 4. 检查余额
        cur.execute("SELECT balance FROM accounts WHERE address = ?", (sender,))
        row = cur.fetchone()
        sender_balance = float(row["balance"]) if row else 0.0
        if sender_balance < amount:
            raise ValueError("Insufficient funds.")

        # 5. 扣除发送者余额
        cur.execute("UPDATE accounts SET balance = balance - ? WHERE address = ?", (amount, sender))

        # 6. 增加接收者余额
        cur.execute(
            "INSERT INTO accounts(address, balance) VALUES(?, ?) "
            "ON CONFLICT(address) DO UPDATE SET balance = accounts.balance + excluded.balance",
            (to_address, amount),
        )

        # 7. 记录交易
        cur.execute(
            "INSERT INTO transactions(...) VALUES (...)",
            (tx_id, sender, to_address, amount, "ETH", now, ""),
        )

        # 8. 提交事务并创建快照
        conn.commit()
        self._snapshot()

        return {"from": sender, "to": to_address, "amount": amount}
```

**作用**：
- 实现实际的转账逻辑
- 包含余额检查、扣款、入账、交易记录
- 操作 SQLite 数据库

---

## 第 5 步：MCP 客户端调用

### 文件：`src/mcp_client/client.py`

```python
class MCPToolClient:
    def call_tool(self, name: str, **kwargs) -> Any:
        """
        同步调用工具的包装方法
        """
        return asyncio.run(self.call_tool_async(name, **kwargs))

    async def call_tool_async(self, name: str, **kwargs) -> Any:
        # 1. 创建到 MCP 服务器的连接
        async with stdio_client(server_cfg) as (read_stream, write_stream):
            # 2. 创建会话
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # 3. 列出可用工具
                resp_tools = await session.list_tools()
                tools = {tool.name: tool for tool in resp_tools.tools}

                # 4. 调用工具
                resp = await session.call_tool(name, kwargs)

                # 5. 解析返回结果
                for item in resp.content:
                    payload = item.data
                    return payload
```

**作用**：
- 封装与 MCP 服务器的通信
- 处理连接、超时、重试
- 解析工具返回结果

---

## 第 6 步：Agent 使用工具

### 文件：`src/agent/core.py`

```python
class Web3Agent:
    def __init__(self, tool_caller: ToolCaller, ...):
        # 保存工具调用器
        self.tool_caller = tool_caller

        # 构建 LLM 工具 schema
        self.tools_schema = self._build_tools_schema()

    def _build_tools_schema(self) -> List[Dict[str, Any]]:
        # 定义所有可用工具给 LLM
        tool_defs = [
            {
                "name": "transfer_eth",
                "description": "Send ETH.",
                "params": {"to_address": "string", "amount": "number", "sender": "string"}
            },
            # ... 其他工具
        ]
        # 转换为 OpenAI function calling 格式
        return tools

    def chat(self, user_input: str) -> ChatResult:
        # 1. 调用 LLM
        response = self.llm.invoke(messages, tools=self.tools_schema)

        # 2. 如果 LLM 决定调用工具
        if response.tool_calls:
            for call in response.tool_calls:
                name = call["name"]  # "transfer_eth"
                args = call["args"]  # {"to_address": "bob", "amount": 10.0}

                # 3. 调用工具
                result = self._call_tool(name, **args)

    def _call_tool(self, tool_name: str, **kwargs):
        return self.tool_caller(tool_name, **kwargs)
```

**作用**：
- 将工具暴露给 LLM
- 解析 LLM 的工具调用请求
- 调用 `tool_caller` 执行工具

---

## 第 7 步：前端调用

### 文件：`app.py`

```python
def init_state():
    # 1. 创建 MCP 客户端
    client = MCPToolClient(
        server_cmd="python -m src.simulation.server"
    )

    # 2. 创建工具调用器
    tool_caller = client.call_tool

    # 3. 创建 Agent，传入工具调用器
    st.session_state.agent_safe = Web3Agent(
        tool_caller=tool_caller
    )

def render_chat():
    # 4. 用户输入
    if prompt := st.chat_input("输入消息..."):
        # 5. 调用 Agent
        agent = st.session_state.agent_safe
        result = agent.chat(prompt)

        # 6. 显示结果
        st.markdown(result.reply)
```

---

## 完整调用时序图

```
用户输入          Streamlit         Agent           LLM           MCP Client        MCP Server         Ledger
   │                 │                │               │                │                   │
   │ "给bob转10 ETH"  │                │               │                │                   │
   ├─────────────────>│                │               │                │                   │
   │                 │ agent.chat()    │               │                │                   │
   │                 ├───────────────>│               │                │                   │
   │                 │                │ llm.invoke()  │                │                   │
   │                 │                ├──────────────>│                │                   │
   │                 │                │               │                │                   │
   │                 │                │<───────────────│                │                   │
   │                 │                │ tool_calls:   │                │                   │
   │                 │                │ [transfer_eth, │                │                   │
   │                 │                │  {to:"bob",     │                │                   │
   │                 │                │   amount:10}]  │                │                   │
   │                 │                │               │                │                   │
   │                 │                │ _call_tool()  │                │                   │
   │                 │                ├───────────────┼───────────────>│                   │
   │                 │                │               │ call_tool()    │                   │
   │                 │                │               │                ├──────────────────>│
   │                 │                │               │                │ transfer_eth()    │
   │                 │                │               │                │                   │
   │                 │                │               │                │                   │ UPDATE accounts...
   │                 │                │               │                │                   │
   │                 │                │               │                │<───────────────────│
   │                 │                │               │                │ result            │
   │                 │                │<──────────────┼───────────────│                   │
   │                 │                │ result         │                │                   │
   │                 │<───────────────│               │                │                   │
   │                 │ ChatResult       │               │                │                   │
   │<─────────────────│                │               │                │                   │
   │                 │                │               │                │                   │
   │ 显示: "转账成功"  │                │               │                │                   │
```

---

## 关键数据结构

### 1. 工具 Schema (发送给 LLM)
```json
{
  "type": "function",
  "function": {
    "name": "transfer_eth",
    "description": "Send ETH.",
    "parameters": {
      "type": "object",
      "properties": {
        "to_address": {"type": "string"},
        "amount": {"type": "number"},
        "sender": {"type": "string"}
      },
      "required": ["to_address", "amount"]
    }
  }
}
```

### 2. LLM 返回的工具调用
```json
{
  "id": "call_abc123",
  "type": "function",
  "function": {
    "name": "transfer_eth",
    "arguments": "{\"to_address\": \"bob\", \"amount\": 10.0}"
  }
}
```

### 3. 工具返回结果
```json
{
  "from": "alice",
  "to": "bob",
  "amount": 10.0,
  "sender_balance": 990.0
}
```

---

## 文件关系总结

| 文件 | 层级 | 作用 |
|------|------|------|
| `src/simulation/tools/transfer_eth.py` | 工具层 | MCP 工具注册 |
| `src/simulation/services/types.py` | 接口层 | 定义服务接口 |
| `src/simulation/server.py` | 服务器层 | 创建 MCP 服务器，注册所有工具 |
| `src/simulation/ledger.py` | 业务逻辑层 | 实现实际的区块链操作 |
| `src/mcp_client/client.py` | 客户端层 | 封装 MCP 通信 |
| `src/agent/core.py` | Agent 层 | LLM 集成，工具调用决策 |
| `app.py` | 前端层 | 用户界面，调用 Agent |

---

## 如何添加新工具

1. **在 `ledger.py` 添加业务逻辑**：
   ```python
   async def my_new_tool(self, param: str) -> Dict[str, Any]:
       # 实现逻辑
       return {"result": "..."}
   ```

2. **在 `types.py` 添加接口定义**：
   ```python
   async def my_new_tool(self, param: str) -> Dict[str, Any]: ...
   ```

3. **创建工具文件 `src/simulation/tools/my_new_tool.py`**：
   ```python
   def register(mcp: FastMCP, service: LedgerService):
       @mcp.tool()
       async def my_new_tool(param: str):
           return await service.my_new_tool(param)
   ```

4. **在 `server.py` 添加到模块列表**：
   ```python
   TOOL_MODULES.append("src.simulation.tools.my_new_tool")
   ```
