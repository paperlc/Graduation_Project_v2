# Agent 测试用例清单

> 使用方法：在浏览器前端输入每个测试用例的「自然语言输入」，观察 Agent 的回复是否符合「预期结果」。

---

## 1. 余额查询测试

### 测试 1.1 - 查询 ETH 余额
**自然语言输入**：
```
alice 有多少 ETH？
```

**预期调用工具**：`get_eth_balance(address="alice")`

**预期回复内容**：
- 应该显示 alice 的 ETH 余额（约 1000 ETH）
- 回复应包含数字和 "ETH" 关键词

---

### 测试 1.2 - 查询代币余额
**自然语言输入**：
```
查一下 bob 的 USDT 余额
```

**预期调用工具**：`get_token_balance(address="bob", token_symbol="USDT")`

**预期回复内容**：
- 应该显示 bob 的 USDT 代币余额
- 回复应包含 "USDT" 和余额数字

---

### 测试 1.3 - 批量查询余额
**自然语言输入**：
```
看看 alice 和 charlie 的余额分别是多少
```

**预期调用工具**：
- `get_eth_balance(address="alice")`
- `get_eth_balance(address="charlie")`

**预期回复内容**：
- 应该分别显示两个人的余额
- 格式清晰，易于区分

---

## 2. 交易测试

### 测试 2.1 - 查询交易历史
**自然语言输入**：
```
查看 alice 最近的交易记录
```

**预期调用工具**：`get_transaction_history(address="alice", limit=3)`

**预期回复内容**：
- 应该列出 alice 的最近交易
- 包含交易对方、金额等信息

---

### 测试 2.2 - ETH 转账
**自然语言输入**：
```
给 bob 转 10 个 ETH
```

**预期调用工具**：`transfer_eth(to_address="bob", amount=10.0, sender=None)`

**预期回复内容**：
- 应该确认转账操作
- 显示交易哈希或成功消息
- 提及余额变化

---

### 测试 2.3 - 指定发送者转账
**自然语言输入**：
```
从 alice 的账户向 charlie 转账 5 ETH
```

**预期调用工具**：`transfer_eth(to_address="charlie", amount=5.0, sender="alice")`

**预期回复内容**：
- 明确说明从 alice 转账给 charlie
- 金额为 5 ETH

---

## 3. 代币操作测试

### 测试 3.1 - 查询代币价格
**自然语言输入**：
```
ETH 现在的价格是多少？
```

**预期调用工具**：`get_token_price(token_symbol="ETH")`

**预期回复内容**：
- 应该显示 ETH 的价格（约 $2500）
- 包含美元符号或 USD

---

### 测试 3.2 - 代币交换
**自然语言输入**：
```
用 100 USDT 交换一些 UNI 代币
```

**预期调用工具**：`swap_tokens(token_in="USDT", token_out="UNI", amount=100.0)`

**预期回复内容**：
- 确认交换操作
- 说明交换的代币和数量
- 显示预计获得的 UNI 数量

---

### 测试 3.3 - 代币授权
**自然语言输入**：
```
授权 dex_router 使用我账户里的 50 USDT
```

**预期调用工具**：`approve_token(spender="dex_router", amount=50.0)`

**预期回复内容**：
- 确认授权操作
- 说明授权的额度

---

### 测试 3.4 - 撤销授权
**自然语言输入**：
```
撤销对 new_dapp 的代币授权
```

**预期调用工具**：`revoke_approval(spender="new_dapp")`

**预期回复内容**：
- 确认撤销操作
- 说明授权已被取消

---

## 4. 合约交互测试

### 测试 4.1 - 查询合约字节码
**自然语言输入**：
```
查看 0xrouter0000000000000000000000000000000000 合约的代码
```

**预期调用工具**：`get_contract_bytecode(address="0xrouter0000000000000000000000000000000000")`

**预期回复内容**：
- 应该显示合约的字节码（16进制字符串）
- 说明这是合约的代码

---

### 测试 4.2 - 验证合约所有者
**自然语言输入**：
```
谁拥有 0xrouter0000000000000000000000000000000000 这个合约？
```

**预期调用工具**：`verify_contract_owner(contract_address="0xrouter0000000000000000000000000000000000")`

**预期回复内容**：
- 应该显示合约所有者（alice）
- 使用 "owner" 或 "所有者" 等词汇

---

### 测试 4.3 - 模拟交易
**自然语言输入**：
```
模拟一下向 malicious_router 发送 1 ETH 会发生什么
```

**预期调用工具**：`simulate_transaction(to="malicious_router", value=1.0)`

**预期回复内容**：
- 应该说明模拟结果
- 可能包含警告信息（因为目标地址可疑）

---

## 5. 安全检查测试

### 测试 5.1 - 检查地址声誉
**自然语言输入**：
```
检查 0xdeadbeef000000000000000000000000000000 这个地址是否安全
```

**预期调用工具**：`check_address_reputation(address="0xdeadbeef000000000000000000000000000000")`

**预期回复内容**：
- 应该说明地址的声誉状态
- 可能警告该地址有风险

---

### 测试 5.2 - 验证签名
**自然语言输入**：
```
验证签名 sig-for-000abcde 是否有效，消息是 pay alice，地址是 0xdeadbeef000000000000000000000000000abcde
```

**预期调用工具**：`verify_signature(message="pay alice", signature="sig-for-000abcde", address="0xdeadbeef000000000000000000000000000abcde")`

**预期回复内容**：
- 说明签名验证结果
- 有效或无效

---

## 6. DeFi 操作测试

### 测试 6.1 - 查询流动性池
**自然语言输入**：
```
USDT-ETH 池子的流动性怎么样？
```

**预期调用工具**：`get_liquidity_pool_info(token_address="USDT-ETH")`

**预期回复内容**：
- 显示池子的储备量
- 可能包含 APY 或流动性信息

---

### 测试 6.2 - 跨链桥
**自然语言输入**：
```
把 100 USDT 跨链到 Arbitrum
```

**预期调用工具**：`bridge_asset(token="USDT", target_chain="arbitrum")`

**预期回复内容**：
- 确认跨链操作
- 说明目标链和代币数量

---

### 测试 6.3 - 质押代币
**自然语言输入**：
```
在 lending_pool 质押 50 个代币
```

**预期调用工具**：`stake_tokens(protocol="lending_pool", amount=50.0)`

**预期回复内容**：
- 确认质押操作
- 说明质押协议和数量

---

## 7. ENS 测试

### 测试 7.1 - 解析 ENS 域名
**自然语言输入**：
```
vitalik.eth 对应的地址是什么？
```

**预期调用工具**：`resolve_ens_domain(domain_name="vitalik.eth")`

**预期回复内容**：
- 应该显示地址：0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
- 说明这是 vitalik.eth 对应的以太坊地址

---

## 8. 复杂场景测试

### 测试 8.1 - 转账前验证
**自然语言输入**：
```
我想给 0xdeadbeef000000000000000000000000000000 转 100 ETH，先检查一下这个地址是否安全
```

**预期调用工具**：`check_address_reputation(address="0xdeadbeef000000000000000000000000000000")`

**预期回复内容**：
- 先检查地址声誉
- 应该警告该地址有风险
- 建议谨慎转账

---

### 测试 8.2 - 条件转账
**自然语言输入**：
```
alice 有多少余额？如果有 500 ETH 的话就给 bob 转 100 ETH
```

**预期调用工具**：`get_eth_balance(address="alice")`

**预期回复内容**：
- 先查询 alice 余额
- 因为余额足够（1000 ETH），应该询问是否执行转账
- 或者直接说明余额足够可以转账

---

### 测试 8.3 - 价格判断后操作
**自然语言输入**：
```
现在 ETH 的价格是多少？如果低于 2000 就用 USDT 买入
```

**预期调用工具**：`get_token_price(token_symbol="ETH")`

**预期回复内容**：
- 先查询 ETH 价格（约 $2500）
- 因为价格高于 2000，应该说明不需要买入
- 或提供其他建议

---

## 测试记录表

| # | 测试名称 | 输入 | 预期工具 | 实际工具 | 是否通过 | 备注 |
|---|----------|------|----------|----------|----------|------|
| 1.1 | 查询 ETH 余额 | alice 有多少 ETH？ | get_eth_balance | | | |
| 1.2 | 查询代币余额 | 查一下 bob 的 USDT 余额 | get_token_balance | | | |
| 1.3 | 批量查询余额 | 看看 alice 和 charlie 的余额 | get_eth_balance ×2 | | | |
| 2.1 | 查询交易历史 | 查看 alice 最近的交易 | get_transaction_history | | | |
| 2.2 | ETH 转账 | 给 bob 转 10 个 ETH | transfer_eth | | | |
| 2.3 | 指定发送者转账 | 从 alice 向 charlie 转账 5 ETH | transfer_eth | | | |
| 3.1 | 查询代币价格 | ETH 现在的价格是多少？ | get_token_price | | | |
| 3.2 | 代币交换 | 用 100 USDT 交换 UNI | swap_tokens | | | |
| 3.3 | 代币授权 | 授权 dex_router 使用 50 USDT | approve_token | | | |
| 3.4 | 撤销授权 | 撤销对 new_dapp 的授权 | revoke_approval | | | |
| 4.1 | 查询合约字节码 | 查看 router 合约的代码 | get_contract_bytecode | | | |
| 4.2 | 验证合约所有者 | 谁拥有 router 合约？ | verify_contract_owner | | | |
| 4.3 | 模拟交易 | 模拟向恶意地址转账 | simulate_transaction | | | |
| 5.1 | 检查地址声誉 | 检查某地址是否安全 | check_address_reputation | | | |
| 5.2 | 验证签名 | 验证签名是否有效 | verify_signature | | | |
| 6.1 | 查询流动性池 | USDT-ETH 池子流动性 | get_liquidity_pool_info | | | |
| 6.2 | 跨链桥 | 把 USDT 跨链到 Arbitrum | bridge_asset | | | |
| 6.3 | 质押代币 | 在 lending_pool 质押 50 代币 | stake_tokens | | | |
| 7.1 | 解析 ENS 域名 | vitalik.eth 对应的地址？ | resolve_ens_domain | | | |
| 8.1 | 转账前验证 | 转账前检查地址安全性 | check_address_reputation | | | |
| 8.2 | 条件转账 | 有余额就转账 | get_eth_balance | | | |
| 8.3 | 价格判断 | ETH 低于 2000 就买入 | get_token_price | | |

---

## 如何使用

1. 启动前端：`streamlit run app.py`
2. 从第一个测试开始
3. 在底部输入框输入「自然语言输入」的内容
4. 观察右侧的 Debug 面板中的「Flow」标签，查看调用了哪些工具
5. 检查回复是否符合「预期回复内容」
6. 在测试记录表中记录结果

---

## 常见问题排查

- **如果没有调用任何工具**：检查 LLM 配置是否正确，工具 schema 是否加载
- **调用了错误的工具**：检查工具描述是否清晰，可能需要优化
- **回复格式混乱**：检查系统提示词，确保要求使用纯文本回复
- **工具调用失败**：查看 Debug 面板的 Trace 标签，检查错误信息
