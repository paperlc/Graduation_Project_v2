"""
End-to-End Agent Tests with Natural Language Input

This script tests the Web3Agent by simulating natural language inputs
and verifying that the Agent correctly calls the appropriate blockchain tools.

Usage:
    python scripts/test_agent_e2e.py [--verbose] [--suite SUITE]
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.agent.core import Web3Agent
from src.mcp_client.client import MCPToolClient
from src.simulation.db import LedgerDB

DEFAULT_LEDGER = ROOT / "data" / "ledger" / "ledger.json"


@dataclass
class TestCase:
    """单个测试用例"""
    name: str
    user_input: str
    expected_tools: List[str]  # 期望调用的工具列表
    expected_keywords: List[str] = field(default_factory=list)  # 期望在回复中出现的关键词
    check_result: bool = False  # 是否检查返回值的合理性


@dataclass
class TestResult:
    """测试结果"""
    case: TestCase
    passed: bool
    tools_called: List[str] = field(default_factory=list)
    reply: str = ""
    error: str = ""
    trace: List[str] = field(default_factory=list)


class AgentE2ETester:
    """Agent 端到端测试器"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.tool_calls_log: List[str] = []

    def create_mock_tool_caller(self) -> callable:
        """创建一个记录工具调用的 mock caller"""
        calls_made = []

        def mock_caller(tool_name: str, **kwargs):
            # 记录调用
            call_str = f"{tool_name}({json.dumps(kwargs, ensure_ascii=False)})"
            calls_made.append(call_str)
            self.tool_calls_log.append(call_str)

            if self.verbose:
                print(f"  [TOOL CALL] {call_str}")

            # 实际调用真实工具
            return self.real_tool_caller(tool_name, **kwargs)

        # 保存真实的 caller 引用
        self.real_tool_caller = None
        return mock_caller

    def setup_agent(self) -> Web3Agent:
        """设置测试用的 Agent"""
        # 创建 MCP 客户端
        client = MCPToolClient(
            server_cmd=os.getenv("MCP_SERVER_CMD", "python -m src.simulation.server"),
            server_url=os.getenv("MCP_SERVER_URL")
        )

        # 创建 mock caller 来记录工具调用
        def tracking_caller(tool_name: str, **kwargs):
            call_str = f"{tool_name}({', '.join(f'{k}={v}' for k, v in kwargs.items())})"
            self.tool_calls_log.append(call_str)
            if self.verbose:
                print(f"  [TOOL] {call_str}")
            return client.call_tool(tool_name, **kwargs)

        agent = Web3Agent(
            tool_caller=tracking_caller,
            collection_name="test-rag",
        )
        agent.set_mode("chat")
        return agent

    def run_test(self, case: TestCase) -> TestResult:
        """运行单个测试用例"""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"测试: {case.name}")
            print(f"输入: {case.user_input}")
            print(f"期望工具: {', '.join(case.expected_tools)}")

        # 清空工具调用日志
        self.tool_calls_log.clear()

        # 运行 Agent
        agent = self.setup_agent()
        result = TestResult(case=case, tools_called=[])

        try:
            response = agent.chat(case.user_input)
            result.reply = response.reply
            result.trace = response.trace
            result.tools_called = self.tool_calls_log.copy()

            if self.verbose:
                print(f"\n回复: {response.reply[:200]}...")
                print(f"工具调用: {result.tools_called}")

            # 验证工具调用
            tools_found = []
            for expected_tool in case.expected_tools:
                if any(expected_tool in call for call in result.tools_called):
                    tools_found.append(expected_tool)

            # 检查是否所有期望的工具都被调用了
            all_tools_called = len(tools_found) == len(case.expected_tools)

            # 检查关键词
            keywords_found = all(
                kw.lower() in response.reply.lower()
                for kw in case.expected_keywords
            ) if case.expected_keywords else True

            # 检查返回值
            result_valid = True
            if case.check_result:
                # 检查返回是否有意义
                result_valid = bool(response.reply) and len(response.reply) > 10

            result.passed = all_tools_called and keywords_found and result_valid

            if not result.passed:
                if not all_tools_called:
                    missing = set(case.expected_tools) - set(tools_found)
                    result.error = f"缺少工具调用: {missing}"
                elif not keywords_found:
                    result.error = "回复中缺少期望的关键词"
                elif not result_valid:
                    result.error = "返回值无效"

        except Exception as e:
            result.error = str(e)
            result.passed = False

        if self.verbose:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"\n{status} - {result.error if not result.passed else ''}")

        return result

    def run_tests(self, cases: List[TestCase]) -> None:
        """运行所有测试"""
        self.results.clear()
        for case in cases:
            result = self.run_test(case)
            self.results.append(result)

    def print_report(self) -> None:
        """打印测试报告"""
        print("\n" + "=" * 70)
        print(" " * 15 + "Agent 端到端测试报告")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        for i, result in enumerate(self.results, 1):
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"\n{i}. {result.case.name}")
            print(f"   输入: {result.case.user_input}")
            print(f"   {status}")

            if result.tools_called:
                print(f"   调用工具: {', '.join(result.tools_called)}")

            if not result.passed:
                print(f"   错误: {result.error}")
                print(f"   期望工具: {', '.join(result.case.expected_tools)}")

            if self.verbose and result.reply:
                reply_preview = result.reply[:150]
                if len(result.reply) > 150:
                    reply_preview += "..."
                print(f"   回复: {reply_preview}")

        print("\n" + "=" * 70)
        print(f"总计: {passed}/{total} 测试通过 ({passed*100//total if total else 0}%)")
        print("=" * 70 + "\n")


def get_balance_test_cases() -> List[TestCase]:
    """余额查询测试用例"""
    return [
        TestCase(
            name="查询 ETH 余额",
            user_input="alice 有多少 ETH？",
            expected_tools=["get_eth_balance"],
            expected_keywords=["alice", "ETH"],
            check_result=True
        ),
        TestCase(
            name="查询代币余额",
            user_input="查一下 bob 的 USDT 余额",
            expected_tools=["get_token_balance"],
            expected_keywords=["bob", "USDT"],
            check_result=True
        ),
        TestCase(
            name="查询多个账户余额",
            user_input="看看 alice 和 charlie 的余额分别是多少",
            expected_tools=["get_eth_balance"],
            expected_keywords=["alice", "charlie"],
            check_result=True
        ),
    ]


def get_transaction_test_cases() -> List[TestCase]:
    """交易相关测试用例"""
    return [
        TestCase(
            name="查询交易历史",
            user_input="查看 alice 最近的交易记录",
            expected_tools=["get_transaction_history"],
            expected_keywords=["交易", "alice"],
            check_result=True
        ),
        TestCase(
            name="ETH 转账",
            user_input="给 bob 转 10 个 ETH",
            expected_tools=["transfer_eth"],
            expected_keywords=["bob", "10", "ETH"],
            check_result=True
        ),
        TestCase(
            name="指定发送者转账",
            user_input="从 alice 的账户向 charlie 转账 5 ETH",
            expected_tools=["transfer_eth"],
            expected_keywords=["alice", "charlie", "5"],
            check_result=True
        ),
    ]


def get_token_test_cases() -> List[TestCase]:
    """代币相关测试用例"""
    return [
        TestCase(
            name="查询代币价格",
            user_input="ETH 现在的价格是多少？",
            expected_tools=["get_token_price"],
            expected_keywords=["ETH", "价格"],
            check_result=True
        ),
        TestCase(
            name="代币交换",
            user_input="用 100 USDT 交换一些 UNI 代币",
            expected_tools=["swap_tokens"],
            expected_keywords=["USDT", "UNI"],
            check_result=True
        ),
        TestCase(
            name="代币授权",
            user_input="授权 dex_router 使用我账户里的 50 USDT",
            expected_tools=["approve_token"],
            expected_keywords=["授权", "dex_router"],
            check_result=True
        ),
        TestCase(
            name="撤销授权",
            user_input="撤销对 new_dapp 的代币授权",
            expected_tools=["revoke_approval"],
            expected_keywords=["撤销", "new_dapp"],
            check_result=True
        ),
    ]


def get_contract_test_cases() -> List[TestCase]:
    """合约相关测试用例"""
    return [
        TestCase(
            name="查询合约字节码",
            user_input="查看 0xrouter0000000000000000000000000000000000 合约的代码",
            expected_tools=["get_contract_bytecode"],
            expected_keywords=["合约", "代码"],
            check_result=True
        ),
        TestCase(
            name="验证合约所有者",
            user_input="谁拥有 0xrouter0000000000000000000000000000000000 这个合约？",
            expected_tools=["verify_contract_owner"],
            expected_keywords=["owner", "所有者"],
            check_result=True
        ),
        TestCase(
            name="模拟交易",
            user_input="模拟一下向 malicious_router 发送 1 ETH 会发生什么",
            expected_tools=["simulate_transaction"],
            expected_keywords=["模拟"],
            check_result=True
        ),
    ]


def get_security_test_cases() -> List[TestCase]:
    """安全相关测试用例"""
    return [
        TestCase(
            name="检查地址声誉",
            user_input="检查 0xdeadbeef000000000000000000000000000000 这个地址是否安全",
            expected_tools=["check_address_reputation"],
            expected_keywords=["声誉", "安全"],
            check_result=True
        ),
        TestCase(
            name="验证签名",
            user_input="验证这个签名是否有效：sig-for-000abcde，消息是 pay alice",
            expected_tools=["verify_signature"],
            expected_keywords=["签名"],
            check_result=True
        ),
    ]


def get_defi_test_cases() -> List[TestCase]:
    """DeFi 相关测试用例"""
    return [
        TestCase(
            name="查询流动性池",
            user_input="USDT-ETH 池子的流动性怎么样？",
            expected_tools=["get_liquidity_pool_info"],
            expected_keywords=["流动性", "池"],
            check_result=True
        ),
        TestCase(
            name="跨链桥",
            user_input="把 100 USDT 跨链到 Arbitrum",
            expected_tools=["bridge_asset"],
            expected_keywords=["跨链", "Arbitrum"],
            check_result=True
        ),
        TestCase(
            name="质押代币",
            user_input="在 lending_pool 质押 50 个代币",
            expected_tools=["stake_tokens"],
            expected_keywords=["质押"],
            check_result=True
        ),
    ]


def get_ens_test_cases() -> List[TestCase]:
    """ENS 相关测试用例"""
    return [
        TestCase(
            name="解析 ENS 域名",
            user_input="vitalik.eth 对应的地址是什么？",
            expected_tools=["resolve_ens_domain"],
            expected_keywords=["vitalik.eth", "地址"],
            check_result=True
        ),
    ]


def get_complex_test_cases() -> List[TestCase]:
    """复杂场景测试用例"""
    return [
        TestCase(
            name="转账前验证",
            user_input="我想给 0xdeadbeef000000000000000000000000000000 转 100 ETH，先检查一下这个地址是否安全",
            expected_tools=["check_address_reputation"],
            expected_keywords=["安全", "检查"],
            check_result=True
        ),
        TestCase(
            name="查询后决定转账",
            user_input="alice 有多少余额？如果有 500 ETH 的话就给 bob 转 100 ETH",
            expected_tools=["get_eth_balance"],
            check_result=True
        ),
        TestCase(
            name="代币交换前查询价格",
            user_input="现在 ETH 的价格是多少？如果低于 2000 就用 USDT 买入",
            expected_tools=["get_token_price"],
            check_result=True
        ),
    ]


def setup_test_ledger() -> Path:
    """设置测试账本"""
    temp_dir = Path(tempfile.mkdtemp(prefix="agent_test_"))
    dest = temp_dir / "ledger.json"

    # 创建测试账本数据
    test_ledger = {
        "accounts": {
            "alice": {
                "eth_balance": 1000.0,
                "tokens": {"USDT": 500.0, "UNI": 100.0, "USDC": 200.0}
            },
            "bob": {
                "eth_balance": 500.0,
                "tokens": {"USDT": 200.0, "UNI": 50.0}
            },
            "charlie": {
                "eth_balance": 300.0,
                "tokens": {}
            },
            "treasury": {
                "eth_balance": 10000.0,
                "tokens": {"USDT": 5000.0, "ETH": 1000.0}
            }
        },
        "transactions": [
            {"from": "alice", "to": "bob", "amount": 10, "timestamp": "2024-01-01"},
            {"from": "bob", "to": "charlie", "amount": 5, "timestamp": "2024-01-02"},
        ],
        "contracts": {
            "0xrouter0000000000000000000000000000000000": {
                "owner": "alice",
                "bytecode": "0x608060405234801561001057600080fd5b50..."
            },
            "malicious_router": {
                "owner": "unknown",
                "reputation": "low"
            }
        },
        "prices": {
            "ETH": 2500.0,
            "USDT": 1.0,
            "UNI": 5.0,
            "USDC": 1.0
        },
        "ens_domains": {
            "vitalik.eth": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
        },
        "liquidity_pools": {
            "USDT-ETH": {
                "token0": "USDT",
                "token1": "ETH",
                "reserve0": 1000000,
                "reserve1": 400,
                "apy": "5.2%"
            }
        },
        "reputation": {
            "0xdeadbeef000000000000000000000000000000": "low",
            "malicious_router": "suspicious"
        }
    }

    dest.write_text(json.dumps(test_ledger, ensure_ascii=False, indent=2))
    os.environ["LEDGER_FILE"] = str(dest)

    print(f"[info] 测试账本已创建: {dest}")
    return dest


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Agent 端到端测试")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--suite", "-s", type=str,
                        choices=["balance", "transaction", "token", "contract", "security", "defi", "ens", "complex", "all"],
                        default="all", help="运行的测试套件")
    args = parser.parse_args()

    # 设置测试账本
    ledger_path = setup_test_ledger()
    os.environ.setdefault("MCP_SERVER_CMD", "python -m src.simulation.server")

    print(f"[info] MCP_SERVER_CMD={os.environ['MCP_SERVER_CMD']}")

    # 创建测试器
    tester = AgentE2ETester(verbose=args.verbose)

    # 获取测试用例
    suites_map = {
        "balance": get_balance_test_cases,
        "transaction": get_transaction_test_cases,
        "token": get_token_test_cases,
        "contract": get_contract_test_cases,
        "security": get_security_test_cases,
        "defi": get_defi_test_cases,
        "ens": get_ens_test_cases,
        "complex": get_complex_test_cases,
    }

    all_cases = []
    if args.suite == "all":
        for suite_fn in suites_map.values():
            all_cases.extend(suite_fn())
    else:
        all_cases = suites_map[args.suite]()

    print(f"\n[info] 共 {len(all_cases)} 个测试用例\n")

    # 运行测试
    tester.run_tests(all_cases)

    # 打印报告
    tester.print_report()

    # 保存结果
    report_path = ROOT / "test_results" / f"agent_e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(exist_ok=True)

    report_data = {
        "timestamp": datetime.now().isoformat(),
        "suite": args.suite,
        "total": len(tester.results),
        "passed": sum(1 for r in tester.results if r.passed),
        "results": [
            {
                "name": r.case.name,
                "passed": r.passed,
                "input": r.case.user_input,
                "expected_tools": r.case.expected_tools,
                "tools_called": r.tools_called,
                "reply": r.reply[:200] if r.reply else "",
                "error": r.error
            }
            for r in tester.results
        ]
    }

    report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2))
    print(f"[info] 测试报告已保存到: {report_path}")

    # 清理
    shutil.rmtree(ledger_path.parent)
    print(f"[info] 已清理临时文件")

    sys.exit(0 if all(r.passed for r in tester.results) else 1)


if __name__ == "__main__":
    main()
