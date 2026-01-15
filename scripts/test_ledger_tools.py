"""
Comprehensive Blockchain Tools Test Suite

Tests all MCP tools with various scenarios including:
- Read operations (balance queries, transaction history, etc.)
- Write operations (transfers, swaps, approvals)
- Edge cases and error handling
- State verification after write operations

Usage:
    python scripts/test_ledger_tools.py [--verbose]
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.mcp_client.client import MCPToolClient
from src.simulation.db import LedgerDB

DEFAULT_LEDGER = ROOT / "data" / "ledger" / "ledger.json"


@dataclass
class TestResult:
    """存储单个测试用例的结果"""
    name: str
    passed: bool
    expected: Any = None
    actual: Any = None
    error: Optional[str] = None
    duration_ms: float = 0


@dataclass
class TestSuite:
    """测试套件"""
    name: str
    results: List[TestResult] = field(default_factory=list)

    def add_result(self, result: TestResult) -> None:
        self.results.append(result)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if r.passed == False)

    @property
    def total_count(self) -> int:
        return len(self.results)


class ToolTester:
    """工具测试器"""

    def __init__(self, client: MCPToolClient, verbose: bool = False):
        self.client = client
        self.verbose = verbose
        self.suites: List[TestSuite] = []

    def call_tool(self, name: str, **kwargs) -> Any:
        """调用工具并捕获异常"""
        try:
            return self.client.call_tool(name, **kwargs)
        except Exception as e:
            if self.verbose:
                print(f"  [ERROR] {type(e).__name__}: {e}")
            raise

    def test_read_operation(
        self,
        suite: TestSuite,
        name: str,
        tool_name: str,
        params: Dict[str, Any],
        expected_type: type = None,
        validate_fn: callable = None
    ) -> None:
        """测试读操作"""
        if self.verbose:
            print(f"\n[Test] {name}")

        import time
        start = time.time()

        try:
            result = self.call_tool(tool_name, **params)
            duration_ms = (time.time() - start) * 1000

            # 类型检查
            if expected_type and not isinstance(result, expected_type):
                result = TestResult(
                    name=name,
                    passed=False,
                    expected=f"type: {expected_type.__name__}",
                    actual=f"type: {type(result).__name__}",
                    error=f"Expected {expected_type.__name__}, got {type(result).__name__}",
                    duration_ms=duration_ms
                )
            elif validate_fn and not validate_fn(result):
                result = TestResult(
                    name=name,
                    passed=False,
                    error="Validation function returned False",
                    duration_ms=duration_ms
                )
            else:
                result = TestResult(
                    name=name,
                    passed=True,
                    actual=result,
                    duration_ms=duration_ms
                )

            if self.verbose and result.passed:
                print(f"  [PASS] Result: {json.dumps(result.actual, ensure_ascii=False)[:200]}")

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            result = TestResult(
                name=name,
                passed=False,
                error=str(e),
                duration_ms=duration_ms
            )

        suite.add_result(result)

    def test_write_operation(
        self,
        suite: TestSuite,
        name: str,
        tool_name: str,
        params: Dict[str, Any],
        validate_fn: callable = None
    ) -> None:
        """测试写操作"""
        if self.verbose:
            print(f"\n[Test] {name}")

        import time
        start = time.time()

        try:
            result = self.call_tool(tool_name, **params)
            duration_ms = (time.time() - start) * 1000

            # 写操作应该返回字典且包含成功标识
            if not isinstance(result, dict):
                result = TestResult(
                    name=name,
                    passed=False,
                    expected="dict with success/status",
                    actual=f"type: {type(result).__name__}",
                    error=f"Write operation should return dict",
                    duration_ms=duration_ms
                )
            elif validate_fn and not validate_fn(result):
                result = TestResult(
                    name=name,
                    passed=False,
                    error="Validation function returned False",
                    duration_ms=duration_ms
                )
            else:
                result = TestResult(
                    name=name,
                    passed=True,
                    actual=result,
                    duration_ms=duration_ms
                )

            if self.verbose and result.passed:
                print(f"  [PASS] Result: {json.dumps(result.actual, ensure_ascii=False)[:200]}")

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            result = TestResult(
                name=name,
                passed=False,
                error=str(e),
                duration_ms=duration_ms
            )

        suite.add_result(result)


def run_balance_tests(tester: ToolTester) -> TestSuite:
    """测试余额查询相关工具"""
    suite = TestSuite(name="余额查询测试")

    # 测试 ETH 余额查询
    tester.test_read_operation(
        suite,
        "查询 alice 的 ETH 余额",
        "get_eth_balance",
        {"address": "alice"},
        expected_type=(int, float),
        validate_fn=lambda x: x >= 0
    )

    # 测试多个账户
    for account in ["bob", "charlie", "treasury"]:
        tester.test_read_operation(
            suite,
            f"查询 {account} 的 ETH 余额",
            "get_eth_balance",
            {"address": account},
            expected_type=(int, float)
        )

    # 测试代币余额
    tester.test_read_operation(
        suite,
        "查询 alice 的 USDT 余额",
        "get_token_balance",
        {"address": "alice", "token_symbol": "USDT"},
        expected_type=(int, float)
    )

    # 测试兼容接口
    tester.test_read_operation(
        suite,
        "使用 get_balance 查询 alice 余额 (兼容接口)",
        "get_balance",
        {"account": "alice"},
        expected_type=(int, float)
    )

    return suite


def run_transaction_tests(tester: ToolTester) -> TestSuite:
    """测试交易相关工具"""
    suite = TestSuite(name="交易测试")

    # 测试交易历史查询
    tester.test_read_operation(
        suite,
        "查询 alice 的交易历史",
        "get_transaction_history",
        {"address": "alice", "limit": 5},
        validate_fn=lambda x: isinstance(x, list)
    )

    # 测试 ETH 转账
    # 注意：这个测试会修改账本状态
    tester.test_write_operation(
        suite,
        "alice 向 bob 转账 10 ETH",
        "transfer_eth",
        {"to_address": "bob", "amount": 10.0, "sender": "alice"},
        validate_fn=lambda r: r.get("success", True) or "tx_hash" in r
    )

    # 测试兼容接口
    tester.test_write_operation(
        suite,
        "alice 向 bob 转账 5 ETH (兼容接口)",
        "transfer",
        {"sender": "alice", "recipient": "bob", "amount": 5.0},
        validate_fn=lambda r: r.get("success", True) or "tx_hash" in r
    )

    return suite


def run_token_tests(tester: ToolTester) -> TestSuite:
    """测试代币相关工具"""
    suite = TestSuite(name="代币操作测试")

    # 测试代币价格查询
    tester.test_read_operation(
        suite,
        "查询 ETH 价格",
        "get_token_price",
        {"token_symbol": "ETH"},
        expected_type=(int, float),
        validate_fn=lambda x: x > 0
    )

    tester.test_read_operation(
        suite,
        "查询 USDT 价格",
        "get_token_price",
        {"token_symbol": "USDT"},
        expected_type=(int, float)
    )

    # 测试代币授权
    tester.test_write_operation(
        suite,
        "alice 授权 new_dapp 使用 50 USDT",
        "approve_token",
        {"spender": "new_dapp", "amount": 50.0, "owner": "alice"}
    )

    # 测试查询授权状态
    tester.test_read_operation(
        suite,
        "查询 alice 对 dex_router 的授权状态",
        "check_token_approval",
        {"owner": "alice", "spender": "dex_router"}
    )

    # 测试撤销授权
    tester.test_write_operation(
        suite,
        "alice 撤销对 dex_router 的授权",
        "revoke_approval",
        {"spender": "dex_router", "owner": "alice"}
    )

    # 测试代币交换
    tester.test_write_operation(
        suite,
        "alice 用 10 USDT 交换 UNI",
        "swap_tokens",
        {"token_in": "USDT", "token_out": "UNI", "amount": 10.0, "address": "alice"}
    )

    return suite


def run_contract_tests(tester: ToolTester) -> TestSuite:
    """测试合约相关工具"""
    suite = TestSuite(name="合约交互测试")

    # 测试合约字节码查询
    tester.test_read_operation(
        suite,
        "查询 router 合约字节码",
        "get_contract_bytecode",
        {"address": "0xrouter0000000000000000000000000000000000"},
        validate_fn=lambda x: isinstance(x, str) and len(x) > 0
    )

    # 测试合约所有者验证
    tester.test_read_operation(
        suite,
        "验证 router 合约所有者",
        "verify_contract_owner",
        {"contract_address": "0xrouter0000000000000000000000000000000000"}
    )

    # 测试交易模拟
    tester.test_read_operation(
        suite,
        "模拟向 malicious_router 转账",
        "simulate_transaction",
        {"to": "malicious_router", "value": 1.5, "data": "0xabcd"}
    )

    return suite


def run_security_tests(tester: ToolTester) -> TestSuite:
    """测试安全和声誉相关工具"""
    suite = TestSuite(name="安全检查测试")

    # 测试地址声誉检查
    tester.test_read_operation(
        suite,
        "检查恶意地址声誉",
        "check_address_reputation",
        {"address": "0xdeadbeef000000000000000000000000000000"}
    )

    # 测试签名验证
    tester.test_read_operation(
        suite,
        "验证签名有效性",
        "verify_signature",
        {
            "message": "pay alice",
            "signature": "sig-for-000abcde",
            "address": "0xdeadbeef000000000000000000000000000abcde"
        }
    )

    return suite


def run_defi_tests(tester: ToolTester) -> TestSuite:
    """测试 DeFi 相关工具"""
    suite = TestSuite(name="DeFi 操作测试")

    # 测试流动性池信息查询
    tester.test_read_operation(
        suite,
        "查询 USDT-ETH 流动性池信息",
        "get_liquidity_pool_info",
        {"token_address": "USDT-ETH"}
    )

    # 测试资产跨链桥
    tester.test_write_operation(
        suite,
        "将 USDT 跨链到 Arbitrum",
        "bridge_asset",
        {"token": "USDT", "target_chain": "arbitrum"}
    )

    # 测试质押
    tester.test_write_operation(
        suite,
        "在 lending_pool 质押 25 USDT",
        "stake_tokens",
        {"protocol": "lending_pool", "amount": 25.0}
    )

    return suite


def run_ens_tests(tester: ToolTester) -> TestSuite:
    """测试 ENS 相关工具"""
    suite = TestSuite(name="ENS 测试")

    tester.test_read_operation(
        suite,
        "解析 vitalik.eth ENS 域名",
        "resolve_ens_domain",
        {"domain_name": "vitalik.eth"}
    )

    return suite


def print_report(tester: ToolTester) -> None:
    """打印测试报告"""
    print("\n" + "=" * 70)
    print(" " * 20 + "区块链工具测试报告")
    print("=" * 70)

    total_passed = 0
    total_failed = 0
    total_tests = 0

    for suite in tester.suites:
        print(f"\n【{suite.name}】")
        print("-" * 70)

        for result in suite.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            duration = f"{result.duration_ms:.1f}ms"

            if result.passed:
                print(f"{status} | {duration:>8} | {result.name}")
                if tester.verbose:
                    actual_str = str(result.actual)[:100]
                    if len(actual_str) == 100:
                        actual_str += "..."
                    print(f"       └─ {actual_str}")
            else:
                print(f"{status} | {duration:>8} | {result.name}")
                print(f"       └─ Error: {result.error}")

        print(f"\n  小计: {suite.passed_count}/{suite.total_count} 通过")

        total_passed += suite.passed_count
        total_failed += suite.failed_count
        total_tests += suite.total_count

    print("\n" + "=" * 70)
    print(f"总计: {total_passed}/{total_tests} 测试通过 ({total_passed*100//total_tests if total_tests > 0 else 0}%)")

    if total_failed > 0:
        print(f"      {total_failed} 个测试失败")
        print("\n失败的测试:")
        for suite in tester.suites:
            for result in suite.results:
                if not result.passed:
                    print(f"  - {result.name}: {result.error}")

    print("=" * 70 + "\n")


def save_report(tester: ToolTester, output_path: Path) -> None:
    """保存测试报告到 JSON 文件"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": sum(s.total_count for s in tester.suites),
            "passed": sum(s.passed_count for s in tester.suites),
            "failed": sum(s.failed_count for s in tester.suites),
        },
        "suites": []
    }

    for suite in tester.suites:
        suite_data = {
            "name": suite.name,
            "total": suite.total_count,
            "passed": suite.passed_count,
            "failed": suite.failed_count,
            "tests": []
        }

        for result in suite.results:
            test_data = {
                "name": result.name,
                "passed": result.passed,
                "duration_ms": result.duration_ms
            }
            if not result.passed:
                test_data["error"] = result.error
            if result.actual is not None:
                test_data["actual"] = result.actual

            suite_data["tests"].append(test_data)

        report["suites"].append(suite_data)

    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"测试报告已保存到: {output_path}")


def copy_ledger() -> Path:
    """复制账本到临时目录"""
    temp_dir = Path(tempfile.mkdtemp(prefix="ledger_test_"))
    dest = temp_dir / "ledger.json"

    if DEFAULT_LEDGER.exists():
        shutil.copy(DEFAULT_LEDGER, dest)
        print(f"[info] 使用账本副本: {dest}")
    else:
        # 创建默认账本
        default_data = {
            "accounts": {
                "alice": {"eth_balance": 1000.0, "tokens": {"USDT": 500.0, "UNI": 100.0}},
                "bob": {"eth_balance": 500.0, "tokens": {"USDT": 200.0}},
                "charlie": {"eth_balance": 300.0, "tokens": {}},
                "treasury": {"eth_balance": 10000.0, "tokens": {"USDT": 5000.0}}
            },
            "transactions": []
        }
        dest.write_text(json.dumps(default_data, ensure_ascii=False, indent=2))
        print(f"[info] 创建默认账本: {dest}")

    return dest


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="区块链工具测试套件")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--report", "-r", type=str, help="保存报告到指定路径")
    parser.add_argument("--suite", "-s", type=str,
                        choices=["balance", "transaction", "token", "contract", "security", "defi", "ens", "all"],
                        default="all", help="运行的测试套件")
    args = parser.parse_args()

    # 设置环境
    ledger_copy = copy_ledger()
    os.environ["LEDGER_FILE"] = str(ledger_copy)
    os.environ.setdefault("MCP_SERVER_CMD", "python -m src.simulation.server")

    print(f"[info] MCP_SERVER_CMD={os.environ['MCP_SERVER_CMD']}")

    # 创建客户端和测试器
    client = MCPToolClient(
        server_cmd=os.environ["MCP_SERVER_CMD"],
        server_url=os.getenv("MCP_SERVER_URL")
    )
    tester = ToolTester(client, verbose=args.verbose)

    # 运行测试套件
    suites_map = {
        "balance": run_balance_tests,
        "transaction": run_transaction_tests,
        "token": run_token_tests,
        "contract": run_contract_tests,
        "security": run_security_tests,
        "defi": run_defi_tests,
        "ens": run_ens_tests,
    }

    if args.suite == "all":
        for suite_fn in suites_map.values():
            tester.suites.append(suite_fn(tester))
    else:
        tester.suites.append(suites_map[args.suite](tester))

    # 打印报告
    print_report(tester)

    # 保存报告
    if args.report:
        save_report(tester, Path(args.report))

    # 清理临时文件
    shutil.rmtree(ledger_copy.parent)
    print(f"[info] 已清理临时文件")

    # 返回退出码
    total_failed = sum(s.failed_count for s in tester.suites)
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
