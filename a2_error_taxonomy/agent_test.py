"""真实多轮会话压测：用生产配置（默认 1M 窗口）跑长会话，统计增长与成本。

和另外三个测试的分工：
  test_context / test_loop   假对象，零 API，测分支逻辑
  test_session（本文件）      真 API，真实多轮，测"长会话下系统怎么变"
  agent_test.py              真 API 冒烟，测"典型问法能不能跑通"

跑：.venv\\Scripts\\python.exe -m a2_error_taxonomy.test_session
"""
import logging
import os
import time

from dotenv import load_dotenv

from a2_error_taxonomy.llm_client import LLMClient
from a2_error_taxonomy.llm_types import (TextPart, ToolCallPart,
                                         ToolResultPart, user_message)
from a2_error_taxonomy.loop import run
from a2_error_taxonomy.tool_types import Session

logging.basicConfig(level=logging.WARNING,      # 压低日志，让统计表是主角
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("test_session")

load_dotenv()

SESSION = Session(operator_id="op_01", role="csr",
                  scopes=frozenset({"u_10237", "u_10238", "u_10239"}))

# 一次长会话的追问序列 —— 覆盖不同工具路径 + 让上下文持续增长
TURNS = [
    "查一下华信最近的订单",
    "是北京那家华信",
    "有没有退款中的",
    "那已发货还没签收的有几笔",
    "帮我看看上海鼎新贸易的单子",
    "鼎新最近一单是什么时候",
]

_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


class LedgerClient:
    """包一层 LLMClient，记录每次聊天返回的 usage —— 用来画增长曲线。

    ★ 为什么包一层而不是改 LLMClient：
      LLMClient 已经有 usage_sink，但那是"落盘"用的；
      测试需要的是"在内存里按顺序拿到每轮的数字"。
      包装比改生产代码便宜。
    """

    def __init__(self, inner: LLMClient):
        self.inner = inner
        self.records: list[dict] = []

    def chat(self, messages, tools=None, **kw):
        resp = self.inner.chat(messages, tools=tools, **kw)
        self.records.append({
            "in": resp.input_tokens,
            "cache": resp.cache_read_tokens,
            "out": resp.output_tokens,
            "ms": resp.total_ms,
            "stop": str(resp.stop_reason),
        })
        return resp

    # 供 run() 内部调用（比如压缩后清基线）
    def __getattr__(self, name):
        return getattr(self.inner, name)


def _assert_pairing(msgs) -> tuple[bool, str]:
    """每个 ToolCall 都要有对应的 ToolResult —— 否则 provider 会 400。"""
    calls = {p.id for m in msgs for p in m.parts if isinstance(p, ToolCallPart)}
    results = {p.tool_call_id for m in msgs for p in m.parts
               if isinstance(p, ToolResultPart)}
    if calls == results:
        return True, f"{len(calls)} 对"
    return False, f"缺结果: {calls - results}；多结果: {results - calls}"


def case_long_session() -> None:
    """一次 6 轮的连续会话：上下文持续增长，观察 token / 缓存 / 成本。"""
    print("\n" + "=" * 78)
    print("用例：6 轮连续会话（生产配置，默认窗口）")
    print("=" * 78)

    inner = LLMClient(
        provider="deepseek",
        model=os.getenv("ANTHROPIC_MODEL_NAME"),
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url=os.getenv("ANTHROPIC_BASE_URL"),
        session_id="sess_long",
    )
    client = LedgerClient(inner)

    # ★ 整个会话共用一个 messages 列表 —— 这是"连续会话"的定义
    msgs = []
    t0 = time.perf_counter()

    for i, q in enumerate(TURNS, start=1):
        before = len(client.records)
        msgs.append(user_message(q))
        try:
            answer = run(SESSION, msgs, client, max_turns=6)
        except Exception as e:
            check(f"轮 {i} 正常结束", False, f"{type(e).__name__}: {e}")
            continue

        n_calls = len(client.records) - before
        last = client.records[-1]
        check(f"轮 {i} 正常结束（{n_calls} 次调用）", n_calls > 0,
              f"stop={last['stop']} answer={answer[:24]!r}")

        ok, detail = _assert_pairing(msgs)
        check(f"轮 {i} 配对完整", ok, detail)

    wall = time.perf_counter() - t0

    # ---- 增长曲线 ----
    print("\n每次模型调用的 token 构成（按时间顺序）")
    print(f"  {'#':<4}{'未命中in':>9}{'缓存命中':>10}{'总输入':>9}"
          f"{'输出':>7}{'命中率':>8}{'延迟':>9}")
    for i, r in enumerate(client.records, start=1):
        total_in = r["in"] + r["cache"]
        rate = r["cache"] / total_in * 100 if total_in else 0
        print(f"  {i:<4}{r['in']:>9,}{r['cache']:>10,}{total_in:>9,}"
              f"{r['out']:>7,}{rate:>7.0f}%{r['ms']:>8.0f}ms")

    # ---- 增长是否真的是"台阶式"（工具结果造成的） ----
    ins = [r["in"] + r["cache"] for r in client.records]
    check("输入随轮次增长", ins[-1] > ins[0],
          f"{ins[0]:,} -> {ins[-1]:,}")
    growth = ins[-1] / ins[0] if ins[0] else 1
    print(f"\n  输入增长：{ins[0]:,} -> {ins[-1]:,}（{growth:.1f}×）")

    # ---- 缓存命中率 ----
    tot_in = sum(r["in"] for r in client.records)
    tot_cache = sum(r["cache"] for r in client.records)
    rate = tot_cache / (tot_in + tot_cache) * 100
    check("缓存整体命中率 > 50%", rate > 50, f"{rate:.0f}%")

    # ---- 成本 ----
    # 空闲时段单价（元/百万）；高峰要翻倍
    cost = (tot_cache * 0.02 + tot_in * 1.0
            + sum(r["out"] for r in client.records) * 4.0) / 1_000_000
    print(f"\n  会话合计：{len(client.records)} 次调用 | "
          f"未命中 {tot_in:,} | 缓存 {tot_cache:,} | "
          f"输出 {sum(r['out'] for r in client.records):,}")
    print(f"  成本 ≈ ¥{cost:.5f}（空闲时段）| 墙钟 {wall:.1f}s")

    # ---- 上下文规模（用来回答"离爆窗多远"） ----
    final_tokens = sum(r["in"] + r["cache"] for r in client.records[-2:]) // 2
    print(f"  末轮输入 ≈ {final_tokens:,} token | 窗口 {inner.context_window:,}")
    print(f"  余量：{inner.context_window / max(final_tokens, 1):.0f}×")


if __name__ == "__main__":
    case_long_session()

    passed = sum(1 for _, ok, _ in _results if ok)
    print(f"\n合计：{passed} passed, {len(_results) - passed} failed")