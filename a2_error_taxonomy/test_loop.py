"""A4-4：用假 client 注入路径测试。零 API 消耗。

跑：.venv\\Scripts\\python.exe a2_error_taxonomy\\test_loop.py
"""
from a2_error_taxonomy.error_type import ContextLengthExceeded
from a2_error_taxonomy.llm_types import (LLMResponse, StopReason, ToolCallPart,
                                         ToolResultPart, user_message)
from a2_error_taxonomy.loop import run
from a2_error_taxonomy.tool_types import Session


# ==================== 测试替身 ====================

class ScriptedClient:
    """按脚本依次返回预设响应。鸭子类型 —— 不继承 LLMClient。

    只要具备 chat(messages, tools=...) 就能替换 run() 里的真 client。
    """

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[int] = []          # 每次调用时的 messages 长度

    def chat(self, messages, tools=None, system=None, max_tokens=None):
        self.calls.append(len(messages))
        if not self._responses:
            # ★ 脚本耗尽要显式报错，不要静默返回 None ——
            #   "多调了一次模型"本身就是一个要抓的 bug
            raise AssertionError("脚本响应已耗尽 —— run() 比你预期多调用了一次")
        return self._responses.pop(0)


def resp(stop, *, text="", tools=(), out_tokens=10) -> LLMResponse:
    """构造 LLMResponse。tools 是 [(id, name, args), ...]。"""
    return LLMResponse(
        id="msg_fake", model="fake", created_at=0.0, text=text,
        content_blocks=([{"type": "text", "text": text}] if text else [])
                       + [{"type": "tool_use", "id": i, "name": n, "input": a}
                          for i, n, a in tools],
        tool_calls=[ToolCallPart(id=i, name=n, arguments=a) for i, n, a in tools],
        stop_reason=stop, input_tokens=100, output_tokens=out_tokens,
        total_ms=1.0,
    )


SESSION = Session(operator_id="op_01", role="csr",
                  scopes=frozenset({"u_10237", "u_10238", "u_10239"}))

_results: list[tuple[str, bool]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def tool_results(msgs) -> list[ToolResultPart]:
    return [p for m in msgs for p in m.parts if isinstance(p, ToolResultPart)]


# ==================== T1：去重守卫 ====================

def t1_duplicate_guard():
    """同一工具同一参数连续调用两次 → 第二次被拦，且原因要能让模型看到。"""
    fc = ScriptedClient([
        resp(StopReason.TOOL_USE,
             tools=[("c1", "search_orders", {"state": "shipped"})]),
        resp(StopReason.TOOL_USE,                       # 模拟原样复读
             tools=[("c2", "search_orders", {"state": "shipped"})]),
        resp(StopReason.END_TURN, text="最终回答"),
    ])
    msgs = [user_message("查已发货订单")]
    out = run(SESSION, msgs, fc, max_turns=5)

    tr = tool_results(msgs)
    check("T1 两条 tool 结果", len(tr) == 2, f"实际 {len(tr)}")
    check("T1 第一条真实执行", tr[0].is_error is False)
    check("T1 第二条被拦截", len(tr) > 1 and tr[1].is_error is True)
    # ★ 核心断言：错误原因必须进 content，否则模型看不到
    check("T1 拦截原因对模型可见",
          len(tr) > 1 and "重复" in tr[1].content,
          tr[1].content[:70] if len(tr) > 1 else "")
    check("T1 最终返回正常", out == "最终回答", repr(out))


# ==================== T2 / T3：MAX_TOKENS ====================

def t2_max_tokens_empty():
    fc = ScriptedClient([resp(StopReason.MAX_TOKENS, text="", out_tokens=8192)])
    out = run(SESSION, [user_message("q")], fc)
    check("T2 不返回空串", out.strip() != "", repr(out))
    check("T2 提示未能产出内容", "未能产出内容" in out, out)
    check("T2 不重试（只调一次）", len(fc.calls) == 1, f"实际 {len(fc.calls)}")


def t3_max_tokens_partial():
    fc = ScriptedClient([resp(StopReason.MAX_TOKENS, text="半截回答", out_tokens=8192)])
    out = run(SESSION, [user_message("q")], fc)
    check("T3 保留部分输出", out.startswith("半截回答"), out)
    check("T3 标注被截断", "被截断" in out, out)


# ==================== T4：REFUSAL ====================

def t4_refusal():
    fc = ScriptedClient([resp(StopReason.REFUSAL, text="我不能回答这个")])
    out = run(SESSION, [user_message("q")], fc)
    check("T4 返回降级话术", "无法回答" in out, out)
    check("T4 不重试（只调一次）", len(fc.calls) == 1, f"实际 {len(fc.calls)}")


# ==================== T5：上下文窗口超限 ====================

def t5_window_exceeded():
    """窗口超限必须是 ContextLengthExceeded —— 它是给 A5 的信号，不是用户话术。

    ★ 断言"子类"而不是 DomainError 父类：否则验不出"这个信号有专属类型"。
    ★ 断言"没被兜底吞掉"：兜底会 return e.content，那样 A5 永远收不到信号。
    """
    fc = ScriptedClient([resp(StopReason.MODEL_CONTEXT_WINDOW_EXCEEDED)])
    raised = None
    try:
        run(SESSION, [user_message("q")], fc)
    except ContextLengthExceeded as e:
        raised = e
    check("T5 抛 ContextLengthExceeded", raised is not None, repr(raised))
    check("T5 异常信息含'压缩'", bool(raised) and "压缩" in str(raised), str(raised))


# ==================== T6：撞 max_turns ====================

def t6_max_turns():
    """每次参数都不同（绕过去重守卫）→ 只能靠 max_turns 兜底。"""
    responses = [
        resp(StopReason.TOOL_USE,
             tools=[(f"c{i}", "search_orders", {"created_from": f"2024-01-{i:02d}"})])
        for i in range(1, 6)            # 给 5 个，但 max_turns=3
    ]
    fc = ScriptedClient(responses)
    out = run(SESSION, [user_message("q")], fc, max_turns=3)
    check("T6 撞上限停下", "轮次上限" in out, out)
    check("T6 调用次数 == max_turns", len(fc.calls) == 3, f"实际 {len(fc.calls)}")


# ==================== 主入口 ====================

if __name__ == "__main__":
    for t in (t1_duplicate_guard, t2_max_tokens_empty, t3_max_tokens_partial,
              t4_refusal, t5_window_exceeded, t6_max_turns):
        try:
            t()
        except Exception as e:
            check(f"{t.__name__} 未捕获异常", False, f"{type(e).__name__}: {e}")

    passed = sum(1 for _, ok in _results if ok)
    print(f"\n合计：{passed} passed, {len(_results) - passed} failed")
