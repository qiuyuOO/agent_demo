import json
import os
import logging
from dotenv import load_dotenv

from a2_error_taxonomy.context import estimate_tokens, compress
from a2_error_taxonomy.llm_client import LLMClient
from a2_error_taxonomy.llm_types import Message, user_message, assistant_message, StopReason, tool_message
from a2_error_taxonomy.tool_types import Session, ToolResult
from a2_error_taxonomy.tools.tool_registry import build_tools, execute
from a2_error_taxonomy.error_type import ContextLengthExceeded, DomainError

load_dotenv()

log = logging.getLogger("loop")


MAX_COMPRESS = 2

def _one_turn(client, messages, tools, session, seen, turn) -> str | None:
    """跑一轮：发请求 → 回填 → 终止判断 → 执行工具。

    ★ 返回值语义：
        返回 str  —— 命中终态，这就是最终答复
        返回 None —— 本轮未终止（工具已执行，或模型要求暂停），调用方继续下一轮
    """
    resp = client.chat(messages, tools=tools)
    messages.append(assistant_message(resp))

    log.info("turn=%d stop=%s tool_calls=%d in=%d out=%d cache_read=%d cache_new=%d ms=%.0f",
             turn, resp.stop_reason, len(resp.tool_calls),
             resp.input_tokens, resp.output_tokens,
             resp.cache_read_tokens, resp.cache_creation_tokens, resp.total_ms)

    # ---- 终态：模型不再需要我做什么 ----
    if resp.stop_reason == StopReason.END_TURN:
        return resp.text

    if resp.stop_reason == StopReason.STOP_SEQUENCE:
        return resp.text

    if resp.stop_reason == StopReason.MAX_TOKENS:
        # ★ 空输出和"截断了但有一些"要分开 —— 前者重试/加大配额都可能无效
        if not resp.text.strip():
            log.error("turn=%d max_tokens 撞限且无可见输出（output_tokens=%d）",
                      turn, resp.output_tokens)
            return "（回答超出长度限制且未能产出内容，请缩小问题范围后重试）"
        log.warning("turn=%d 回答被截断（output_tokens=%d）", turn, resp.output_tokens)
        return resp.text + "\n\n（注：回答因长度限制被截断）"

    if resp.stop_reason == StopReason.REFUSAL:
        # ★ 不重试 —— 重试会得到同样的拒绝
        log.warning("turn=%d 模型拒绝回答", turn)
        return "（该问题我无法回答，已记录。请转人工或换个问法）"

    if resp.stop_reason == StopReason.MODEL_CONTEXT_WINDOW_EXCEEDED:
        # ★ 和 413 同一个出口：等一等没用，必须先压缩上下文（A5 接住）
        #   用专属域异常而不是 RuntimeError —— 兜底才分得清"信号"和"意外"
        log.error("turn=%d 上下文窗口超限，需要压缩：in=%d out=%d",
                  turn, resp.input_tokens, resp.output_tokens)
        raise ContextLengthExceeded()

    if resp.stop_reason == StopReason.UNKNOWN:
        # ★ 未知 ≠ 失败。有输出就当成功用，但必须记录原始值供排查
        log.error("turn=%d 未知 stop_reason，原始值见 adapter 的 warning 日志 "
                  "in=%d out=%d text_len=%d",
                  turn, resp.input_tokens, resp.output_tokens, len(resp.text))
        if resp.text.strip():
            return resp.text
        return "（模型返回了未识别的终止原因且无输出，请转人工）"

    if resp.stop_reason == StopReason.PAUSE_TURN:
        # ★ 不实现 resume：DeepSeek 兼容层大概率不返回此值
        log.error("turn=%d 收到 pause_turn —— 当前未实现 resume 协议，按终态处理", turn)
        return resp.text or "（模型暂停了长任务但本环境不支持继续，请转人工）"

    if resp.stop_reason == StopReason.TOOL_USE:
        for call in resp.tool_calls:
            key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
            seen[key] = seen.get(key, 0) + 1

            if seen[key] >= 2:
                respond = ToolResult(
                    ok=False, error_code="DUPLICATE_CALL",
                    content=(f"你刚刚已经用完全相同的参数调用过 {call.name}，"
                             f"结果不会变化。"),
                    hint="不要用相同参数重复调用。若已有结果足够请直接作答，"
                         "否则请换参数（例如换状态、换时间范围或换客户）。",
                    retryable=False)
            else:
                respond = execute(call.name, call.arguments, session)

            messages.append(tool_message(call.id, respond.model_text,
                                         is_error=not respond.ok))
        return None          # ★ 工具结果已回填，继续下一轮

    # 兜底：枚举里出现了新值（StopReason 是你自己的枚举，理论上不该走到这）
    log.error("turn=%d 未处理的 stop_reason=%s", turn, resp.stop_reason)
    return resp.text or "（未识别的终止状态，请转人工）"


def run(session: Session, messages: list[Message], client: LLMClient,
        max_turns: int = 8) -> str:
    seen: dict[str, int] = {}
    tools = build_tools(session.role)
    compress_count = 0
    for turn in range(1, max_turns + 1):
        try:
            result = _one_turn(client, messages, tools, session, seen, turn)

        except ContextLengthExceeded as e:
            log.warning("turn=%d 上下文不足，尝试压缩：%s", turn, e)

            if compress_count >= MAX_COMPRESS:
                log.error("turn=%d 已压缩 %d 次仍不足，放弃", turn, compress_count)
                raise

            before = estimate_tokens(messages)
            compressed = compress(messages, client=client)
            after = estimate_tokens(compressed)

            # ★ 按 token 判断，不按条数 ——
            #   中间段是单条消息时压缩是"1 换 1"，条数不变但 token 可能下降。
            if after >= before:
                log.error("turn=%d 压缩无效（%d -> %d token），放弃",
                          turn, before, after)
                raise

            messages[:] = compressed  # ★ 原地替换，保持调用方的引用
            client.reset_usage_baseline()  # ★ 清掉压缩前的实测基线
            compress_count += 1
            log.info("turn=%d 压缩完成：%d -> %d token（第 %d 次），重试本轮",
                     turn, before, after, compress_count)
            continue
        except DomainError as e:
            # ★ log.exception 会带上堆栈和 __cause__ 链 ——
            #   这是唯一能看到 anthropic 原始错误的地方
            log.exception("turn=%d 模型调用失败：%s", turn, type(e).__name__)
            if isinstance(e, ContextLengthExceeded):
                raise                   # 信号 —— 交给 A5 压缩，不变成用户话术
            return e.content            # 其余：统一回一句用户能懂的话
        if result is not None:
            return result

    return "（已达单次处理轮次上限，请缩小问题范围或转人工）"