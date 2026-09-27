"""工具执行层的契约。

★ 这一层不允许出现任何业务语义。
  "客户"、"订单"、"坐席" 这些词只能出现在具体工具里。
  通用层用 scopes 这种不解释内容的容器承载权限信息。

与 llm_types 的分工：
  llm_types  —— 怎么跟模型通信（Message / Part / ToolSpec / LLMResponse）
  tool_types —— 怎么执行工具（Session / ToolResult / Tool / ToolValidationError）
两者正交：可以只用其中一个。
"""
from collections.abc import Callable
from dataclasses import dataclass

from a2_error_taxonomy.llm_types import ToolSpec


@dataclass(frozen=True)
class Session:
    """调用者身份。

    ★ scopes 是通用容器，本层不解释其含义：
      客服场景 = 负责的客户 ID 集合
      知识库场景 = 可访问的部门 / 密级
      运维场景 = 可操作的主机分组
    由各工具的 authorize() 决定怎么用。
    """
    operator_id: str
    role: str
    scopes: frozenset[str] = frozenset()


# tool_types.py
@dataclass
class ToolResult:
    ok: bool
    content: str = ""
    error_code: str | None = None
    hint: str | None = None
    retryable: bool = False

    @property
    def model_text(self) -> str:
        """给模型看的文本。★ hint 是指令，必须一起给模型 ——
        否则像 DUPLICATE_CALL 这种 content 为空的结果，
        模型只会收到一个空洞的失败然后原样重试。"""
        if self.content and self.hint:
            return f"{self.content}\n\n[HINT] {self.hint}"
        return self.content or self.hint or self.error_code or "（工具未返回内容）"


class ToolValidationError(Exception):
    def __init__(self, error_code: str, detail: str, hint: str,
                 retryable: bool = False):
        super().__init__(detail)
        self.error_code = error_code
        self.detail = detail
        self.hint = hint
        self.retryable = retryable


@dataclass
class Tool:
    spec: ToolSpec
    handler: Callable[..., ToolResult]
    required_scopes: frozenset[str] = frozenset()   # 空集 = 无需特定权限
    validate: Callable[[dict], dict] | None = None
    authorize: Callable[[dict, Session], None] | None = None