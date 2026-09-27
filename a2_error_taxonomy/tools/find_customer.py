"""find_customer 工具实现。

职责：把坐席口述的客户名（可能是简称、对接人称呼）解析成 user_id。
它是 search_orders 的前置工具 —— 后者的 user_id 参数要求"必须由本工具解析得到"。
"""
import json
from typing import Any

from a2_error_taxonomy.llm_types import ToolSpec
from a2_error_taxonomy.tool_types import (Session, ToolResult,
                                          ToolValidationError)

# 关键词在这些字段里做子串匹配：全称 / 简称 / 对接人
_MATCH_FIELDS = ("user_name", "short_name", "contact")

DEFAULT_LIMIT = 5
MAX_LIMIT = 10

# 和 search_orders 的 _FAKE_ORDERS 同源：同一个客户必须有一致的 user_id / user_name
_FAKE_CUSTOMERS: list[dict[str, Any]] = [
    {"user_id": "u_10237", "user_name": "北京华信科技有限公司",
     "short_name": "华信", "contact": "林晓（小林）",
     "last_order_at": "2024-01-05", "open_orders": 2},
    {"user_id": "u_10238", "user_name": "上海鼎新贸易",
     "short_name": "鼎新", "contact": "王磊",
     "last_order_at": "2024-01-07", "open_orders": 2},
    {"user_id": "u_10239", "user_name": "华信物流（天津）有限公司",
     "short_name": "华信天津", "contact": "周敏",
     "last_order_at": "2024-01-08", "open_orders": 1},
]


# ==================== 工具声明 ====================

FIND_CUSTOMER_SPEC = ToolSpec(
    name="find_customer",
    description=(
        "按名称关键词查找客户，返回候选列表及其 user_id。\n\n"
        "使用场景：坐席提到客户名（如「小林那家公司」「华信」）但未给出 user_id 时，"
        "先用本工具解析。search_orders 的 user_id 参数必须来自本工具，不要自行编造。\n\n"
        "重要：返回的是候选列表，不是确定结果。若 ambiguous 为 true，"
        "必须把候选念给坐席并请他确认，不要自行选择第一个。"
        "若只有一个候选，仍应说明「找到 X，确认是这家吗」再继续。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": ("客户名称的任意关键词，支持部分匹配。"
                                "坐席给的别名、简称、对接人称呼都可以直接传"
                                "（如「小林」「华信」）—— 本工具会同时匹配"
                                "客户全称、简称和对接人备注。"),
            },
            "limit": {
                "type": "integer", "minimum": 1, "maximum": MAX_LIMIT,
                "default": DEFAULT_LIMIT,
                "description": f"最多返回的候选客户数，默认 {DEFAULT_LIMIT}。",
            },
        },
        "required": ["keyword"],
        "additionalProperties": False,
    },
)


# ==================== 校验 ====================

def _coerce_keyword(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ToolValidationError(
            "INVALID_ARGS",
            f"keyword={raw!r} 不是有效关键词",
            "请传客户名称的任意关键词，例如「华信」或「小林」。")
    return raw.strip()


def _coerce_limit(raw: Any) -> int:
    if isinstance(raw, bool):
        raise ToolValidationError(
            "INVALID_ARGS", f"limit 期望整数，实际收到布尔值 {raw!r}",
            f"请传 1~{MAX_LIMIT} 之间的整数。")
    if isinstance(raw, str):
        try:
            raw = int(raw.strip())
        except ValueError:
            raise ToolValidationError(
                "INVALID_ARGS", f"limit={raw!r} 无法解析为整数",
                f"请传 1~{MAX_LIMIT} 之间的整数。") from None
    if not isinstance(raw, int):
        raise ToolValidationError(
            "INVALID_ARGS", f"limit={raw!r} 类型错误",
            f"请传 1~{MAX_LIMIT} 之间的整数。")
    if not (1 <= raw <= MAX_LIMIT):
        raise ToolValidationError(
            "INVALID_ARGS", f"limit={raw} 超出允许范围 1~{MAX_LIMIT}",
            f"上限 {MAX_LIMIT}：候选客户太多时坐席无法快速确认，"
            "请让坐席给出更具体的名称。")
    return raw


def validate(args: dict[str, Any]) -> dict[str, Any]:
    """返回清洗后的参数；无法安全修复时抛 ToolValidationError。"""
    clean: dict[str, Any] = {}

    if args.get("keyword") is not None:
        clean["keyword"] = _coerce_keyword(args["keyword"])

    if args.get("limit") is not None:
        clean["limit"] = _coerce_limit(args["limit"])

    # keyword 必填 —— schema 里写了 required，但那只是给模型的约束
    if "keyword" not in clean:
        raise ToolValidationError(
            "INVALID_ARGS",
            "缺少必填参数 keyword",
            "请传客户名称的任意关键词，例如「华信」。"
            "不要用空关键词查询 —— 那会返回全部客户。")

    return clean


# ==================== 鉴权 ====================
# 本工具没有数据级鉴权：范围收敛在 handler 里做（row 必须属于 session.scopes）。
# 所以注册时 authorize=None。


# ==================== 执行 ====================

def handler(session: Session, **clean: Any) -> ToolResult:
    keyword = clean["keyword"]
    limit = clean.get("limit", DEFAULT_LIMIT)

    # ★ 查询起点是"这个坐席能看的客户"，模型给的 keyword 只是在它之上加筛选
    rows = [c for c in _FAKE_CUSTOMERS if c["user_id"] in session.scopes]

    matched = [
        c for c in rows
        if any(keyword in (c[f] or "") for f in _MATCH_FIELDS)
    ]
    matched = matched[:limit]

    payload: dict[str, Any] = {
        "matches": [
            {
                "user_id": c["user_id"],
                "user_name": c["user_name"],
                # ★ match_hint：坐席说的是"小林"，只给全称他可能认不出，要带上能对上他话的线索
                "match_hint": c["short_name"],
                "contact": c["contact"],
                "last_order_at": c["last_order_at"],
                "open_orders": c["open_orders"],
            }
            for c in matched
        ],
        "total": len(matched),
        # ★ ambiguous 显式标出来 —— 否则模型会把第一个当确定结果直接用
        "ambiguous": len(matched) > 1,
        # ★ 说明结果被权限裁剪过，否则坐席会以为"全公司只有这两家华信"
        "scope_note": f"仅显示当前坐席负责范围内的 {len(session.scopes)} 个客户",
    }

    if payload["ambiguous"]:
        payload["hint"] = ("匹配到多个客户。不要自行选择，"
                           "请把候选念给坐席确认后再用 user_id 查询。")
    elif not matched:
        payload["hint"] = ("该关键词在当前坐席负责范围内没有匹配到客户。"
                           "不要重试同一关键词；可请坐席提供更完整的客户名称，"
                           "或说明该客户可能不在其负责范围内。")

    return ToolResult(ok=True, content=json.dumps(payload, ensure_ascii=False))