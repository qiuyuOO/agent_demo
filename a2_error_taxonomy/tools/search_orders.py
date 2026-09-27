"""search_orders 工具实现。"""
import json
import re
from datetime import date
from typing import Any

from a2_error_taxonomy.llm_types import ToolSpec
from a2_error_taxonomy.tool_types import (Session, ToolResult,
                                          ToolValidationError)
_FILTER_KEYS = frozenset(
        {"user_id", "user_name", "state", "created_from", "created_to"})

VALID_STATES = frozenset(
    {"pending", "paid", "shipped", "signed", "closed", "refunding"})

_DATE_PATTERNS = (
    re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$"),
    re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})$"),
    re.compile(r"^(\d{4})\.(\d{1,2})\.(\d{1,2})$"),
)

_FAKE_ORDERS: list[dict[str, Any]] = [
    {"order_id": "SO-20240102-1001", "user_id": "u_10237",
     "user_name": "北京华信科技有限公司", "status": "shipped",
     "created_at": "2024-01-02", "shipped_at": "2024-01-03",
     "signed_at": None, "total_amount": 128800},
    {"order_id": "SO-20240105-8871", "user_id": "u_10237",
     "user_name": "北京华信科技有限公司", "status": "shipped",
     "created_at": "2024-01-05", "shipped_at": "2024-01-06",
     "signed_at": None, "total_amount": 45200},
    {"order_id": "SO-20240106-9002", "user_id": "u_10238",
     "user_name": "上海鼎新贸易", "status": "refunding",
     "created_at": "2024-01-06", "shipped_at": "2024-01-07",
     "signed_at": None, "total_amount": 9900},
    {"order_id": "SO-20240107-9103", "user_id": "u_10238",
     "user_name": "上海鼎新贸易", "status": "signed",
     "created_at": "2024-01-07", "shipped_at": "2024-01-08",
     "signed_at": "2024-01-10", "total_amount": 33600},

    {"order_id": "SO-20240108-9204", "user_id": "u_10239",
     "user_name": "华信物流（天津）有限公司", "status": "pending",
     "created_at": "2024-01-08", "shipped_at": None,
     "signed_at": None, "total_amount": 5600},
]


DEFAULT_LIMIT = 5
MAX_LIMIT = 10


# ==================== 工具声明（给模型看的唯一入口）====================

SEARCH_ORDERS_SPEC = ToolSpec(
    name="search_orders",
    description=(
        "查询订单列表，可按客户、订单状态、订单创建时间筛选。"
        "返回订单的概要信息（不含商品明细与物流轨迹）。\n\n"
        "使用场景：坐席询问某笔订单的状态、某个客户的订单列表、"
        "「上周发货还没签收的有哪些」、「有没有在退款的」、「1 月份的单子都调出来」。\n\n"
        "注意：本工具只能查到当前坐席负责范围内的客户订单。"
        "查询范围之外的客户会返回 PERMISSION_DENIED，此时不要重试，应向坐席说明。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": ("客户唯一标识，格式为 u_ 加数字，例如 u_10237。"
                                "必须由 find_customer 解析得到，不要自行推测或编造。"
                                "不传则查询当前坐席负责范围内的全部客户。"),
            },
            "user_name": {
                "type": "string",
                "description": ("客户名称关键词，支持部分匹配，例如「华信」。"
                                "仅在坐席提到客户名但未提供 user_id 时使用。"
                                "若匹配到多个客户，本工具会返回 MULTIPLE_CUSTOMERS 错误，"
                                "此时应改用 find_customer 让坐席确认，不要重复调用本工具。"),
            },
            "state": {
                "type": "string",
                "enum": sorted(VALID_STATES),
                "description": (
                    "按订单状态筛选：pending=待付款，paid=已付款待发货，"
                    "shipped=已发货在途，signed=已签收，"
                    "closed=已关闭（取消或退款完成），refunding=退款中。"
                    "查「还没签收的」用 shipped；查「退款中的」用 refunding"
                    "——已退款订单若按在途处理会导致重复发货。不传则返回所有状态。"),
            },
            "created_from": {
                "type": "string", "format": "date",
                "description": ("订单创建时间的起始日期，格式 YYYY-MM-DD"
                                "（如 2024-01-05），含当日，按北京时间计算。"
                                "坐席说「上周」「1 月份」时需要换算为具体日期。"),
            },
            "created_to": {
                "type": "string", "format": "date",
                "description": ("订单创建时间的结束日期，格式 YYYY-MM-DD，"
                                "含当日，按北京时间计算。必须大于或等于 created_from。"),
            },
            "limit": {
                "type": "integer", "minimum": 1, "maximum": MAX_LIMIT,
                "default": DEFAULT_LIMIT,
                "description": (f"最多返回的订单条数，默认 {DEFAULT_LIMIT}，"
                                f"上限 {MAX_LIMIT}。上限是上下文预算限制而非接口限制"
                                "——需要更多结果时，请用 created_from/created_to "
                                "缩小时间范围后分批查询，不要试图突破上限"
                                "或用相同参数重复调用。"),
            },
        },
        "required": [],
        "additionalProperties": False,
    },
)


def _coerce_date(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ToolValidationError(
            "INVALID_ARGS",
            f"{field_name} 期望字符串，实际收到 {type(value).__name__}",
            f"请用 YYYY-MM-DD 格式传入 {field_name}，例如 2024-01-05。")

    for pattern in _DATE_PATTERNS:
        m = pattern.match(value.strip())
        if m:
            try:
                y, mo, d = (int(g) for g in m.groups())
                return date(y, mo, d).isoformat()
            except ValueError:
                raise ToolValidationError(
                    "INVALID_ARGS",
                    f"{field_name}={value!r} 不是有效日期",
                    "请检查年月日是否合法，例如 2024-02-30 不存在。") from None

    raise ToolValidationError(
        "INVALID_ARGS",
        f"{field_name}={value!r} 格式无法识别",
        "请用 YYYY-MM-DD 格式（例如 2024-01-05）。"
        "不接受「1月5日」这类缺年份的写法，"
        "也不接受 5/1/2024 这类月日顺序有歧义的写法。")

def _coerce_limit(raw : str):
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
            f"上限 {MAX_LIMIT} 是上下文预算限制。需要更多结果时，"
            "请用 created_from/created_to 缩小时间范围后分批查询，"
            "不要试图突破上限。")

    return raw

def _coerce_user_id (uid:str):
    if isinstance(uid, str) and re.fullmatch(r"u_[0-9]{5}", uid):
        return uid
    else:
        raise ToolValidationError(
            "INVALID_ARGS",
            f"user_id={uid!r} 不符合格式 u_ 加 5 位数字",
            "user_id 形如 u_10237。若只有客户名，请先调用 find_customer 解析，"
            "不要自行编造或省略前缀。")

def _coerce_user_name (name:str):

    if not isinstance(name, str) or not name.strip():
        raise ToolValidationError(
            "INVALID_ARGS", f"user_name={name!r} 不是有效关键词",
            "请传客户名称的任意关键词，例如「华信」。")
    return name.strip()

def  _coerce_state (st:str):
    if st not in VALID_STATES:
        raise ToolValidationError(
            "INVALID_ARGS",
            f"state={st!r} 不是合法订单状态",
            f"合法取值为：{', '.join(sorted(VALID_STATES))}。"
            "请从这 6 个值中选择；若坐席描述的状态不在其中，"
            "请向客服确认对应的系统状态，不要猜测。")
    return st

def validate(args: dict[str, Any]) -> dict[str, Any]:
    """返回清洗后的参数；无法安全修复时抛 ToolValidationError。"""
    clean: dict[str, Any] = {}

    if args.get("limit") is not None:
        raw = args["limit"]
        clean["limit"] = _coerce_limit(raw)

    for f in ("created_from", "created_to"):
        if args.get(f) is not None:
            clean[f] = _coerce_date(args[f], f)

    if args.get("user_id") is not None:
        uid = args["user_id"]
        clean ["user_id"] = _coerce_user_id(uid)

    if args.get("user_name") is not None:
        name = args["user_name"]
        clean["user_name"] = _coerce_user_name(name)

    if args.get("state") is not None:
        st = args["state"]
        clean["state"] = _coerce_state(st)


    if "created_from" in clean and "created_to" in clean:
        if clean["created_from"] > clean["created_to"]:
            raise ToolValidationError(
                "INVALID_ARGS",
                f"created_from({clean['created_from']}) 晚于 "
                f"created_to({clean['created_to']})",
                "请确认时间范围的起止顺序后重新调用。")

    if not (_FILTER_KEYS & clean.keys()):
        raise ToolValidationError(
            "INVALID_ARGS",
            "未提供任何筛选条件，等于查询全部订单",
            "请至少指定以下之一：user_id / user_name / state / "
            "created_from / created_to。若坐席想看「我负责的客户里有没有退款的」，"
            "传 state=refunding 即可，不需要指定客户。")

    return clean

def authorize(args: dict[str, Any], session: Session) -> None:
    uid = args.get("user_id")
    if uid is not None and uid not in session.scopes:
        raise ToolValidationError(
            "PERMISSION_DENIED",
            f"客户 {uid} 不在你负责的范围内",
            "请勿重试。告知坐席该客户不在其负责范围，请其联系对应负责人或主管。",
            retryable=False)



def handler(session: Session, **clean: Any) -> ToolResult:
    # ★ 查询起点是"这个坐席能看的全部客户"。
    #   不传 user_id 时范围自动收敛到 scope ——
    #   这让「我负责的客户里有没有退款的」变成合法且有边界的查询。
    rows = [o for o in _FAKE_ORDERS if o["user_id"] in session.scopes]

    if (uid := clean.get("user_id")):
        rows = [o for o in rows if o["user_id"] == uid]

    name = clean.get("user_name")
    if name:
        rows = [o for o in rows if name in o["user_name"]]

    if (st := clean.get("state")):
        rows = [o for o in rows if o["status"] == st]
    if (f := clean.get("created_from")):
        rows = [o for o in rows if o["created_at"] >= f]
    if (t := clean.get("created_to")):
        rows = [o for o in rows if o["created_at"] <= t]

    # ★ 一个关键词可能命中多个客户（子串匹配）
    #   数的是"不同客户数"，不是"订单数" —— 同一客户 5 单仍然是一个客户
    if name:
        name_by_id: dict[str, str] = {}
        for o in rows:
            name_by_id.setdefault(o["user_id"], o["user_name"])

        if len(name_by_id) > 1:
            return ToolResult(
                ok=False,
                error_code="MULTIPLE_CUSTOMERS",
                content=json.dumps(
                    {"candidates": [{"user_id": uid, "user_name": uname}
                                    for uid, uname in sorted(name_by_id.items())]},
                    ensure_ascii=False),
                hint=("关键词匹配到多个客户。不要自行选择，"
                      "请列出候选请坐席确认，或改用 find_customer。"
                      "不要用相同参数重复调用本工具。"),
                retryable=False)

    # 按创建时间倒序 —— 最近的在前面
    # ★ 是原地排序，不要写 rows = rows.sort(...)（那会把 rows 变成 None）
    rows.sort(key=lambda o: o["created_at"], reverse=True)

    limit = clean.get("limit", DEFAULT_LIMIT)
    page = rows[:limit]

    # 摘要统计的是全部结果（rows），不是 page —— 这是它有价值的原因
    by_status: dict[str, int] = {}
    for o in rows:
        by_status[o["status"]] = by_status.get(o["status"], 0) + 1

    payload: dict[str, Any] = {
        "total": len(rows),                 # 命中总数
        "returned": len(page),              # 实际返回条数
        "by_status": by_status,             # 分布摘要
        "oldest": rows[-1]["created_at"] if rows else None,
        "newest": rows[0]["created_at"] if rows else None,
        "orders": [
            {
                "order_id": o["order_id"],
                "status": o["status"],
                "user_name": o["user_name"],
                "created_at": o["created_at"],
                "shipped_at": o["shipped_at"],
                "signed_at": o["signed_at"],
                # 接口给的是"分"，模型不知道 → 转成元并改字段名
                "total_amount_yuan": round(o["total_amount"] / 100, 2),
            }
            for o in page
        ],
        "scope_note": f"查询范围：当前坐席负责的 {len(session.scopes)} 个客户",
    }

    if len(rows) > len(page):
        payload["hint"] = ("结果超出返回上限，未显示全部。"
                           "如需完整列表，请用 created_from/created_to "
                           "缩小时间范围后分批查询；若只需了解分布，见 by_status。")

    return ToolResult(ok=True, content=json.dumps(payload, ensure_ascii=False))

