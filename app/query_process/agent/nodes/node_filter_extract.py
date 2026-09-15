"""
节点：检索条件解析（node_filter_extract）

职责：
1. 读取当前会话历史，结合历史做指代消解
2. 调用大模型抽取检索条件（内容类型 / 年龄段 / 问题类型 / 场景）并改写问题
3. 判断本次是否真的需要检索（拦截寒暄与无意义输入，如「你好」「1」）
4. 落库本轮用户消息，并回填历史消息缺失的检索标签
"""
import json

from langchain_core.messages import SystemMessage, HumanMessage

from app.clients.mongo_history_utils import (
    get_recent_messages,
    save_chat_message,
    update_message_filter_tags,
)
from app.conf.domain_config import METADATA_FIELDS
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.filters import describe_filters
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 空问题时的提示语
EMPTY_QUERY_ANSWER = "请输入您想咨询的育儿问题，例如「3-6岁孩子发脾气怎么办」。"

# 无需检索时（寒暄 / 无意义输入）直答用的角色设定
DIRECT_REPLY_SYSTEM = "你是一名亲切、专业的家庭教育育儿助手。"


@step_log("step_1_get_history")
def step_1_get_history(session_id: str) -> list:
    """读取当前会话最近的历史对话"""
    return get_recent_messages(session_id)


def format_history_text(history_list: list) -> str:
    """把历史对话拼成提示词里的文本（家长 / 助手 视角）"""
    lines = []
    for history in history_list or []:
        role = "家长" if history.get("role") == "user" else "助手"
        text = (history.get("text") or "").strip()
        if text:
            lines.append(f"{role}：{text}")
    return "\n".join(lines) if lines else "（无历史对话）"


@step_log("step_2_extract_filters")
def step_2_extract_filters(original_query: str, history_list: list):
    """
    调用大模型抽取检索条件、改写问题，并判断是否需要检索

    :return: (filters, rewritten_query, need_retrieval)
             异常时返回 ({}, original_query, True) —— 保守策略：宁可多检索，不要漏答
    """
    history_text = format_history_text(history_list)
    prompt = load_prompt("filter_extract", history_text=history_text, query=original_query)

    messages = [
        SystemMessage(content="你是一个专业的家庭教育育儿助手，擅长理解家长意图并提取关键信息。"),
        HumanMessage(content=prompt),
    ]

    try:
        llm = get_llm_client(json_mode=True)
        result = llm.invoke(messages).content

        # 兼容模型输出 ```json 包裹的情况
        if result.startswith("```"):
            result = result.replace("```json", "").replace("```", "")
        parsed = json.loads(result)
    except Exception as e:
        logger.error(f"检索条件抽取失败，将退化为无条件检索：{e}")
        return {}, original_query, True

    raw_filters = parsed.get("filters")
    if not isinstance(raw_filters, dict):
        raw_filters = {}

    # 只保留规范字段，并统一转成去空字符串
    filters = {field: str(raw_filters.get(field) or "").strip() for field in METADATA_FIELDS}

    rewritten_query = (parsed.get("rewritten_query") or "").strip() or original_query

    # 是否需要检索：模型漏输出该字段时默认 True（宁可多检索，不要漏答）
    need_retrieval = parsed.get("need_retrieval")
    if not isinstance(need_retrieval, bool):
        need_retrieval = True

    logger.info(f"检索条件抽取结果：{filters}，改写问题：{rewritten_query}，需要检索={need_retrieval}")
    return filters, rewritten_query, need_retrieval


@step_log("step_3_write_history")
def step_3_write_history(session_id: str, original_query: str, rewritten_query: str,
                         filters: dict, history_list: list):
    """
    落库本轮用户消息，并回填历史消息中缺失的检索标签

    回填的价值：下一轮做指代消解时，历史消息能带上「当时聊的是哪个年龄段 / 哪个问题类型」
    """
    tags = [value for value in (filters or {}).values() if value]

    save_chat_message(
        session_id=session_id,
        role="user",
        text=original_query,
        rewritten_query=rewritten_query,
        filter_tags=tags,
    )

    if tags and history_list:
        ids = [str(h["_id"]) for h in history_list if not h.get("filter_tags")]
        if ids:
            update_message_filter_tags(ids, tags)


@step_log("step_4_direct_reply")
def step_4_direct_reply(original_query: str) -> str:
    """无需检索时（寒暄 / 无意义输入）生成一句引导回复"""
    prompt = (
        f"家长对你说：{original_query}\n\n"
        "请用一句亲切简短的中文回应（不超过 50 字），并自然地引导他描述具体的育儿问题，"
        "引导时举一个例子，例如「3-6岁孩子发脾气怎么办」。"
        "不要展开讲育儿方法，不要输出小标题，不要输出【来源】。"
    )
    llm = get_llm_client()
    messages = [
        SystemMessage(content=DIRECT_REPLY_SYSTEM),
        HumanMessage(content=prompt),
    ]
    return (llm.invoke(messages).content or "").strip()


@node_log("node_filter_extract")
def node_filter_extract(state: QueryGraphState):
    session_id = state["session_id"]
    original_query = (state.get("original_query") or "").strip()
    is_stream = state["is_stream"]

    add_running_task(session_id, "node_filter_extract", is_stream)

    # 空问题：直接给出提示，由条件边跳到答案输出节点
    if not original_query:
        state["answer"] = EMPTY_QUERY_ANSWER
        add_done_task(session_id, "node_filter_extract", is_stream)
        return state

    # 步骤1：读历史
    history_list = step_1_get_history(session_id)
    # 步骤2：抽取条件 + 改写问题 + 判断是否需要检索
    filters, rewritten_query, need_retrieval = step_2_extract_filters(original_query, history_list)
    # 步骤3：落库与回填（无论是否检索都记历史，保证多轮上下文完整）
    step_3_write_history(session_id, original_query, rewritten_query, filters, history_list)

    state["filters"] = filters
    state["rewritten_query"] = rewritten_query
    state["history"] = history_list

    # 寒暄 / 无意义输入（如「你好」「1」「测试」）不做检索，直接给一句引导回复。
    # 条件边检测到 state["answer"] 有值后，会跳过检索直达 node_answer_output。
    if not need_retrieval:
        logger.info(f"判定为无需检索，跳过检索直接回复。原问题：{original_query}")
        state["answer"] = step_4_direct_reply(original_query)
        add_done_task(session_id, "node_filter_extract", is_stream)
        return state

    logger.info(f"本轮检索条件：{describe_filters(filters)}")

    add_done_task(session_id, "node_filter_extract", is_stream)
    return state
