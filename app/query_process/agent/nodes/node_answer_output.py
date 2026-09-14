"""
节点：答案生成（node_answer_output）

职责：
1. 把精排后的切片组装成带来源信息的上下文
2. 按 answer_out 提示词生成「详细解释 / 方法建议 / 沟通话术 / 注意事项」四段式回答
3. 流式模式下逐字推送到 SSE 队列，非流式模式写入任务结果
4. 把助手回复落库到会话历史
"""
from langchain_core.messages import SystemMessage, HumanMessage

from app.clients.mongo_history_utils import save_chat_message
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.nodes.node_filter_extract import format_history_text
from app.query_process.agent.state import QueryGraphState
from app.utils.sse_utils import SSEEvent, push_to_session
from app.utils.task_utils import add_running_task, add_done_task, set_task_result

SYSTEM_PROMPT = "你是一名专业、耐心的家庭教育育儿助手，回答务必基于给定的参考内容，不编造事实。"

# 检索结果为空时的兜底回答
NO_RESULT_ANSWER = (
    "抱歉，知识库中暂未收录与您问题相关的内容。\n"
    "您可以补充一下孩子的年龄（0-3岁 / 3-6岁 / 6-12岁 / 12+岁）和具体场景，我再帮您查一次。"
)


@step_log("step_1_build_context")
def step_1_build_context(reranked_docs: list) -> str:
    """把切片组装成带来源与元数据的上下文，便于大模型引用与溯源"""
    parts = []
    for idx, doc in enumerate(reranked_docs, start=1):
        meta_bits = [doc.get("content_type", ""), doc.get("age_range", ""), doc.get("problem_type", "")]
        meta = " / ".join([bit for bit in meta_bits if bit])

        parts.append(
            f"【片段{idx}】来源文件：{doc.get('file_title', '')}（{meta}）\n"
            f"章节：{doc.get('title', '')}\n"
            f"内容：{doc.get('content', '')}"
        )
    return "\n\n".join(parts)


@step_log("step_2_build_prompt")
def step_2_build_prompt(state: QueryGraphState, context: str) -> str:
    """组装最终提示词"""
    question = state.get("rewritten_query") or state.get("original_query") or ""
    history = format_history_text(state.get("history") or [])

    return load_prompt(
        "answer_out",
        context=context,
        history=history,
        question=question,
    )


def _push_final(session_id: str, answer: str, is_stream: bool):
    """流式模式下推送最终答案并通知前端关闭连接"""
    if not is_stream:
        return
    push_to_session(session_id, SSEEvent.FINAL, {"answer": answer})
    push_to_session(session_id, SSEEvent.CLOSE, {})


def _save_assistant_message(session_id: str, answer: str, filters: dict):
    """把助手回复落库"""
    tags = [value for value in (filters or {}).values() if value]
    try:
        save_chat_message(session_id=session_id, role="assistant", text=answer, filter_tags=tags)
    except Exception as e:
        logger.error(f"保存助手回复失败：{e}")


def _generate_answer(prompt: str, session_id: str, is_stream: bool) -> str:
    """
    调用大模型生成答案

    流式：逐字推送到 SSE，同时累积完整答案
    非流式：一次性拿到完整答案
    """
    llm = get_llm_client()
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]

    if not is_stream:
        response = llm.invoke(messages)
        return response.content or ""

    full_answer = ""
    try:
        for chunk in llm.stream(messages):
            piece = chunk.content or ""
            if piece:
                full_answer += piece
                push_to_session(session_id, SSEEvent.DELTA, {"delta": piece})
    except Exception as e:
        logger.error(f"流式生成过程中断，将降级为一次性生成：{e}")

    # 流式完全没有产出内容时降级
    if not full_answer:
        response = llm.invoke(messages)
        full_answer = response.content or ""

    return full_answer


@node_log("node_answer_output")
def node_answer_output(state: QueryGraphState):
    session_id = state["session_id"]
    is_stream = state["is_stream"]
    add_running_task(session_id, "node_answer_output", is_stream)

    # 情况一：上游已经给出了答案（例如空问题提示），直接输出
    if state.get("answer"):
        _push_final(session_id, state["answer"], is_stream)
        add_done_task(session_id, "node_answer_output", is_stream)
        return state

    reranked_docs = state.get("reranked_docs") or []

    # 情况二：检索结果为空，给出兜底回答
    if not reranked_docs:
        logger.warning("检索结果为空，返回兜底回答")
        state["answer"] = NO_RESULT_ANSWER
        set_task_result(session_id, "answer", NO_RESULT_ANSWER)
        _save_assistant_message(session_id, NO_RESULT_ANSWER, state.get("filters"))
        _push_final(session_id, NO_RESULT_ANSWER, is_stream)
        add_done_task(session_id, "node_answer_output", is_stream)
        return state

    # 情况三：正常生成
    context = step_1_build_context(reranked_docs)
    prompt = step_2_build_prompt(state, context)
    state["prompt"] = prompt

    answer = _generate_answer(prompt, session_id, is_stream)

    state["answer"] = answer
    set_task_result(session_id, "answer", answer)
    _save_assistant_message(session_id, answer, state.get("filters"))
    _push_final(session_id, answer, is_stream)

    logger.info(f"答案生成完成，长度 {len(answer)} 字")
    add_done_task(session_id, "node_answer_output", is_stream)
    return state
