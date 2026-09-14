"""
查询流程的状态定义

与导入流程的区别：查询侧不再做「商品名对齐」，而是先解析出
年龄段 / 问题类型 / 场景等检索条件，再带着条件去做混合检索。
"""
import copy
from typing import TypedDict, List, Dict

from typing_extensions import NotRequired


class QueryGraphState(TypedDict):
    """查询流程状态：所有节点产生和消费的数据字段"""

    session_id: str                 # 会话唯一标识
    original_query: str             # 用户原始问题

    # --- 条件解析结果 ---
    rewritten_query: str            # 改写后的问题（指代消解后，可脱离上下文）
    filters: Dict[str, str]         # 检索条件：content_type / age_range / problem_type / scene

    # --- 检索过程中的中间数据 ---
    embedding_chunks: list          # 常规混合检索结果
    hyde_embedding_chunks: list     # HyDE 假设性文档检索结果
    hyde_doc: NotRequired[str]      # HyDE 生成的假设性文档

    # --- 排序过程中的数据 ---
    rrf_chunks: list                # RRF 融合后的切片
    reranked_docs: list             # 精排后的最终 Top-K 切片

    # --- 生成过程中的数据 ---
    prompt: str                     # 组装好的 Prompt
    answer: str                     # 最终答案

    # --- 辅助信息 ---
    history: list                   # 历史对话记录
    is_stream: bool                 # 是否流式输出


# 默认状态（全部为空）
query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "rewritten_query": "",
    "filters": {},
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "hyde_doc": "",
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "history": [],
    "is_stream": False,
}


def create_query_default_state(**overrides) -> QueryGraphState:
    """创建查询流程默认状态，支持字段覆盖"""
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


def get_query_default_state() -> QueryGraphState:
    """返回一个干净的查询状态实例"""
    return copy.deepcopy(query_graph_default_state)


def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    """深拷贝状态并覆盖字段，不污染原数据"""
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state
