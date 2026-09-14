"""
查询流水线 LangGraph 编排

流程：
node_filter_extract ──(条件：已有答案则直达)──> node_answer_output ─> END
        │
        ├─> node_search_embedding ─────┐
        └─> node_search_embedding_hyde ┴─> node_rrf ─> node_rerank ─> node_answer_output
"""
from dotenv import load_dotenv
from langgraph.constants import END
from langgraph.graph import StateGraph

from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_filter_extract import node_filter_extract
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.state import QueryGraphState

load_dotenv()

builder = StateGraph(QueryGraphState)

# 注册节点
builder.add_node(node_filter_extract)
builder.add_node(node_search_embedding)
builder.add_node(node_search_embedding_hyde)
builder.add_node(node_rrf)
builder.add_node(node_rerank)
builder.add_node(node_answer_output)


def condition_fun(state: QueryGraphState):
    """
    条件边：条件解析阶段已经产出 answer（例如空问题提示）时直接跳去输出答案，
    否则并行发起两路检索。
    """
    if state.get("answer"):
        return "node_answer_output"
    return "node_search_embedding", "node_search_embedding_hyde"


builder.set_entry_point("node_filter_extract")
builder.add_conditional_edges(
    "node_filter_extract",
    condition_fun,
    {
        "node_search_embedding": "node_search_embedding",
        "node_search_embedding_hyde": "node_search_embedding_hyde",
        "node_answer_output": "node_answer_output",
    },
)

# 两路检索汇合到 RRF，再依次精排、生成答案
builder.add_edge("node_search_embedding", "node_rrf")
builder.add_edge("node_search_embedding_hyde", "node_rrf")
builder.add_edge("node_rrf", "node_rerank")
builder.add_edge("node_rerank", "node_answer_output")
builder.add_edge("node_answer_output", END)

kb_query_app = builder.compile()
