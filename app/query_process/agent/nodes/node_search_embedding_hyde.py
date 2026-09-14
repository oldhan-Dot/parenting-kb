"""
节点：HyDE 假设性文档检索（node_search_embedding_hyde）

思路：先让大模型「凭空写一段可能的答案」，再用这段答案去检索。
因为答案和知识库里的文档在语义空间上更接近，可以明显提升召回质量，
特别适合家长口语化提问（"我家娃躺地上哭"）与知识库书面表达不匹配的场景。
"""
from langchain_core.messages import HumanMessage

from app.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.conf.domain_config import CHUNK_OUTPUT_FIELDS
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.filters import build_metadata_filter_expr
from app.query_process.agent.state import QueryGraphState
from app.utils.milvus_hit_utils import flatten_hybrid_hits
from app.utils.task_utils import add_running_task, add_done_task

HYDE_REQ_LIMIT = 10
HYDE_FINAL_LIMIT = 5
RANKER_WEIGHTS = (0.8, 0.2)


@step_log("step_1_create_hyde_doc")
def step_1_create_hyde_doc(rewritten_query: str) -> str:
    """生成假设性文档（一段可能的回答范文）"""
    prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
    llm = get_llm_client()
    response = llm.invoke([HumanMessage(content=prompt)])
    return (response.content or "").strip()


@step_log("step_2_search_embedding_hyde")
def step_2_search_embedding_hyde(rewritten_query: str, hyde_doc: str, expr=None):
    """用「问题 + 假设性文档」拼接后的文本做混合检索"""
    text = f"{rewritten_query} {hyde_doc}"

    embeddings = generate_embeddings([text])
    dense_vector = embeddings["dense"][0]
    sparse_vector = embeddings["sparse"][0]

    milvus_client = get_milvus_client()
    if milvus_client is None:
        logger.error("Milvus 客户端初始化失败，无法检索")
        return []

    reqs = create_hybrid_search_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=expr,
        limit=HYDE_REQ_LIMIT,
    )

    results = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=reqs,
        ranker_weights=RANKER_WEIGHTS,
        norm_score=True,
        limit=HYDE_FINAL_LIMIT,
        output_fields=CHUNK_OUTPUT_FIELDS,
    )

    if not results:
        return []
    return flatten_hybrid_hits(results[0])


@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state: QueryGraphState):
    session_id = state["session_id"]
    is_stream = state["is_stream"]
    add_running_task(session_id, "node_search_embedding_hyde", is_stream)

    rewritten_query = state.get("rewritten_query") or state.get("original_query") or ""
    filters = state.get("filters") or {}

    if not rewritten_query:
        logger.warning("HyDE 检索：问题为空，跳过检索")
        add_done_task(session_id, "node_search_embedding_hyde", is_stream)
        return {"hyde_embedding_chunks": []}

    # 步骤1：生成假设性文档（失败则退化为只用原问题检索）
    hyde_doc = ""
    try:
        hyde_doc = step_1_create_hyde_doc(rewritten_query)
    except Exception as e:
        logger.error(f"生成假设性文档失败，退化为普通检索：{e}")

    # 步骤2：混合检索
    expr = build_metadata_filter_expr(filters)
    try:
        hits = step_2_search_embedding_hyde(rewritten_query, hyde_doc, expr)
        if not hits and expr:
            logger.warning("HyDE 带条件检索无命中，退化为全库检索")
            hits = step_2_search_embedding_hyde(rewritten_query, hyde_doc, None)
    except Exception as e:
        logger.error(f"HyDE 检索失败：{e}")
        hits = []

    logger.info(f"HyDE 检索共命中 {len(hits)} 条")

    add_done_task(session_id, "node_search_embedding_hyde", is_stream)
    return {"hyde_embedding_chunks": hits, "hyde_doc": hyde_doc}
