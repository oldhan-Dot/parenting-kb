"""
节点：切片检索（node_search_embedding）

用改写后的问题做 BGE-M3 稠密 + 稀疏混合检索，并用元数据过滤条件缩小范围。
如果带条件检索没有命中，会自动退化为全库检索，避免「条件太严导致召不回」。
"""
from app.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.conf.domain_config import CHUNK_OUTPUT_FIELDS
from app.conf.milvus_config import milvus_config
from app.core.logger import logger, node_log, step_log
from app.lm.embedding_utils import generate_embeddings
from app.query_process.agent.filters import build_metadata_filter_expr, describe_filters
from app.query_process.agent.state import QueryGraphState
from app.utils.milvus_hit_utils import flatten_hybrid_hits
from app.utils.task_utils import add_running_task, add_done_task

# 单路向量检索条数 / 融合后返回条数
REQ_LIMIT = 10
FINAL_LIMIT = 5
# 稠密向量权重更高（0.8），稀疏向量补充关键词匹配能力（0.2）
RANKER_WEIGHTS = (0.8, 0.2)


@step_log("step_1_vectorize")
def step_1_vectorize(rewritten_query: str):
    """把改写后的问题转成稠密 + 稀疏向量"""
    embeddings = generate_embeddings([rewritten_query])
    return embeddings["dense"][0], embeddings["sparse"][0]


@step_log("step_2_hybrid_search")
def step_2_hybrid_search(dense_vector, sparse_vector, expr):
    """执行带过滤条件的混合检索"""
    milvus_client = get_milvus_client()
    if milvus_client is None:
        logger.error("Milvus 客户端初始化失败，无法检索")
        return []

    reqs = create_hybrid_search_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=expr,
        limit=REQ_LIMIT,
    )

    results = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=reqs,
        ranker_weights=RANKER_WEIGHTS,
        norm_score=True,
        limit=FINAL_LIMIT,
        output_fields=CHUNK_OUTPUT_FIELDS,
    )

    if not results:
        return []
    return flatten_hybrid_hits(results[0])


@node_log("node_search_embedding")
def node_search_embedding(state: QueryGraphState):
    session_id = state["session_id"]
    is_stream = state["is_stream"]
    add_running_task(session_id, "node_search_embedding", is_stream)

    rewritten_query = state.get("rewritten_query") or state.get("original_query") or ""
    filters = state.get("filters") or {}

    if not rewritten_query:
        logger.warning("切片检索：问题为空，跳过检索")
        add_done_task(session_id, "node_search_embedding", is_stream)
        return {"embedding_chunks": []}

    dense_vector, sparse_vector = step_1_vectorize(rewritten_query)

    expr = build_metadata_filter_expr(filters)
    logger.info(f"切片检索条件：{describe_filters(filters)}，过滤表达式：{expr}")

    hits = step_2_hybrid_search(dense_vector, sparse_vector, expr)

    # 召回兜底：带条件查不到时，去掉条件再查一次
    if not hits and expr:
        logger.warning("带条件检索无命中，退化为全库检索")
        hits = step_2_hybrid_search(dense_vector, sparse_vector, None)

    logger.info(f"切片检索共命中 {len(hits)} 条")

    add_done_task(session_id, "node_search_embedding", is_stream)
    return {"embedding_chunks": hits}
