"""
节点：多路结果融合（node_rrf）

把「常规向量检索」和「HyDE 检索」两路结果用 RRF（Reciprocal Rank Fusion）融合。

RRF 的核心思想：只看「排名」不看「绝对分数」，因此可以避免
两路检索分数分布不同导致无法比较的问题。公式：
    score(d) = Σ 1 / (k + rank_i(d))
k 一般取 60，作用是压平头部排名差距，让多个来源都靠前的文档胜出。
"""
from app.core.logger import logger, node_log
from app.query_process.agent.state import QueryGraphState
from app.utils.milvus_hit_utils import dedupe_by_chunk_id
from app.utils.task_utils import add_running_task, add_done_task

# RRF 平滑常数
RRF_K = 60


def fuse_by_rrf(chunk_lists):
    """
    对多路检索结果执行 RRF 融合

    :param chunk_lists: [[路1命中...], [路2命中...]]
    :return: 融合并按分数降序排列的切片列表（带 rrf_score 字段）
    """
    scores = {}
    store = {}

    for hits in chunk_lists:
        for rank, item in enumerate(hits or [], start=1):
            chunk_id = item.get("chunk_id")
            if chunk_id is None:
                continue

            # 同一文档在多路中出现，分数累加，排名越靠前贡献越大
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

            # 保留首次出现的切片内容
            if chunk_id not in store:
                store[chunk_id] = item

    fused = []
    for chunk_id, item in store.items():
        merged = dict(item)
        merged["rrf_score"] = round(scores[chunk_id], 6)
        fused.append(merged)

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


@node_log("node_rrf")
def node_rrf(state: QueryGraphState):
    session_id = state["session_id"]
    is_stream = state["is_stream"]
    add_running_task(session_id, "node_rrf", is_stream)

    embedding_chunks = dedupe_by_chunk_id(state.get("embedding_chunks") or [])
    hyde_chunks = dedupe_by_chunk_id(state.get("hyde_embedding_chunks") or [])

    fused = fuse_by_rrf([embedding_chunks, hyde_chunks])

    logger.info(
        f"RRF 融合完成：常规检索 {len(embedding_chunks)} 条 + HyDE 检索 {len(hyde_chunks)} 条 "
        f"→ 融合后 {len(fused)} 条"
    )

    add_done_task(session_id, "node_rrf", is_stream)
    return {"rrf_chunks": fused}
