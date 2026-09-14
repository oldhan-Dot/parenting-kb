"""
节点：结果精排（node_rerank）

用 bge-reranker-large 交叉编码器对「问题 - 切片」逐对打分。
向量检索是「近似召回」，精排是「精确打分」，两者配合能显著提升 Top 结果的相关性。

注意：交叉编码器计算量较大，所以只对 RRF 融合后的前 N 条候选做精排。
"""
from app.core.logger import logger, node_log
from app.lm.reranker_utils import get_reranker_model
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 进入精排的候选数量上限（控制耗时）
RERANK_CANDIDATE_LIMIT = 20
# 精排后最终交给大模型的切片数量
TOP_K = 5


@node_log("node_rerank")
def node_rerank(state: QueryGraphState):
    session_id = state["session_id"]
    is_stream = state["is_stream"]
    add_running_task(session_id, "node_rerank", is_stream)

    docs = state.get("rrf_chunks") or []
    query = state.get("rewritten_query") or state.get("original_query") or ""

    # 没有候选或没有问题时，直接取前 TOP_K 条
    if not docs or not query:
        logger.warning("精排输入为空，跳过精排")
        add_done_task(session_id, "node_rerank", is_stream)
        return {"reranked_docs": docs[:TOP_K]}

    candidates = docs[:RERANK_CANDIDATE_LIMIT]
    pairs = [[query, doc.get("content", "")] for doc in candidates]

    try:
        model = get_reranker_model()
        scores = model.compute_score(pairs, normalize=True)
    except Exception as e:
        # 精排失败时降级为 RRF 顺序，保证流程可用
        logger.error(f"精排失败，降级使用 RRF 排序结果：{e}")
        add_done_task(session_id, "node_rerank", is_stream)
        return {"reranked_docs": candidates[:TOP_K]}

    # 只有一对时 compute_score 返回单个数值，统一成列表
    if isinstance(scores, (int, float)):
        scores = [float(scores)]

    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:TOP_K]

    reranked_docs = []
    for doc, score in ranked:
        item = dict(doc)
        item["rerank_score"] = float(score)
        reranked_docs.append(item)

    logger.info(f"精排完成，候选 {len(candidates)} 条 → 取 Top{len(reranked_docs)}")
    for idx, doc in enumerate(reranked_docs, start=1):
        logger.info(f"  第{idx}名 score={doc['rerank_score']:.4f} 来源={doc.get('file_title')} 标题={doc.get('title')}")

    add_done_task(session_id, "node_rerank", is_stream)
    return {"reranked_docs": reranked_docs}
