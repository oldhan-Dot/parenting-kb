"""
节点：结果精排（node_rerank）

用 bge-reranker-large 交叉编码器对「问题 - 切片」逐对打分。
向量检索是「近似召回」，精排是「精确打分」，两者配合能显著提升 Top 结果的相关性。

★ 相关性阈值：
交叉编码器给出的是「问题与切片是否真的相关」的判断，所以这里加一道**阈值过滤**，
分数过低的切片直接丢弃。全部被丢弃时，下游 node_answer_output 会自动返回
「知识库中暂未收录相关内容」，从而避免"问什么都强行凑一篇答案"（例如"你好"）。

注意：交叉编码器计算量较大，所以只对 RRF 融合后的前 N 条候选做精排。
"""
import os

from dotenv import load_dotenv

from app.core.logger import logger, node_log
from app.lm.reranker_utils import get_reranker_model
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task

load_dotenv()

# 进入精排的候选数量上限（控制耗时）
RERANK_CANDIDATE_LIMIT = 20
# 精排后最终交给大模型的切片数量
TOP_K = 5

# 精排分数阈值（归一化后 0~1）：低于此值视为与问题无关，直接丢弃
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.3"))
# 精排不可用时（模型加载失败 / 内存不足）的兜底阈值：用 Milvus 混合检索的归一化分数过滤
MIN_DISTANCE_SCORE = float(os.getenv("MIN_DISTANCE_SCORE", "0.3"))


def _log_score_distribution(docs: list, score_key: str, tag: str):
    """把候选分数打成一行，方便观察分布、标定阈值"""
    scores = [float(doc.get(score_key) or 0.0) for doc in docs]
    if scores:
        logger.info(f"[{tag}] 分数分布：" + " | ".join(f"{s:.3f}" for s in scores))


def _filter_low_score(docs: list, score_key: str, threshold: float, tag: str) -> list:
    """按分数阈值过滤低相关切片（保留项与丢弃项都打日志，便于调阈值）"""
    kept, dropped = [], 0
    for doc in docs:
        score = float(doc.get(score_key) or 0.0)
        if score < threshold:
            dropped += 1
            logger.info(
                f"  [{tag}] 丢弃 {score_key}={score:.4f} "
                f"来源={doc.get('file_title')} 标题={doc.get('title')}"
            )
            continue
        kept.append(doc)

    if dropped:
        logger.warning(f"[{tag}] 过滤：丢弃 {dropped} 条低相关切片（阈值 {threshold}），保留 {len(kept)} 条")
    return kept


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
        # 精排失败（模型加载失败 / 内存不足）时降级：
        # 改用 Milvus 混合检索的归一化分数过滤，至少把明显无关的切片挡掉
        logger.error(f"精排失败，降级为 Milvus 分数过滤：{e}")
        _log_score_distribution(candidates, "distance", "降级")
        passed = _filter_low_score(candidates, "distance", MIN_DISTANCE_SCORE, "降级")
        if not passed:
            logger.warning("降级过滤后无相关切片，将由答案节点返回「知识库暂未收录」")
        add_done_task(session_id, "node_rerank", is_stream)
        return {"reranked_docs": passed[:TOP_K]}

    # 只有一对时 compute_score 返回单个数值，统一成列表
    if isinstance(scores, (int, float)):
        scores = [float(scores)]

    scored = [(doc, float(score)) for doc, score in zip(candidates, scores)]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    logger.info(f"精排完成：候选 {len(candidates)} 条，阈值 {MIN_RERANK_SCORE}")
    _log_score_distribution([{"s": s} for _, s in scored], "s", "精排")

    # 先按阈值过滤（丢弃低相关），再取 Top-K
    reranked_docs, dropped = [], 0
    for doc, score in scored:
        if score < MIN_RERANK_SCORE:
            dropped += 1
            logger.info(
                f"  丢弃低相关切片 score={score:.4f} "
                f"来源={doc.get('file_title')} 标题={doc.get('title')}"
            )
            continue
        item = dict(doc)
        item["rerank_score"] = score
        reranked_docs.append(item)
        if len(reranked_docs) >= TOP_K:
            break

    if dropped:
        logger.warning(f"精排过滤：丢弃 {dropped} 条低相关切片（阈值 {MIN_RERANK_SCORE}）")

    for idx, doc in enumerate(reranked_docs, start=1):
        logger.info(
            f"  第{idx}名 score={doc['rerank_score']:.4f} "
            f"来源={doc.get('file_title')} 标题={doc.get('title')}"
        )

    if not reranked_docs:
        logger.warning("精排后无相关切片，将由答案节点返回「知识库暂未收录」")

    add_done_task(session_id, "node_rerank", is_stream)
    return {"reranked_docs": reranked_docs}
