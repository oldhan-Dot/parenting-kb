"""
Milvus 客户端与混合检索封装

- get_milvus_client：客户端单例
- create_hybrid_search_requests：构建稠密 + 稀疏两路 ANN 检索请求
- hybrid_search：WeightedRanker 加权融合两路结果
- fetch_chunks_by_chunk_ids：按主键批量补全切片内容
"""
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker

from app.conf.milvus_config import milvus_config
from app.core.logger import logger

# 全局 Milvus 客户端实例，实现单例复用
_milvus_client = None


def get_milvus_client():
    """
    获取 Milvus 客户端单例

    :return: MilvusClient 实例；连接失败返回 None
    """
    try:
        global _milvus_client
        if _milvus_client is None:
            milvus_uri = milvus_config.milvus_url
            if not milvus_uri:
                logger.error("Milvus客户端连接失败：缺少 MILVUS_URL 配置")
                return None
            _milvus_client = MilvusClient(uri=milvus_uri)
            logger.info("Milvus客户端连接成功")
        return _milvus_client
    except Exception as e:
        logger.error(f"Milvus客户端连接异常：{str(e)}", exc_info=True)
        return None


def _coerce_int64_ids(ids):
    """把 chunk_id 统一转成 INT64（主键字段为 INT64），返回 (可用, 不可用)"""
    ok, bad = [], []
    for x in (ids or []):
        if x is None:
            continue
        try:
            ok.append(int(x))
        except Exception:
            bad.append(x)
    return ok, bad


def fetch_chunks_by_chunk_ids(
        client,
        collection_name: str,
        chunk_ids,
        *,
        output_fields=None,
        batch_size: int = 100,
):
    """
    按 chunk_id 主键批量查询切片（优先 get，失败回退 query）

    :return: Milvus 实体字典列表；查询失败返回空列表
    """
    if client is None:
        return []
    if not collection_name:
        return []

    if output_fields is None:
        output_fields = ["chunk_id", "content", "title", "parent_title", "file_title"]

    ok_ids, bad_ids = _coerce_int64_ids(chunk_ids)
    if bad_ids:
        logger.warning(f"存在无法转换为INT64的chunk_id，将跳过查询：{bad_ids}")
    if not ok_ids:
        return []

    results = []
    for i in range(0, len(ok_ids), batch_size):
        batch = ok_ids[i: i + batch_size]

        if hasattr(client, "get"):
            try:
                got = client.get(collection_name=collection_name, ids=batch, output_fields=output_fields)
                if got:
                    results.extend(got)
                continue
            except Exception as e:
                logger.warning(f"Milvus get方法查询失败，将回退至query方法：{str(e)}")

        try:
            expr = f"chunk_id in [{', '.join(str(x) for x in batch)}]"
            q = client.query(collection_name=collection_name, filter=expr, output_fields=output_fields)
            if q:
                results.extend(q)
        except Exception as e:
            logger.error(f"Milvus query方法批量查询chunk_id失败：{str(e)}", exc_info=True)

    return results


def create_hybrid_search_requests(dense_vector, sparse_vector, dense_params=None, sparse_params=None,
                                  expr=None, limit=5):
    """
    构建 Milvus 混合搜索请求对象（稠密 + 稀疏两路）

    :param expr: 标量过滤表达式，用于按年龄段 / 问题类型等条件缩小检索范围
    :param limit: 单路检索返回条数
    :return: [dense_req, sparse_req]
    """
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit,
    )

    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit,
    )

    return [dense_req, sparse_req]


def hybrid_search(client, collection_name, reqs, ranker_weights=(0.5, 0.5), norm_score=False, limit=5,
                  output_fields=None, search_params=None):
    """
    执行 Milvus 稠密 + 稀疏混合搜索

    :param ranker_weights: 加权融合权重，(稠密, 稀疏)，默认 (0.5, 0.5)
    :param norm_score: 是否归一化评分后再融合
    :return: 混合搜索结果（二维列表：外层是单次查询，内层是命中列表）；失败返回 None
    """
    try:
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)

        if output_fields is None:
            output_fields = ["chunk_id", "content", "title", "file_title"]

        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params,
        )

        logger.info(f"Milvus混合搜索完成，集合[{collection_name}]共检索到{len(res[0]) if res else 0}条结果")
        return res
    except Exception as e:
        logger.error(f"Milvus混合搜索执行失败，集合[{collection_name}]：{str(e)}", exc_info=True)
        return None


def get_collection_stats(collection_name: str):
    """获取集合统计信息（用于自检：确认数据是否真的写入成功）"""
    client = get_milvus_client()
    if client is None:
        return None
    try:
        if not client.has_collection(collection_name=collection_name):
            return {"exists": False, "row_count": 0}
        stats = client.get_collection_stats(collection_name=collection_name)
        return {"exists": True, "row_count": stats.get("row_count", 0)}
    except Exception as e:
        logger.error(f"获取集合统计信息失败：{str(e)}", exc_info=True)
        return None
