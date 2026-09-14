"""
Milvus 检索结果拍平工具

Milvus 混合检索返回的结构是：
[
    [
        {"id": 123, "distance": 0.91, "entity": {"chunk_id": 123, "content": "...", ...}},
        ...
    ]
]
本模块把它拍平成扁平的字典列表，方便后续 RRF 融合、精排与 Prompt 组装。
"""
from typing import Dict, List


def flatten_hybrid_hits(hits) -> List[Dict]:
    """
    把 Milvus 命中项拍平为 {字段: 值, chunk_id, distance}

    :param hits: 单次混合检索的命中列表
    :return: 扁平字典列表
    """
    flattened = []
    for hit in hits or []:
        if not isinstance(hit, dict):
            continue

        entity = hit.get("entity") or {}
        item = dict(entity)

        # chunk_id 可能落在 entity 里，也可能用 id 返回
        item["chunk_id"] = item.get("chunk_id") or hit.get("id")
        item["distance"] = hit.get("distance", 0.0)

        flattened.append(item)

    return flattened


def dedupe_by_chunk_id(docs: List[Dict]) -> List[Dict]:
    """按 chunk_id 去重（保持顺序，保留首次出现的记录）"""
    seen = set()
    result = []
    for doc in docs or []:
        cid = doc.get("chunk_id")
        if cid in seen:
            continue
        seen.add(cid)
        result.append(doc)
    return result
