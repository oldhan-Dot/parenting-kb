"""
BGE-M3 向量生成封装

- 单例加载模型，避免重复初始化
- 一次编码同时产出「稠密向量 + 稀疏向量」，适配 Milvus 混合检索
- 开启模型原生 L2 归一化，适配 Milvus 的 IP / COSINE 度量
"""
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from app.conf.embedding_config import embedding_config
from app.core.logger import logger

_bge_m3_ef = None


def get_bge_m3_ef():
    """获取 BGE-M3 模型单例"""
    global _bge_m3_ef
    if _bge_m3_ef is not None:
        return _bge_m3_ef

    model_name = embedding_config.bge_m3_path or "BAAI/bge-m3"
    device = embedding_config.bge_device or "cpu"
    use_fp16 = embedding_config.bge_fp16 or False

    logger.info(
        "开始初始化BGE-M3模型",
        extra={
            "model_name": model_name,
            "device": device,
            "use_fp16": use_fp16,
            "normalize_embeddings": True,
        },
    )

    try:
        _bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            normalize_embeddings=True,
        )
        logger.success("BGE-M3模型初始化成功，已开启原生L2归一化")
        return _bge_m3_ef
    except Exception as e:
        logger.error(f"BGE-M3模型初始化失败：{str(e)}", exc_info=True)
        raise


def generate_embeddings(texts):
    """
    为文本列表生成稠密 + 稀疏向量

    :param texts: 文本列表（单条文本也要包成列表）
    :return: {"dense": [[...], ...], "sparse": [{维度: 权重}, ...]}
    """
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成向量入参不合法，texts必须为非空列表")
        raise ValueError("参数texts必须是包含文本的非空列表")

    logger.info(f"开始为{len(texts)}条文本生成混合向量嵌入")
    try:
        model = get_bge_m3_ef()
        embeddings = model.encode_documents(texts)

        # 把 CSR 格式的稀疏矩阵拆成「字典列表」，便于序列化与入库
        processed_sparse = []
        for i in range(len(texts)):
            sparse_indices = embeddings["sparse"].indices[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            sparse_data = embeddings["sparse"].data[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            processed_sparse.append({k: v for k, v in zip(sparse_indices, sparse_data)})

        result = {
            "dense": [emb.tolist() for emb in embeddings["dense"]],
            "sparse": processed_sparse,
        }
        logger.success(f"{len(texts)}条文本向量生成完成")
        return result
    except Exception as e:
        logger.error(f"文本向量生成失败：{str(e)}", exc_info=True)
        raise
