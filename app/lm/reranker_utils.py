"""bge-reranker-large 精排模型封装（单例）"""
from FlagEmbedding import FlagReranker

from app.conf.reranker_config import reranker_config
from app.core.logger import logger

_reranker_model = None


def get_reranker_model():
    """获取精排模型单例，避免重复加载"""
    global _reranker_model
    if _reranker_model is None:
        logger.info(f"开始初始化精排模型：{reranker_config.bge_reranker_large}")
        _reranker_model = FlagReranker(
            model_name_or_path=reranker_config.bge_reranker_large,
            device=reranker_config.bge_reranker_device,
            use_fp16=reranker_config.bge_reranker_fp16,
        )
        logger.success("精排模型初始化成功")
    return _reranker_model
