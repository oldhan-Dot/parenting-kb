from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class RerankerConfig:
    bge_reranker_large: str    # 本地模型路径
    bge_reranker_device: str   # 运行设备
    bge_reranker_fp16: bool    # 是否开启半精度


reranker_config = RerankerConfig(
    bge_reranker_large=os.getenv("BGE_RERANKER_LARGE"),
    bge_reranker_device=os.getenv("BGE_RERANKER_DEVICE"),
    bge_reranker_fp16=os.getenv("BGE_RERANKER_FP16") in ("1", "True", "true", 1),
)
