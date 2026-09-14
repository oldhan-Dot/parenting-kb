"""把 BGE-M3 向量模型下载到本地（modelscope 源）"""
from modelscope.hub.snapshot_download import snapshot_download

# 下载到本地缓存目录，随后在 .env 里把 BGE_M3_PATH 指向该目录即可离线使用
model_dir = snapshot_download("BAAI/bge-m3", cache_dir="D:/ai_models/modelscope_cache/models")
print(f"模型已下载到: {model_dir}")
