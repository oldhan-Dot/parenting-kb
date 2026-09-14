from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class MilvusConfig:
    milvus_url: str          # Milvus 服务端连接地址
    chunks_collection: str   # 育儿知识切片集合名称


milvus_config = MilvusConfig(
    milvus_url=os.getenv("MILVUS_URL"),
    chunks_collection=os.getenv("CHUNKS_COLLECTION"),
)
