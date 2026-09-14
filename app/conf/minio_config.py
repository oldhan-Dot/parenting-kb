from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class MinIOConfig:
    endpoint: str        # MinIO 服务地址（含 http/https 和端口）
    access_key: str
    secret_key: str
    bucket_name: str     # 存储桶名
    minio_img_dir: str   # 图片存放目录
    minio_secure: bool   # 是否使用 SSL


minio_config = MinIOConfig(
    endpoint=os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    bucket_name=os.getenv("MINIO_BUCKET_NAME"),
    minio_img_dir=os.getenv("MINIO_IMG_DIR"),
    minio_secure=os.getenv("MINIO_SECURE") == "True",
)
