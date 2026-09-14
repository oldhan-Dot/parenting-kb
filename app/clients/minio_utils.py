"""
MinIO 对象存储封装

用于存放从 PDF 解析出来的图片，使其可以通过 HTTP URL 在答案中展示。
"""
import json

from minio import Minio

from app.conf.minio_config import minio_config
from app.core.logger import logger

_minio_client = None


def _create_minio_client() -> Minio:
    """创建 MinIO 客户端连接"""
    return Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        secure=minio_config.minio_secure,
    )


def _set_bucket_policy(bucket_name: str) -> str:
    """生成桶访问策略：允许匿名只读，便于图片 URL 直接访问"""
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
            }
        ],
    }
    return json.dumps(policy)


def _create_bucket_ready(client: Minio):
    """确保桶存在，并设置公开只读策略"""
    bucket_name = minio_config.bucket_name
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        client.set_bucket_policy(bucket_name, _set_bucket_policy(bucket_name))
        logger.info(f"MinIO桶 {bucket_name} 已创建，并设置访问策略")
    else:
        logger.info(f"MinIO桶 {bucket_name} 已存在，无需重复创建")


def get_minio_client() -> Minio:
    """获取 MinIO 客户端（懒加载 + 单例）"""
    global _minio_client

    if _minio_client is None:
        logger.info("开始初始化MinIO客户端（首次调用，执行懒加载）")
        client = _create_minio_client()
        _create_bucket_ready(client)
        _minio_client = client
        logger.info("MinIO客户端初始化完成，已就绪可使用")

    return _minio_client


def build_object_url(object_path: str) -> str:
    """
    拼接对象的公网访问地址

    :param object_path: 对象在桶内的路径，如 /parenting-images/xxx/a.jpg
    """
    bucket = minio_config.bucket_name
    dir_part = minio_config.minio_img_dir.lstrip("/")
    return f"http://{minio_config.endpoint}/{bucket}/{dir_part}/{object_path.lstrip('/')}"
