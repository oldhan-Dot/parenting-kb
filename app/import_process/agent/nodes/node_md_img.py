"""
节点：Markdown 图片处理（node_md_img）

职责：
1. 扫描 md 中引用的本地图片（育儿数据多为纯文本，这里做了「无图直接跳过」的容错）
2. 调用视觉模型为图片生成摘要
3. 上传图片到 MinIO，把 md 中的图片地址替换为 MinIO URL
4. md 中图片与本地 images 目录都不存在时，直接透传原内容
"""
import base64
import re
from collections import deque
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject

from app.clients.minio_utils import get_minio_client
from app.conf.lm_config import lm_config
from app.conf.minio_config import minio_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.lm.lm_utils import get_llm_client
from app.utils.rate_limit_utils import apply_api_rate_limit
from app.utils.task_utils import add_running_task, add_done_task

# MinIO 支持的图片格式（统一小写匹配）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def is_supported_image(file_name: str) -> bool:
    """判断文件名是否为支持的图片类型"""
    return Path(file_name).suffix.lower() in IMAGE_EXTENSIONS


@step_log("step_1_get_content")
def step_1_get_content(state: ImportGraphState):
    """校验 md 路径、必要时补读 md_content，并返回同目录下的 images 目录"""
    md_path = state["md_path"]
    if not md_path:
        raise ValueError("md_path 为空，请检查流程")

    md_path_obj = Path(md_path)
    if not md_path_obj.exists():
        raise FileNotFoundError(f"{md_path} 文件不存在")

    # md_content 为空说明是「直接上传 md」的路径（node_entry → node_md_img）
    if not state["md_content"]:
        state["md_content"] = md_path_obj.read_text(encoding="utf-8")

    images_dir_obj = md_path_obj.parent / "images"
    return state["md_content"], md_path_obj, images_dir_obj


@step_log("step_2_scan_images")
def step_2_scan_images(md_content: str, images_dir_obj: Path):
    """
    扫描 md 中实际引用的图片

    :return: [(图片名, 图片路径, (上文, 下文)), ...]
    """
    image_targets = []

    # 容错：育儿数据基本是纯文本 md，没有 images 目录属于正常情况
    if not images_dir_obj.exists() or not images_dir_obj.is_dir():
        logger.info(f"未发现图片目录（{images_dir_obj}），跳过图片处理")
        return image_targets

    for image_file in images_dir_obj.iterdir():
        image_name = image_file.name
        if not is_supported_image(image_name):
            logger.warning(f"{image_name} 不是支持的图片类型，跳过")
            continue

        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        match_results = list(pattern.finditer(md_content))
        if not match_results:
            logger.warning(f"图片 {image_name} 未在 md 中被引用，跳过")
            continue

        start, end = match_results[0].span()
        pre_text = md_content[max(0, start - 100): start]
        post_text = md_content[end:(min(end + 100, len(md_content)))]
        image_targets.append((image_name, str(image_file), (pre_text, post_text)))

    return image_targets


@step_log("step_3_image_summary")
def step_3_image_summary(image_targets, stem: str):
    """调用视觉模型为每张图片生成一句摘要（用于替换 md 中的图片 alt）"""
    summaries = {}
    requests_limiter = deque()

    for image_name, image_path, context in image_targets:
        # 限流：视觉模型每分钟限制调用次数
        apply_api_rate_limit(requests_limiter, max_requests=100)

        prompt = load_prompt("image_summary", root_folder=stem, image_content=context)
        vl_model = get_llm_client(lm_config.lv_model)

        if isinstance(image_path, str):
            image_path = Path(image_path)
        image_base64 = base64.b64encode(image_path.read_bytes()).decode(encoding="utf-8")

        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
                {"type": "text", "text": prompt},
            ]
        )
        chain = vl_model | StrOutputParser()
        summaries[image_name] = chain.invoke([message])

    return summaries


@step_log("step_4_upload_images_replace")
def step_4_upload_images_replace(image_summaries, image_targets, md_content: str, stem: str):
    """清理旧的同前缀图片 → 上传新图片 → 替换 md 中的图片地址与描述"""
    minio_client = get_minio_client()

    # 先删掉该文档上次导入时上传的图片（幂等）
    prefix = f"{minio_config.minio_img_dir.lstrip('/')}/{stem}"
    object_list = minio_client.list_objects(
        bucket_name=minio_config.bucket_name,
        prefix=prefix,
        recursive=True,
    )
    delete_object_list = [DeleteObject(obj.object_name) for obj in object_list]
    for error in minio_client.remove_objects(
            bucket_name=minio_config.bucket_name,
            delete_object_list=delete_object_list,
    ):
        logger.warning(f"图片删除失败：{error}")

    # 上传图片并记录 URL
    image_urls = {}
    img_dir = minio_config.minio_img_dir.lstrip("/")
    for image_name, image_path, _ in image_targets:
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=f"{img_dir}/{stem}/{image_name}",
                file_path=image_path,
                content_type="image/jpeg",
            )
            image_urls[image_name] = (
                f"http://{minio_config.endpoint}/{minio_config.bucket_name}/{img_dir}/{stem}/{image_name}"
            )
        except Exception as e:
            logger.warning(f"{image_name} 上传失败：{e}")

    # 把 ![](...) 替换为 ![摘要](MinIO URL)
    image_infos = {}
    for image_name, summary in image_summaries.items():
        if image_name in image_urls:
            image_infos[image_name] = (summary, image_urls[image_name])

    for image_name, (summary, url) in image_infos.items():
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        md_content = pattern.sub(lambda _: f"![{summary}]({url})", md_content)

    return md_content


@step_log("step_5_backup_md_file")
def step_5_backup_md_file(md_path_obj: Path, new_md_content: str) -> str:
    """把替换后的内容备份为「原文件名_new.md」"""
    new_md_path_obj = md_path_obj.parent / f"{md_path_obj.stem}_new{md_path_obj.suffix}"
    new_md_path_obj.write_text(new_md_content, encoding="utf-8")
    return str(new_md_path_obj)


@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    add_running_task(state["task_id"], "node_md_img")

    md_content, md_path_obj, images_dir_obj = step_1_get_content(state)
    image_targets = step_2_scan_images(md_content, images_dir_obj)

    # 无图直接透传，避免育儿纯文本文档多做一次无用备份
    if not image_targets:
        logger.info("md 中未发现需要处理的图片，图片节点直接透传")
        add_done_task(state["task_id"], "node_md_img")
        return state

    image_summaries = step_3_image_summary(image_targets, md_path_obj.stem)
    new_md_content = step_4_upload_images_replace(
        image_summaries, image_targets, md_content, md_path_obj.stem
    )
    new_md_file_path_str = step_5_backup_md_file(md_path_obj, new_md_content)

    state["md_content"] = new_md_content
    state["md_path"] = new_md_file_path_str

    add_done_task(state["task_id"], "node_md_img")
    return state
