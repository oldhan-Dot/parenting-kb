"""
节点：PDF 转 Markdown（node_pdf_to_md）

调用 MinerU 在线 API 完成解析：
1. 申请上传链接          → /file-urls/batch
2. 上传 PDF 到签名 URL
3. 轮询解析结果          → /extract-results/batch/{batch_id}
4. 下载 ZIP、解压、定位 md 文件，并读入 state["md_content"]
"""
import shutil
import time
import zipfile
from pathlib import Path

import requests

from app.conf.mineru_config import mineru_config
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task


@step_log("step_1_validate_paths")
def step_1_validate_paths(state: ImportGraphState):
    """校验输入路径：PDF 必须存在；输出目录为空则补默认值并自动创建"""
    pdf_path = state.get("pdf_path")
    local_dir = state.get("local_dir")

    if not pdf_path:
        logger.error("pdf_path 为空，请重新上传文件")
        raise ValueError("pdf_path 为空，请重新上传文件")

    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        logger.error(f"{pdf_path} 对应的文件不存在，请检查文件来源")
        raise ValueError(f"{pdf_path} 对应的文件不存在，请检查文件来源")

    if not local_dir:
        from app.utils.path_util import PROJECT_ROOT
        local_dir = PROJECT_ROOT / "output"
        logger.warning(f"local_dir 为空，使用默认值：{local_dir}")

    local_dir_obj = Path(local_dir)
    if not local_dir_obj.exists():
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj, local_dir_obj


@step_log("step_2_upload_and_poll")
def step_2_upload_and_poll(pdf_path_obj: Path, local_dir_obj: Path) -> str:
    """
    上传 PDF 至 MinerU 并轮询解析状态

    :return: 解析结果 ZIP 包下载链接
    """
    if not mineru_config.api_key or not mineru_config.base_url:
        logger.error("MinerU 配置为空，请检查 .env")
        raise ValueError("MinerU 配置为空，请检查 .env")

    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    data = {
        "files": [{"name": "demo.pdf", "data_id": "abcd"}],
        "model_version": "vlm",
    }

    # 第一次请求：申请上传链接
    response = requests.post(url, headers=header, json=data)
    if response.status_code != 200:
        logger.error(f"连接 MinerU 服务器失败，状态码：{response.status_code}")
        raise RuntimeError(f"连接 MinerU 服务器失败，状态码：{response.status_code}")

    result = response.json()
    if result["code"] != 0:
        logger.error(f"MinerU 接口调用失败，code={result['code']}，msg={result['msg']}")
        raise RuntimeError(f"MinerU 接口调用失败，code={result['code']}，msg={result['msg']}")

    batch_id = result["data"]["batch_id"]
    file_upload_url = result["data"]["file_urls"][0]

    # 第二次请求：把 PDF 上传到签名 URL（trust_env=False 避免代理导致 OSS 签名校验失败）
    pdf_file_data = pdf_path_obj.read_bytes()
    with requests.Session() as session:
        session.trust_env = False
        upload_response = session.put(file_upload_url, data=pdf_file_data)
        if upload_response.status_code != 200:
            raise RuntimeError(f"PDF 文件上传失败，状态码：{upload_response.status_code}，请重试")

    # 第三次请求：轮询解析结果
    batch_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    timeout_seconds = 600   # 最长等待 10 分钟
    poll_interval = 3       # 每 3 秒轮询一次
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout_seconds:
            logger.error("获取解析结果超时")
            raise TimeoutError("获取解析结果超时")

        try:
            poll_response = requests.get(batch_url, headers=header)
        except Exception:
            logger.warning("获取解析结果时出现异常，稍后重试")
            time.sleep(poll_interval)
            continue

        status_code = poll_response.status_code
        if status_code != 200:
            if 500 <= status_code < 600:
                logger.warning(f"MinerU 服务端异常，状态码：{status_code}，稍后重试")
                time.sleep(poll_interval)
                continue
            raise RuntimeError(f"访问 MinerU 服务端失败，状态码：{status_code}")

        poll_result = poll_response.json()
        if poll_result["code"] != 0:
            logger.warning(f"MinerU 接口异常，code={poll_result['code']}，msg={poll_result['msg']}，稍后重试")
            time.sleep(poll_interval)
            continue

        extract_result = poll_result["data"]["extract_result"][0]
        if not extract_result:
            logger.warning("未获取到解析结果，稍后重试")
            time.sleep(poll_interval)
            continue

        state = extract_result["state"]
        if state == "done":
            full_zip_url = extract_result.get("full_zip_url")
            if not full_zip_url:
                raise RuntimeError("解析任务已完成，但没有有效的压缩包下载地址")
            return full_zip_url
        elif state == "failed":
            raise RuntimeError("MinerU 解析失败")
        else:
            time.sleep(poll_interval)


@step_log("step_3_download_and_extract")
def step_3_download_and_extract(zip_url: str, local_dir_obj: Path, stem: str) -> str:
    """
    下载并解压 MinerU 结果，定位 md 文件并统一命名为「PDF 同名.md」

    :return: md 文件绝对路径
    """
    response = requests.get(zip_url, timeout=120)
    if response.status_code != 200:
        raise ValueError(f"下载压缩文件失败，状态码：{response.status_code}")

    zip_save_path = local_dir_obj / f"{stem}_result.zip"
    zip_save_path.write_bytes(response.content)

    extract_target_dir = local_dir_obj / stem
    if extract_target_dir.exists():
        shutil.rmtree(extract_target_dir)
    extract_target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_save_path, "r") as zip_file:
        zip_file.extractall(extract_target_dir)

    md_file_list = list(extract_target_dir.rglob("*.md"))
    if not md_file_list:
        raise RuntimeError("解压后的结果中没有任何 md 文件")

    # 优先级：与 PDF 同名的 md → full.md → 第一个 md
    target_md_file = None
    for md_file in md_file_list:
        if md_file.stem == stem:
            target_md_file = md_file
            break
    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name == "full.md":
                target_md_file = md_file
                break
    if not target_md_file:
        target_md_file = md_file_list[0]

    if target_md_file.stem != stem:
        target_md_file = target_md_file.rename(target_md_file.with_name(f"{stem}.md"))

    return str(target_md_file.resolve())


@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    add_running_task(state["task_id"], "node_pdf_to_md")

    pdf_path_obj, local_dir_obj = step_1_validate_paths(state)
    zip_url = step_2_upload_and_poll(pdf_path_obj, local_dir_obj)
    final_md_path = step_3_download_and_extract(zip_url, local_dir_obj, pdf_path_obj.stem)

    state["md_path"] = final_md_path
    with open(final_md_path, "r", encoding="utf-8") as f:
        state["md_content"] = f.read()

    add_done_task(state["task_id"], "node_pdf_to_md")
    return state
