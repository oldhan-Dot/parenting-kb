"""
节点：入口节点（node_entry）

职责：
1. 接收上传文件路径
2. 判断文件类型（PDF / MD）
3. 设置路由标记 is_pdf_read_enabled / is_md_read_enabled
4. 提取文件标题与原始文件名
"""
from pathlib import Path

from app.core.logger import node_log, logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task

SUPPORTED_SUFFIXES = (".pdf", ".md")


@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    # 记录节点为运行中状态
    add_running_task(state["task_id"], "node_entry")

    local_file_path = state["local_file_path"]

    if not local_file_path:
        logger.warning("当前状态中没有 local_file_path，请检查初始状态")
        add_done_task(state["task_id"], "node_entry")
        return state

    if local_file_path.endswith(".pdf"):
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = local_file_path
    elif local_file_path.endswith(".md"):
        state["is_md_read_enabled"] = True
        state["md_path"] = local_file_path
    else:
        logger.warning(f"当前上传文件【{local_file_path}】不是系统支持的格式（仅支持 .pdf / .md）")
        add_done_task(state["task_id"], "node_entry")
        return state

    # 文件标题：文件名去掉后缀
    state["file_title"] = Path(local_file_path).stem
    # 原始文件名：含后缀，用于数据幂等清理
    state["source_file"] = Path(local_file_path).name

    add_done_task(state["task_id"], "node_entry")
    return state
