"""
内存态任务追踪工具（单进程）

用于前端轮询 / SSE 推送时展示「当前处理到哪一步」：
- running_list：正在运行的节点
- done_list：已完成的节点（对外返回中文名）
- status：pending / processing / completed / failed
- result：任务结果字段（如查询的 answer）
"""
from typing import Dict, List

from .sse_utils import push_to_session

# ---------------------------
# 内存态任务追踪（单进程）
# ---------------------------
# key: task_id / session_id
_tasks_running_list: Dict[str, List[str]] = {}
_tasks_done_list: Dict[str, List[str]] = {}
_tasks_status: Dict[str, str] = {}
_tasks_result: Dict[str, Dict[str, str]] = {}

TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# 节点名 -> 中文名映射（key 必须与 LangGraph add_node 的节点名一致）
_NODE_NAME_TO_CN: Dict[str, str] = {
    # --- 导入流程 ---
    "upload_file": "开始上传文件",
    "node_entry": "检查文件类型",
    "node_pdf_to_md": "PDF转Markdown",
    "node_md_img": "Markdown图片处理",
    "node_document_split": "文档切分",
    "node_metadata_extract": "元数据抽取",
    "node_bge_embedding": "向量生成",
    "node_import_milvus": "导入向量库",
    "END": "处理完成",
    "__end__": "处理完成",
    # --- 查询流程 ---
    "node_filter_extract": "解析检索条件",
    "node_search_embedding": "切片检索",
    "node_search_embedding_hyde": "切片检索(HyDE)",
    "node_rrf": "多路结果融合",
    "node_rerank": "结果精排",
    "node_answer_output": "生成答案",
}


def _ensure_task(task_id: str) -> None:
    """确保 task_id 对应的数据结构已初始化"""
    if task_id not in _tasks_running_list:
        _tasks_running_list[task_id] = []
    if task_id not in _tasks_done_list:
        _tasks_done_list[task_id] = []
    if task_id not in _tasks_result:
        _tasks_result[task_id] = {}


def _to_cn(node_name: str) -> str:
    """节点名转中文展示名（无映射则返回原名）"""
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """记录节点开始运行"""
    _ensure_task(task_id)
    running = _tasks_running_list[task_id]
    if node_name not in running:
        running.append(node_name)

    if is_stream:
        task_push_queue(task_id)


def add_done_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """记录节点已完成（同时从 running 中移除同名节点）"""
    _ensure_task(task_id)

    running = _tasks_running_list[task_id]
    _tasks_running_list[task_id] = [n for n in running if n != node_name]

    done = _tasks_done_list[task_id]
    if node_name not in done:
        done.append(node_name)

    if is_stream:
        task_push_queue(task_id)


def set_task_result(task_id: str, key: str, value: str) -> None:
    """存储任务结果字段（如 answer / error）"""
    _ensure_task(task_id)
    _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    """获取任务结果字段"""
    _ensure_task(task_id)
    return _tasks_result.get(task_id, {}).get(key, default)


def get_task_status(task_id: str) -> str:
    """获取当前任务状态"""
    return _tasks_status.get(task_id, "")


def get_done_task_list(task_id: str) -> List[str]:
    """获取已完成节点列表（中文展示）"""
    _ensure_task(task_id)
    return [_to_cn(n) for n in _tasks_done_list.get(task_id, [])]


def get_running_task_list(task_id: str) -> List[str]:
    """获取正在运行节点列表（中文展示）"""
    _ensure_task(task_id)
    return [_to_cn(n) for n in _tasks_running_list.get(task_id, [])]


def update_task_status(task_id: str, status_name: str, push_queue: bool = False) -> None:
    """更新任务状态"""
    _tasks_status[task_id] = status_name
    if push_queue:
        task_push_queue(task_id)


def task_push_queue(task_id: str):
    """把当前进度推送到 SSE 队列"""
    push_to_session(task_id, "progress", {
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    })


def clear_task(task_id: str):
    """清理任务缓存"""
    _tasks_running_list.pop(task_id, None)
    _tasks_done_list.pop(task_id, None)
    _tasks_status.pop(task_id, None)
    _tasks_result.pop(task_id, None)
