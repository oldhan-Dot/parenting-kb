"""
导入流程的状态定义 + 育儿领域元数据规范

LangGraph 用 TypedDict 描述状态，节点之间通过这个字典传递数据。
"""
import copy
from typing import TypedDict, Dict

# 育儿领域常量统一定义在 app/conf/domain_config.py，这里统一转出，方便业务模块直接引用
from app.conf.domain_config import (  # noqa: F401
    METADATA_FIELDS,
    METADATA_FIELD_CN,
    CONTENT_TYPES,
    AGE_RANGES,
    CHUNK_SCALAR_FIELDS,
    CHUNK_OUTPUT_FIELDS,
)


class ImportGraphState(TypedDict):
    """导入流程状态：所有节点产生和消费的数据字段"""

    task_id: str                 # 任务唯一ID，用于追踪日志与进度

    # --- 流程控制标记 ---
    is_md_read_enabled: bool     # 是否走 Markdown 读取路径
    is_pdf_read_enabled: bool    # 是否走 PDF 转换路径

    # --- 路径相关 ---
    local_dir: str               # 当前任务的工作目录
    local_file_path: str         # 原始上传文件路径
    file_title: str              # 文件标题（文件名去后缀）
    source_file: str             # 原始文件名（含后缀）
    pdf_path: str                # PDF 文件路径
    md_path: str                 # Markdown 文件路径

    # --- 内容数据 ---
    md_content: str              # Markdown 全文
    chunks: list                 # 切片列表（含 metadata 与向量）
    metadata: Dict[str, str]     # 文档级元数据（内容类型/年龄段/问题类型/场景/标题/作者）


# 图状态默认初始值
graph_default_state: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "file_title": "",
    "source_file": "",
    "pdf_path": "",
    "md_path": "",
    "md_content": "",
    "chunks": [],
    "metadata": {},
}


def create_default_state(**overrides) -> ImportGraphState:
    """创建默认状态并支持字段覆盖"""
    state = copy.deepcopy(graph_default_state)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """返回一个干净的状态实例，避免全局变量污染"""
    return copy.deepcopy(graph_default_state)
