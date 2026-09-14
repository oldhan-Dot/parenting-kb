"""
节点：文档切分（node_document_split）

两段式切分策略：
1. 按 Markdown 标题层级初切，保证「一个语义段落」不被拆散
2. 同标题下超长的内容再用 RecursiveCharacterTextSplitter 二次切分

切分结果会备份到 md 同目录的 backup.json，便于排查检索效果。
"""
import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 单个切片最大长度（正式导入建议 500，过小会切出大量碎片影响检索）
CHUNK_SIZE = 500
# 切片之间的重叠长度（保证边界语义不丢失）
CHUNK_OVERLAP = 50


@step_log("step_1_get_content")
def step_1_get_content(state: ImportGraphState):
    """取 md 内容并做换行符归一化"""
    md_content = state["md_content"]
    if not md_content:
        logger.error("md 文档内容获取失败，无法完成切分")
        raise RuntimeError("md 文档内容获取失败，无法完成切分")

    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    file_title = state["file_title"]
    return md_content, file_title


@step_log("step_2_split_by_title")
def step_2_split_by_title(md_content: str, file_title: str):
    """按 Markdown 标题初切（自动跳过代码块内的 # 行）"""
    pattern = re.compile(r"^#{1,6}\s+.+")

    lines = md_content.split("\n")
    sections = []
    current_title = ""
    current_lines = []
    is_code_block = False

    for line in lines:
        line = line.strip()

        if line.startswith("```") or line.startswith("~~~"):
            # 代码块开始/结束行：取反标记
            is_code_block = not is_code_block
            current_lines.append(line)
            continue

        if not is_code_block and pattern.match(line):
            if current_title:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_lines),
                    "file_title": file_title,
                })
            current_title = line
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_title:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_lines),
            "file_title": file_title,
        })

    # 兜底：整篇没有任何标题时，全文作为一个切片
    if not sections:
        sections.append({
            "title": "无主题",
            "content": md_content,
            "file_title": file_title,
        })

    return sections


@step_log("step_3_refine_chunks")
def step_3_refine_chunks(sections):
    """对同一标题下的超长内容做二次切分，控制切片粒度"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 切割优先级：段落 → 换行 → 句子 → 空格
        separators=["\n\n", "\n", "。", "！", "；", " "],
    )

    final_chunks = []
    for chunk in sections:
        sub_chunks = splitter.split_text(chunk["content"])
        has_multiple_chunks = len(sub_chunks) > 1

        for idx, sub_chunk in enumerate(sub_chunks, start=1):
            # 多个子切片时标题加序号后缀，便于定位
            current_title = f"{chunk['title']}_{idx}" if has_multiple_chunks else chunk["title"]
            final_chunks.append({
                "title": current_title,
                "content": sub_chunk,
                "parent_title": chunk["title"],
                "file_title": chunk["file_title"],
                "part": idx,
            })

    return final_chunks


@step_log("step_4_backup_chunks")
def step_4_backup_chunks(final_chunks, state: ImportGraphState):
    """把切片结果备份到 md 同目录的 backup.json，便于调试检索效果"""
    chunks_backup_path = Path(state["md_path"]).parent / "backup.json"
    with open(chunks_backup_path, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, ensure_ascii=False, indent=4)
    logger.info(f"切片结果已备份：{chunks_backup_path}，共 {len(final_chunks)} 个切片")


@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    add_running_task(state["task_id"], "node_document_split")

    md_content, file_title = step_1_get_content(state)
    sections = step_2_split_by_title(md_content, file_title)
    final_chunks = step_3_refine_chunks(sections)

    state["chunks"] = final_chunks
    step_4_backup_chunks(final_chunks, state)

    add_done_task(state["task_id"], "node_document_split")
    return state
