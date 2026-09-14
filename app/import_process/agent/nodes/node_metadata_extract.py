"""
节点：元数据抽取（node_metadata_extract）

育儿数据的 md 头部已经带了规范的 `## 元数据` 段，所以这里的策略是：
1. 优先用正则「精确解析」md 里的元数据段（准确率 100%，且零成本）
2. 只有解析不到时，才调用大模型兜底（把成本花在真正需要的文档上）
3. 解析出的元数据会写入 state["metadata"]，并绑定到每一个切片上

产出字段：content_type / age_range / problem_type / scene / title / author / file_title / source_file / source_path
"""
import json
import re
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import (
    ImportGraphState,
    METADATA_FIELDS,
    CONTENT_TYPES,
    AGE_RANGES,
)
from app.lm.lm_utils import get_llm_client
from app.utils.task_utils import add_running_task, add_done_task

# 「## 元数据」这一节的正文（直到下一个二级标题或文末）
METADATA_BLOCK_PATTERN = re.compile(r"##\s*元数据\s*\n(.*?)(?=\n#{1,6}\s|\Z)", re.S)

# 各字段在元数据段中的取值行
FIELD_PATTERNS = {
    "content_type": re.compile(r"内容类型\s*[:：]\s*(.+)"),
    "age_range": re.compile(r"年龄段\s*[:：]\s*(.+)"),
    "problem_type": re.compile(r"问题类型\s*[:：]\s*(.+)"),
    "scene": re.compile(r"场景描述\s*[:：]\s*(.+)"),
    # 作者字段育儿数据里没有，但需求要求保留，兜底为「未知」
    "author": re.compile(r"作者\s*[:：]\s*(.+)"),
}

# 多值分隔符：中英文斜杠、顿号、分号、逗号都算
MULTI_VALUE_SPLITTER = re.compile(r"[/／、；;,，]")

# 大模型兜底时用的上下文长度上限
FALLBACK_CONTEXT_MAX_CHARS = 2000


def _normalize_multi(value: str) -> str:
    """
    把多值字符串规范化为「英文逗号拼接」形式

    例："3-6岁 / 6-12岁" -> "3-6岁,6-12岁"
    这样 Milvus 里用 like "%3-6岁%" 就能命中
    """
    if not value:
        return ""
    items = [item.strip() for item in MULTI_VALUE_SPLITTER.split(value) if item.strip()]
    # 去重且保持顺序
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ",".join(ordered)


@step_log("step_1_get_content")
def step_1_get_content(state: ImportGraphState):
    """取 md 内容、文件标题、原始文件名、来源路径"""
    md_content = state.get("md_content")
    if not md_content:
        raise RuntimeError("md_content 为空，无法抽取元数据")

    file_title = state.get("file_title") or Path(state.get("md_path", "")).stem
    source_file = state.get("source_file") or Path(state.get("local_file_path", "")).name
    source_path = state.get("local_file_path", "")

    return md_content, file_title, source_file, source_path


@step_log("step_2_extract_title")
def step_2_extract_title(md_content: str) -> str:
    """取文档第一个一级标题作为文章标题"""
    for line in md_content.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line.lstrip("#").strip()
    return ""


@step_log("step_3_parse_metadata_block")
def step_3_parse_metadata_block(md_content: str) -> dict:
    """用正则精确解析 `## 元数据` 段"""
    raw = {}
    block_match = METADATA_BLOCK_PATTERN.search(md_content)

    if not block_match:
        logger.warning("未找到「## 元数据」段，将尝试使用大模型兜底")
        return raw

    block = block_match.group(1)
    for field, pattern in FIELD_PATTERNS.items():
        match = pattern.search(block)
        if match:
            raw[field] = match.group(1).strip()

    logger.info(f"元数据段解析结果：{raw}")
    return raw


@step_log("step_4_normalize")
def step_4_normalize(raw: dict, file_title: str, source_file: str, source_path: str, title: str) -> dict:
    """规范化字段值：多值统一、去重、白名单校验、兜底"""
    metadata = dict(raw)

    for field in ("age_range", "problem_type", "scene"):
        metadata[field] = _normalize_multi(metadata.get(field, ""))

    # 年龄段白名单校验：只保留规范取值，避免脏数据影响过滤
    age_range = metadata.get("age_range", "")
    if age_range:
        valid = [item for item in age_range.split(",") if item in AGE_RANGES]
        if valid:
            metadata["age_range"] = ",".join(valid)
        else:
            logger.warning(f"年龄段【{age_range}】不在规范取值范围内，保留原值")

    # 内容类型兜底：从来源路径的父目录名推断
    content_type = metadata.get("content_type", "")
    if content_type not in CONTENT_TYPES:
        parent_dir_name = Path(source_path).parent.name
        if parent_dir_name in CONTENT_TYPES:
            logger.warning(f"内容类型【{content_type}】不规范，按父目录推断为【{parent_dir_name}】")
            metadata["content_type"] = parent_dir_name
        elif content_type:
            logger.warning(f"内容类型【{content_type}】不在规范取值范围内，保留原值")

    metadata["title"] = title or file_title
    metadata["author"] = metadata.get("author") or "未知"
    metadata["file_title"] = file_title
    metadata["source_file"] = source_file
    metadata["source_path"] = source_path

    return metadata


def _need_llm_fallback(metadata: dict) -> bool:
    """核心字段一个都没解析到时，才走大模型兜底"""
    return not any(metadata.get(field) for field in METADATA_FIELDS)


@step_log("step_5_llm_fallback")
def step_5_llm_fallback(md_content: str, file_title: str, metadata: dict) -> dict:
    """大模型兜底：从文档开头内容中抽取元数据"""
    context = md_content[:FALLBACK_CONTEXT_MAX_CHARS]
    prompt = load_prompt("metadata_extract", file_title=file_title, context=context)
    messages = [
        SystemMessage(content="你是育儿知识库的元数据抽取专家，只输出 JSON，不要输出任何解释。"),
        HumanMessage(content=prompt),
    ]

    try:
        llm = get_llm_client(json_mode=True)
        result = llm.invoke(messages).content
        if result.startswith("```"):
            result = result.replace("```json", "").replace("```", "")
        parsed = json.loads(result)
    except Exception as e:
        logger.error(f"大模型元数据兜底失败：{e}")
        return metadata

    # 只补齐缺失字段，不覆盖已解析出的结果
    for field in METADATA_FIELDS:
        value = parsed.get(field) or ""
        if isinstance(value, list):
            value = ",".join(str(v).strip() for v in value if str(v).strip())
        value = str(value).strip()
        if value and (value not in ("未知", "不确定", "none", "None")):
            metadata[field] = _normalize_multi(value) if field in ("age_range", "problem_type", "scene") else value

    logger.info(f"大模型兜底后的元数据：{metadata}")
    return metadata


@step_log("step_6_bind_to_chunks")
def step_6_bind_to_chunks(chunks: list, metadata: dict) -> list:
    """把文档级元数据绑定到每一个切片上"""
    if not chunks:
        raise RuntimeError("chunks 为空，无法绑定元数据")

    # 只把需要落库的元数据字段绑定到切片，避免携带无用信息
    bind_fields = METADATA_FIELDS + ["title", "author", "file_title", "source_file", "source_path"]
    for chunk in chunks:
        for field in bind_fields:
            if field == "title":
                continue  # title 用切片自身的章节标题，不用文档标题覆盖
            chunk[field] = metadata.get(field, "")
    return chunks


@node_log("node_metadata_extract")
def node_metadata_extract(state: ImportGraphState) -> ImportGraphState:
    add_running_task(state["task_id"], "node_metadata_extract")

    chunks = state.get("chunks")
    if not chunks:
        raise RuntimeError("chunks 为空，即没有任何切片")

    # 步骤1：取内容与基础信息
    md_content, file_title, source_file, source_path = step_1_get_content(state)
    # 步骤2：取文章标题
    title = step_2_extract_title(md_content)
    # 步骤3：正则精确解析元数据段
    raw_metadata = step_3_parse_metadata_block(md_content)
    # 步骤4：规范化
    metadata = step_4_normalize(raw_metadata, file_title, source_file, source_path, title)
    # 步骤5：必要时用大模型兜底
    if _need_llm_fallback(metadata):
        metadata = step_5_llm_fallback(md_content, file_title, metadata)
    # 步骤6：绑定到切片
    state["chunks"] = step_6_bind_to_chunks(chunks, metadata)
    state["metadata"] = metadata

    logger.info(f"元数据抽取完成：{metadata}")
    add_done_task(state["task_id"], "node_metadata_extract")
    return state
