"""
检索条件 → Milvus 过滤表达式的转换工具

育儿元数据是多值存储的（例如 age_range = "3-6岁,6-12岁"），
所以这里用「包含匹配」来过滤：
  同一字段的多个取值之间是 OR（满足其一即可）
  不同字段之间是 AND（需要同时满足）
"""
from typing import Dict, Optional

from app.conf.domain_config import METADATA_FIELDS
from app.utils.escape_milvus_string_utils import build_like_condition


def build_metadata_filter_expr(filters: Optional[Dict[str, str]]) -> Optional[str]:
    """
    把条件字典转换成 Milvus 过滤表达式

    :param filters: {"age_range": "3-6岁", "problem_type": "情绪管理"} 这样的字典
    :return: 过滤表达式字符串；没有任何有效条件时返回 None
    """
    if not filters:
        return None

    field_conditions = []
    for field in METADATA_FIELDS:
        value = (filters.get(field) or "").strip()
        if not value:
            continue

        # 条件里可能包含多个取值，用逗号 / 顿号分隔
        keywords = [k.strip() for k in value.replace("、", ",").split(",") if k.strip()]
        if not keywords:
            continue

        if len(keywords) == 1:
            field_conditions.append(build_like_condition(field, keywords[0]))
        else:
            ors = " or ".join(build_like_condition(field, k) for k in keywords)
            field_conditions.append(f"({ors})")

    if not field_conditions:
        return None

    return " and ".join(field_conditions)


def describe_filters(filters: Optional[Dict[str, str]]) -> str:
    """把条件转成人类可读的中文描述（用于日志与答案来源展示）"""
    if not filters:
        return "无"

    from app.conf.domain_config import METADATA_FIELD_CN

    parts = []
    for field in METADATA_FIELDS:
        value = (filters.get(field) or "").strip()
        if value:
            parts.append(f"{METADATA_FIELD_CN.get(field, field)}={value}")
    return "；".join(parts) if parts else "无"
