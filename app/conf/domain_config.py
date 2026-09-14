"""
育儿领域常量配置

导入侧（元数据抽取、Milvus schema）与查询侧（条件提取、过滤表达式）共用同一份规范，
保证「写进去的字段」和「查出来的条件」始终一致。
"""
from typing import List

# 需要写进每个切片、并作为 Milvus 标量字段的元数据
METADATA_FIELDS: List[str] = ["content_type", "age_range", "problem_type", "scene"]

# 元数据字段的中文名（用于日志与提示词展示）
METADATA_FIELD_CN = {
    "content_type": "内容类型",
    "age_range": "年龄段",
    "problem_type": "问题类型",
    "scene": "场景描述",
}

# 内容类型白名单（与需求说明「内容要求」一致）
CONTENT_TYPES: List[str] = ["育儿建议", "专家建议", "亲子案例", "沟通话术", "知识科普"]

# 年龄段白名单（与需求说明「内容要求」一致）
AGE_RANGES: List[str] = ["0-3岁", "3-6岁", "6-12岁", "12+岁"]

# 写入 Milvus 的标量字段（与集合 schema 严格对应）
CHUNK_SCALAR_FIELDS: List[str] = [
    "file_title",      # 来源文件名（去后缀）
    "source_file",     # 来源文件名（含后缀）
    "source_path",     # 来源路径
    "content_type",    # 内容类型
    "age_range",       # 年龄段（多值用英文逗号拼接）
    "problem_type",    # 问题类型（多值用英文逗号拼接）
    "scene",           # 场景描述（多值用英文逗号拼接）
    "title",           # 切片标题
    "parent_title",    # 所属章节标题
    "part",            # 切片在同章节内的序号
    "author",          # 作者
    "content",         # 切片正文
]

# 检索时返回给大模型的切片字段
CHUNK_OUTPUT_FIELDS: List[str] = [
    "chunk_id",
    "content",
    "title",
    "parent_title",
    "file_title",
    "content_type",
    "age_range",
    "problem_type",
    "scene",
]
