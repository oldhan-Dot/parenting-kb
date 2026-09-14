def escape_milvus_string(value: str) -> str:
    """
    Milvus 过滤表达式专用字符串安全转义

    转义规则：
    1. 反斜杠（\\）→ 双反斜杠（\\\\）
    2. 双引号（"）→ 转义双引号（\\"）
    3. 换行/回车/制表符 → 空格（防止表达式被换行截断）

    :param value: 需要转义的原始字符串（如年龄段、问题类型）
    :return: 可安全用于 Milvus filter_expr 的字符串
    """
    if value is None:
        return ""
    s = str(value)
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return s


def build_like_condition(field_name: str, value: str) -> str:
    """
    构建「包含匹配」过滤条件（适配多值存储场景）

    示例：age_range 存的是 "3-6岁,6-12岁"，
          build_like_condition("age_range", "3-6岁")
          -> 'age_range like "%3-6岁%"'

    :param field_name: Milvus 字段名
    :param value: 待匹配的子串
    :return: 过滤表达式片段；value 为空时返回空字符串
    """
    if not value:
        return ""
    safe_value = escape_milvus_string(value)
    return f'{field_name} like "%{safe_value}%"'
