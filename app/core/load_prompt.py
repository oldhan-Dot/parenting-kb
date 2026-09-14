from pathlib import Path

from app.core.logger import logger
from app.utils.path_util import PROJECT_ROOT


def load_prompt(name: str, **kwargs) -> str:
    """
    加载提示词并渲染变量占位符

    :param name: 提示词文件名（不带 .prompt 后缀，如 answer_out）
    :param kwargs: 需要渲染的变量键值对（键名必须与 .prompt 中的占位符一致）
    :return: 渲染后的最终提示词字符串
    """
    prompt_path = PROJECT_ROOT / "prompts" / f"{name}.prompt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"提示词文件不存在：{prompt_path.absolute()}")

    raw_prompt = prompt_path.read_text(encoding="utf-8")

    if kwargs:
        # 提示词里可能出现 JSON 花括号，统一用双花括号转义后再格式化
        rendered_prompt = raw_prompt.format(**kwargs)
        logger.debug(f"提示词渲染成功，替换变量：{list(kwargs.keys())}")
        return rendered_prompt

    return raw_prompt
