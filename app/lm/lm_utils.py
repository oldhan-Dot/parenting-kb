"""
大模型客户端封装

统一使用 OpenAI 兼容接口（百炼 / 其他代理均可），
带全局缓存避免重复初始化，并支持 JSON 输出模式。
"""
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.exceptions import LangChainException

from app.conf.lm_config import lm_config
from app.core.logger import logger

# 全局缓存：键为 (模型名, JSON模式)，值为 ChatOpenAI 实例
_llm_client_cache = {}


def get_llm_client(model: Optional[str] = None, json_mode: bool = False) -> ChatOpenAI:
    """
    获取带缓存的 LangChain ChatOpenAI 客户端实例

    :param model: 模型名称，优先级：传入参数 > 配置 lm_config.llm_model > 内置默认
    :param json_mode: 是否开启 JSON 输出模式（结构化抽取场景使用）
    :return: ChatOpenAI 实例
    """
    target_model = model or lm_config.llm_model or "qwen-flash"
    cache_key = (target_model, json_mode)

    if cache_key in _llm_client_cache:
        logger.debug(f"[LLM客户端] 缓存命中：模型={target_model}，JSON模式={json_mode}")
        return _llm_client_cache[cache_key]

    if not lm_config.api_key:
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置 OPENAI_API_KEY")
    if not lm_config.base_url:
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置 OPENAI_BASE_URL")

    logger.info(f"[LLM客户端] 初始化新实例：模型={target_model}，JSON模式={json_mode}")

    # 国产模型私有参数透传（关闭思考链输出，减少冗余内容）
    extra_body = {"enable_thinking": False}
    model_kwargs = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    try:
        llm_client = ChatOpenAI(
            model=target_model,
            temperature=lm_config.llm_temperature or 0.1,
            api_key=lm_config.api_key,
            base_url=lm_config.base_url,
            extra_body=extra_body,
            model_kwargs=model_kwargs,
        )
    except LangChainException as e:
        raise Exception(f"[LLM客户端] 模型【{target_model}】初始化失败（LangChain层）：{str(e)}") from e

    _llm_client_cache[cache_key] = llm_client
    logger.info(f"[LLM客户端] 实例初始化成功并缓存：模型={target_model}")
    return llm_client
