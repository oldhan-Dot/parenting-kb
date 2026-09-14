from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class LLMConfig:
    base_url: str          # OpenAI 兼容接口地址
    api_key: str           # API Key
    lv_model: str          # 视觉模型（图片摘要用）
    llm_model: str         # 默认对话/抽取模型
    llm_temperature: float


lm_config = LLMConfig(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    lv_model=os.getenv("VL_MODEL"),
    llm_model=os.getenv("LLM_DEFAULT_MODEL"),
    llm_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE") or 0.1),
)
