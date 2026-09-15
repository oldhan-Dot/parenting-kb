from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ASRConfig:
    model_path: str    # 本地模型路径
    device: str   # 运行设备



asr_config = ASRConfig(
    model_path=os.getenv("ASR_MODEL_PATH"),
    device=os.getenv("ASR_DEVICE"),
)
