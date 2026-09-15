"""SenseVoiceSmall 语音识别模型封装（单例）"""
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

from app.conf.asr_config import asr_config
from app.core.logger import logger

_asr_model = None

# 育儿领域高频误识别词：跑通后每发现一条就加一条
_TERM_FIX = {
    "分离焦炉": "分离焦虑",
    "情绪关里": "情绪管理",
}


def get_asr_model() -> AutoModel:
    """获取ASR语音识别模型单例，避免重复加载（首次约 20 秒）"""
    global _asr_model
    if _asr_model is None:
        logger.info(f"正在加载ASR SenseVoiceSmall模型：{asr_config.model_path}")
        _asr_model = AutoModel(
            model=asr_config.model_path,
            device=asr_config.device,
            trust_remote_code=True,                          # SenseVoice 必须开
            vad_model="fsmn-vad",                            # 长音频自动切分
            vad_kwargs={"max_single_segment_time": 30000},   # 单段最长 30 秒
            disable_update=True,                             # 关掉联网自检
        )
        logger.success("ASR语音模型初始化成功")
    return _asr_model


def transcribe(audio_path: str) -> str:
    """音频文件 → 文字（16kHz 单声道 WAV 效果最好）"""
    model = get_asr_model()
    res = model.generate(
        input=audio_path,
        cache={},
        language="zh",           # zh / en / yue / ja / ko / auto，固定 zh 更快
        use_itn=True,            # 逆文本归一化：数字、单位转规范写法
        batch_size_s=60,         # 单次最多处理 60 秒
        merge_vad=True,          # 合并切开的片段，避免答案被切碎
        merge_length_s=15,
    )
    # 原始输出：<|zh|><|NEUTRAL|><|Speech|><|withitn|>三到六岁孩子发脾气怎么办
    # 必须过 postprocess，否则脏字符会流进 node_filter_extract 干扰检索条件提取
    text = rich_transcription_postprocess(res[0]["text"]).strip()

    for wrong, right in _TERM_FIX.items():
        text = text.replace(wrong, right)

    logger.info(f"语音识别完成：{text}")
    return text
