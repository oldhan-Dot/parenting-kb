"""
节点：向量化（node_bge_embedding）

使用 BGE-M3 为每个切片同时生成「稠密向量 + 稀疏向量」。

关键点：向量化时的文本会把内容类型 / 年龄段 / 标题一起拼进去，
让「3-6岁 情绪管理」这类条件词也能参与到语义召回中，明显提升命中率。
"""
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.utils.task_utils import add_running_task, add_done_task

# 批量大小：受显存 / 内存限制，CPU 环境下建议不超过 5
BATCH_SIZE = 5


def build_embedding_text(chunk: dict) -> str:
    """
    构造用于向量化的文本

    格式：【内容类型】【年龄段】切片标题：正文
    """
    content_type = chunk.get("content_type", "") or ""
    age_range = chunk.get("age_range", "") or ""
    title = chunk.get("title", "") or ""
    content = chunk.get("content", "") or ""

    prefix_parts = []
    if content_type:
        prefix_parts.append(f"【{content_type}】")
    if age_range:
        prefix_parts.append(f"【{age_range}】")

    prefix = "".join(prefix_parts)
    return f"{prefix}{title}：{content}"


@step_log("step_1_validate_input")
def step_1_validate_input(state: ImportGraphState):
    """校验输入切片"""
    text_to_embed = state.get("chunks")
    if not isinstance(text_to_embed, list) or not text_to_embed:
        logger.error("向量化输入校验失败：chunks 字段为空或非有效列表")
        raise Exception("错误: 无有效文本切片数据，无法执行向量化处理")
    return text_to_embed


@step_log("step_2_init_model")
def step_2_init_model():
    """初始化 BGE-M3 模型（单例，只加载一次）"""
    try:
        ef = get_bge_m3_ef()
        if ef is None:
            raise ValueError("BGE-M3 模型实例为 None：pymilvus.model 模块未找到或模型加载失败")
        logger.info("BGE-M3 模型实例初始化成功（单例模式）")
        return ef
    except Exception as e:
        error_msg = f"BGE-M3 模型初始化失败：{e}，请检查模型路径 / 环境配置是否正确"
        logger.error(error_msg)
        raise Exception(error_msg)


@step_log("step_3_generate_embeddings")
def step_3_generate_embeddings(texts_to_embed, bge_m3_ef):
    """分批生成向量，并把向量字段写回每个切片"""
    output_data = []
    total = len(texts_to_embed)

    for i in range(0, total, BATCH_SIZE):
        batch_texts = texts_to_embed[i:i + BATCH_SIZE]
        try:
            input_texts = [build_embedding_text(chunk) for chunk in batch_texts]
            docs_embeddings = generate_embeddings(input_texts)

            if not docs_embeddings:
                logger.error("向量化结果为空：请检查输入文本是否为空")
                output_data.extend(batch_texts)
                continue

            for j, doc in enumerate(batch_texts):
                item = doc.copy()
                item["dense_vector"] = docs_embeddings["dense"][j]
                item["sparse_vector"] = docs_embeddings["sparse"][j]
                output_data.append(item)
        except Exception as e:
            # 单批失败不影响整体流程，保留原始切片继续
            logger.error(f"向量化批次处理异常：{e}")
            output_data.extend(batch_texts)
            continue

    return output_data


@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    add_running_task(state["task_id"], "node_bge_embedding")

    texts_to_embed = step_1_validate_input(state)
    bge_m3_ef = step_2_init_model()
    output_data = step_3_generate_embeddings(texts_to_embed, bge_m3_ef)
    state["chunks"] = output_data

    add_done_task(state["task_id"], "node_bge_embedding")
    return state
