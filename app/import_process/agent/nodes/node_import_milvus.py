"""
节点：导入向量库（node_import_milvus）

职责：
1. 集合不存在则按育儿元数据 schema 创建
2. 按 source_file 删除旧数据（同一份文档重复导入时保证幂等）
3. 批量插入带向量的切片数据，并回显主键
"""
from pymilvus import DataType

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState, CHUNK_SCALAR_FIELDS
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_running_task, add_done_task

# 切片集合名称（从配置读取，便于环境切换）
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection


@step_log("step_1_prepare_collection")
def step_1_prepare_collection(milvus_client):
    """集合不存在则创建 schema 与索引"""
    if milvus_client.has_collection(collection_name=CHUNKS_COLLECTION_NAME):
        return milvus_client

    schema = milvus_client.create_schema(
        auto_id=True,             # 主键自增
        enable_dynamic_field=True,
    )

    # --- 主键与标量字段（对齐需求说明「内容要求」的字段清单） ---
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="source_file", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="source_path", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="age_range", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="problem_type", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="scene", datatype=DataType.VARCHAR, max_length=4096)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="part", datatype=DataType.INT8)
    schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)

    # --- 向量字段 ---
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # --- 索引 ---
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type="HNSW",
        metric_type="COSINE",
        params={
            "M": 32,               # 图中每个节点的最大连接数，越大召回越高、内存越多
            "efConstruction": 300, # 建索引时的候选节点数，越大图质量越好
        },
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_vector_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )

    milvus_client.create_collection(
        collection_name=CHUNKS_COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    logger.info(f"Milvus 集合 {CHUNKS_COLLECTION_NAME} 已创建")
    return milvus_client


@step_log("step_2_delete_old_data")
def step_2_delete_old_data(milvus_client, source_file: str):
    """
    按来源文件名删除旧数据，保证同一份文档重复导入时是「替换」而不是「追加」
    """
    if not source_file:
        logger.warning("source_file 为空，跳过旧数据清理")
        return

    safe_source_file = escape_milvus_string(source_file)
    milvus_client.delete(
        collection_name=CHUNKS_COLLECTION_NAME,
        filter=f'source_file == "{safe_source_file}"',
    )
    # 重新加载集合，确保删除真正生效，避免新旧数据混杂
    milvus_client.load_collection(collection_name=CHUNKS_COLLECTION_NAME)
    logger.info(f"已清理 source_file = {source_file} 的旧数据")


def _build_row(chunk: dict) -> dict:
    """把切片整理成 Milvus 插入要求的字段结构（字段类型严格对齐 schema）"""
    row = {}
    for field in CHUNK_SCALAR_FIELDS:
        value = chunk.get(field, "")
        if value is None:
            value = ""
        if field == "part":
            row[field] = int(value or 1)
        else:
            row[field] = str(value)

    row["dense_vector"] = chunk["dense_vector"]
    row["sparse_vector"] = chunk["sparse_vector"]
    return row


@step_log("step_3_insert_collections")
def step_3_insert_collections(milvus_client, chunks: list):
    """批量写入切片数据，并回显主键"""
    rows = [_build_row(chunk) for chunk in chunks]

    insert_result = milvus_client.insert(collection_name=CHUNKS_COLLECTION_NAME, data=rows)
    insert_count = insert_result.get("insert_count", 0)
    logger.info(f"数据插入完成，成功插入 {insert_count} 条")

    ids = insert_result.get("ids", [])
    if ids and len(ids) == len(chunks):
        for index, chunk in enumerate(chunks):
            chunk["chunk_id"] = ids[index]

    return chunks


@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    add_running_task(state["task_id"], "node_import_milvus")

    chunks = state.get("chunks")
    if not chunks:
        logger.error("node_import_milvus: chunks 数据不存在")
        raise ValueError("node_import_milvus: chunks 数据不存在")

    # 校验向量是否都已生成
    valid_chunks = [c for c in chunks if c.get("dense_vector") and c.get("sparse_vector")]
    if len(valid_chunks) != len(chunks):
        logger.warning(f"存在 {len(chunks) - len(valid_chunks)} 条切片未生成向量，将跳过这些切片")

    if not valid_chunks:
        raise ValueError("所有切片都未生成向量，无法写入向量库")

    milvus_client = get_milvus_client()
    if milvus_client is None:
        raise ConnectionError("Milvus 客户端初始化失败，请检查 MILVUS_URL 配置与服务状态")

    step_1_prepare_collection(milvus_client)
    step_2_delete_old_data(milvus_client, state.get("source_file", ""))
    with_id_chunks = step_3_insert_collections(milvus_client, valid_chunks)
    state["chunks"] = with_id_chunks

    add_done_task(state["task_id"], "node_import_milvus")
    return state
