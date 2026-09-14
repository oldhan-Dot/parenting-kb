"""
育儿知识库 - 导入服务（端口 8000）

接口：
- GET  /import.html          上传页面
- POST /upload               上传文件（支持多文件），后台异步跑导入流水线
- GET  /status/{task_id}     查询任务进度（已完成/正在进行的节点）
- GET  /stats               查看向量库当前数据量（自检用）
"""
import shutil
import uuid
from datetime import datetime
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

from app.clients.milvus_utils import get_collection_stats
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import create_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    update_task_status,
    TASK_STATUS_PROCESSING,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    add_done_task,
    add_running_task,
    get_running_task_list,
    get_task_status,
    get_done_task_list,
)

app = FastAPI(title="parenting-kb-import", description="育儿知识库导入服务")

# 解决跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/import.html")
def import_file():
    """返回导入页面"""
    import_file_path = PROJECT_ROOT / "app/import_process/page/import.html"
    if not import_file_path.exists():
        raise HTTPException(status_code=404, detail="import.html 不存在")
    return FileResponse(str(import_file_path))


def run_graph_task(task_id, local_file_path, local_dir):
    """后台任务：执行导入流水线"""
    try:
        logger.info(f"导入流程开始，task_id={task_id}")
        update_task_status(task_id, TASK_STATUS_PROCESSING)

        init_state = create_default_state(
            task_id=task_id,
            local_file_path=local_file_path,
            local_dir=local_dir,
        )

        # 流式执行图，实时记录每个节点的完成情况
        results = kb_import_app.stream(init_state)
        for result in results:
            for node_name, _node_result in result.items():
                add_done_task(task_id, node_name)

        logger.info(f"导入流程成功，task_id={task_id}")
        update_task_status(task_id, TASK_STATUS_COMPLETED)
    except Exception as e:
        logger.error(f"导入流程失败，task_id={task_id}，原因：{e}")
        update_task_status(task_id, TASK_STATUS_FAILED)


@app.post("/upload")
async def upload_files(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...),
):
    """
    文件上传接口

    1. 为每个文件生成全局唯一 task_id
    2. 落到 output/年月日/task_id/文件名
    3. 挂后台任务执行导入流水线
    """
    today_str = datetime.now().strftime("%Y%m%d")
    today_str_dir = PROJECT_ROOT / "output" / today_str

    task_ids = []
    for file in files:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)

        add_running_task(task_id, "upload_file")

        upload_dir_path = today_str_dir / task_id
        upload_dir_path.mkdir(parents=True, exist_ok=True)
        upload_file_path = upload_dir_path / file.filename

        with upload_file_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        add_done_task(task_id, "upload_file")

        background_tasks.add_task(
            run_graph_task,
            task_id,
            str(upload_file_path),
            str(upload_dir_path),
        )

    return {
        "code": 200,
        "message": "upload success",
        "task_ids": task_ids,
    }


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """查询任务进度"""
    return {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),
        "running_list": get_running_task_list(task_id),
        "done_list": get_done_task_list(task_id),
    }


@app.get("/stats")
async def stats():
    """查看向量库当前数据量，便于导入后自检"""
    return {
        "code": 200,
        "collection": milvus_config.chunks_collection,
        "stats": get_collection_stats(milvus_config.chunks_collection),
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.import_process.api.file_import_service:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
