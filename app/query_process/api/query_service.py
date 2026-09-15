"""
育儿知识库 - 查询服务（端口 8001）

接口：
- GET    /chat.html            对话页面
- POST   /query                提问（流式 / 非流式）
- POST   /asr                  语音转文字（语音交互）
- GET    /stream/{session_id}  流式输出（SSE）
- GET    /history/{session_id} 查询历史对话
- DELETE /delete/{session_id}  清空会话历史
- GET    /health               健康检查
"""
import shutil
import uuid
from pathlib import Path

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from app.clients.mongo_history_utils import clear_history, get_recent_messages
from app.core.logger import logger
from app.lm.asr_utils import transcribe
from app.query_process.agent.main_graph import kb_query_app
from app.query_process.agent.state import create_query_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.sse_utils import SSEEvent, create_sse_queue, push_to_session, sse_generator
from app.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    get_task_result,
    update_task_status,
)

app = FastAPI(title="parenting-kb-query", description="育儿知识库查询服务")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """问答接口入参"""
    query: str = Field(..., description="家长的问题")
    session_id: str = Field(None, description="会话ID，不传则新建会话")
    is_stream: bool = Field(False, description="是否流式返回")


@app.get("/chat.html")
def chat():
    """返回对话页面"""
    chat_file_path = Path(__file__).parent.parent / "page" / "chat.html"
    if not chat_file_path.exists():
        raise HTTPException(status_code=404, detail="chat.html 页面不存在")
    return FileResponse(str(chat_file_path))


def run_query_graph(session_id: str, query: str, is_stream: bool):
    """后台任务：执行查询流水线"""
    init_state = create_query_default_state(
        session_id=session_id,
        original_query=query,
        is_stream=is_stream,
    )

    try:
        kb_query_app.invoke(init_state)
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    except Exception as e:
        logger.error(f"会话 {session_id} 查询失败：{e}")
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
            push_to_session(session_id, SSEEvent.CLOSE, {})


@app.post("/query")
async def query(background_task: BackgroundTasks, request: QueryRequest):
    """处理家长提问"""
    session_id = request.session_id or str(uuid.uuid4())
    user_query = request.query
    is_stream = request.is_stream

    if is_stream:
        # 流式模式：先建队列，再挂后台任务，前端随后连 /stream
        create_sse_queue(session_id)

    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)

    if is_stream:
        background_task.add_task(run_query_graph, session_id, user_query, is_stream)
        return {
            "message": "结果正在处理中",
            "session_id": session_id,
        }

    # 非流式模式：同步执行后直接返回答案
    run_query_graph(session_id, user_query, is_stream)
    answer = get_task_result(session_id, "answer", "")
    return {
        "message": "处理完成！",
        "session_id": session_id,
        "answer": answer,
    }


@app.post("/asr")
async def speech_to_text(file: UploadFile = File(...)):
    """语音转文字：接收音频文件，返回识别文字"""
    # 保留原扩展名：解码后端会按后缀/容器格式判断（前端录的是 .wav）
    suffix = Path(file.filename or "").suffix or ".wav"
    tmp_path = PROJECT_ROOT / "output" / f"asr_{uuid.uuid4().hex}{suffix}"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        if tmp_path.stat().st_size == 0:
            raise ValueError("上传的音频文件为空")
        text = transcribe(str(tmp_path))
    except Exception as e:
        logger.error(f"语音识别失败：{e}")
        raise HTTPException(status_code=500, detail=f"语音识别失败：{e}")
    finally:
        # 音频不落盘，识别完立即删除（语音涉及隐私）
        tmp_path.unlink(missing_ok=True)

    return {"text": text}


@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    """SSE 流式输出"""
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 10):
    """查询最近的历史对话"""
    try:
        history_list = get_recent_messages(session_id, limit)
        for item in history_list:
            item["_id"] = str(item["_id"])
        return {"session_id": session_id, "items": history_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


@app.delete("/delete/{session_id}")
async def clear_chat_history(session_id: str):
    """清空会话历史"""
    count = clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}


@app.get("/health")
def health():
    """健康检查"""
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(
        "app.query_process.api.query_service:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
    )
