"""
SSE（Server-Sent Events）工具模块

基于「会话队列」实现服务端向浏览器的实时推送：
- 进度推送（progress）：展示流水线节点完成情况
- 增量推送（delta）：展示大模型逐字输出
- 完成推送（final）：展示最终完整答案
"""
import json
import queue
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator


class SSEEvent:
    READY = "ready"        # 连接建立
    PROGRESS = "progress"  # 任务节点进度
    DELTA = "delta"        # LLM 流式输出增量
    FINAL = "final"        # 最终完整答案
    ERROR = "error"        # 错误信息
    CLOSE = "__close__"    # 关闭连接信号


# 全局 SSE 会话队列：Key = session_id, Value = queue.Queue
_session_stream: Dict[str, queue.Queue] = {}


def get_sse_queue(session_id: str) -> Optional["queue.Queue"]:
    """获取指定会话的队列"""
    return _session_stream.get(session_id)


def create_sse_queue(session_id: str) -> "queue.Queue":
    """创建并注册一个新的 SSE 队列"""
    q = queue.Queue()
    _session_stream[session_id] = q
    return q


def remove_sse_queue(session_id: str):
    """移除指定会话的队列"""
    _session_stream.pop(session_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """按 SSE 协议打包消息"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_to_session(session_id: str, event: str, data: Dict[str, Any]):
    """通过 session_id 向对应队列推送事件"""
    stream_queue = get_sse_queue(session_id)
    if stream_queue:
        stream_queue.put({"event": event, "data": data})


async def sse_generator(session_id: str, request) -> AsyncGenerator[str, None]:
    """
    SSE 生成器，供 FastAPI 的 StreamingResponse 使用

    :param session_id: 会话ID
    :param request: FastAPI Request 对象，用于检测客户端断开
    """
    stream_queue = get_sse_queue(session_id)
    if stream_queue is None:
        return

    loop = asyncio.get_running_loop()
    try:
        yield _sse_pack(SSEEvent.READY, {})

        while True:
            # 客户端断开则尽快退出，避免空转
            if await request.is_disconnected():
                break

            try:
                # 使用 run_in_executor 避免阻塞事件循环
                msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
            except queue.Empty:
                continue

            event = msg.get("event")
            data = msg.get("data")

            if event == SSEEvent.CLOSE:
                break

            yield _sse_pack(event, data)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        return
    except Exception:
        return
    finally:
        remove_sse_queue(session_id)
