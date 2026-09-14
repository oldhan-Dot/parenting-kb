import time
from typing import Deque

from app.core.logger import logger


def apply_api_rate_limit(
        request_times: Deque[float],
        max_requests: int,
        window_seconds: int = 60,
) -> None:
    """
    通用滑动窗口限流器

    维护请求时间戳双端队列，窗口内请求数达到上限时自动阻塞等待，
    防止批量调用第三方 API（如视觉模型）时触发限流。

    :param request_times: 存储请求时间戳的双端队列（需外部初始化并跨调用复用）
    :param max_requests: 窗口内最大允许请求次数
    :param window_seconds: 滑动窗口时长（秒）
    """
    current_time = time.time()

    # 1. 清理窗口外的过期时间戳
    while request_times and current_time - request_times[0] >= window_seconds:
        request_times.popleft()

    # 2. 达到上限则阻塞等待
    if len(request_times) >= max_requests:
        sleep_duration = window_seconds - (current_time - request_times[0])
        if sleep_duration > 0:
            logger.debug(
                f"触发API速率限制，窗口{window_seconds}秒内最多{max_requests}次，需等待：{sleep_duration:.2f} 秒"
            )
            time.sleep(sleep_duration)
            current_time = time.time()
            while request_times and current_time - request_times[0] >= window_seconds:
                request_times.popleft()

    # 3. 记录本次请求时间戳
    request_times.append(current_time)
