"""
MongoDB 会话历史读写工具

集合：chat_message
文档结构：
{
    session_id: 会话ID,
    role: "user" / "assistant",
    text: 消息正文,
    rewritten_query: 改写后的问题（用于指代消解）,
    filter_tags: 该轮问答涉及的检索条件标签（如 ["3-6岁", "情绪管理"]）,
    ts: 时间戳
}
"""
import os
from datetime import datetime
from typing import List, Dict, Any

from bson import ObjectId
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

from app.core.logger import logger

load_dotenv()


class HistoryMongoTool:
    """MongoDB 会话历史读写工具类"""

    def __init__(self):
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")

            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self.chat_message = self.db["chat_message"]

            # 复合索引：session_id 升序 + ts 降序，适配「按会话查最新记录」
            self.chat_message.create_index([("session_id", 1), ("ts", -1)])

            logger.info(f"成功连接 MongoDB：{self.db_name}")
        except Exception as e:
            logger.error(f"连接 MongoDB 失败：{e}")
            raise


_history_mongo_tool = None


def get_history_mongo_tool() -> HistoryMongoTool:
    """获取 HistoryMongoTool 单例（懒加载）"""
    global _history_mongo_tool
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    return _history_mongo_tool


def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        filter_tags: List[str] = None,
        message_id: str = None,
) -> str:
    """
    写入 / 更新单条会话记录

    :param message_id: 有值则按主键更新，无值则新增
    :return: 记录主键（新增返回 ObjectId 字符串，更新返回入参 message_id）
    """
    ts = datetime.now().timestamp()

    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query or "",
        "filter_tags": filter_tags,
        "ts": ts,
    }

    mongo_tool = get_history_mongo_tool()
    if message_id:
        mongo_tool.chat_message.update_one(
            {"_id": ObjectId(message_id)},
            {"$set": document},
        )
        return message_id
    else:
        result = mongo_tool.chat_message.insert_one(document)
        return str(result.inserted_id)


def update_message_filter_tags(ids: List[str], filter_tags: List[str]) -> int:
    """
    批量回填历史记录的检索条件标签

    仅更新 filter_tags 为空 / 不存在的记录，避免覆盖已有信息

    :return: 实际更新的文档数量
    """
    mongo_tool = get_history_mongo_tool()
    try:
        object_ids = [ObjectId(i) for i in ids]
        result = mongo_tool.chat_message.update_many(
            {
                "_id": {"$in": object_ids},
                "$or": [
                    {"filter_tags": {"$exists": False}},
                    {"filter_tags": []},
                    {"filter_tags": None},
                ],
            },
            {"$set": {"filter_tags": filter_tags}},
        )
        logger.info(f"已回填 {result.modified_count} 条历史记录的检索条件：{filter_tags}")
        return result.modified_count
    except Exception as e:
        logger.error(f"回填历史记录检索条件失败：{e}")
        return 0


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    查询指定会话的最近 N 条记录（按时间正序，可直接作为 LLM 上下文）

    :return: 记录列表；查询失败返回空列表
    """
    mongo_tool = get_history_mongo_tool()
    try:
        cursor = mongo_tool.chat_message.find({"session_id": session_id}).sort("ts", ASCENDING).limit(limit)
        return list(cursor)
    except Exception as e:
        logger.error(f"查询会话历史失败：{e}")
        return []


def clear_history(session_id: str) -> int:
    """
    清空指定会话的全部记录

    :return: 实际删除的文档数量
    """
    mongo_tool = get_history_mongo_tool()
    try:
        result = mongo_tool.chat_message.delete_many({"session_id": session_id})
        logger.info(f"已清空会话 {session_id} 的 {result.deleted_count} 条记录")
        return result.deleted_count
    except Exception as e:
        logger.error(f"清空会话历史失败：{e}")
        return 0
