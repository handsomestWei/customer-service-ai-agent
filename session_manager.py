"""
会话管理器
使用 LangChain 标准的 Memory 接口实现会话管理
支持多种存储后端和统一的 API 接口
"""

import os
import time
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# LangChain Memory 相关导入
from langchain_core.memory import BaseMemory
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# 存储后端导入
from langchain_community.chat_message_histories import (
    RedisChatMessageHistory,
    MongoDBChatMessageHistory,
    PostgresChatMessageHistory,
    FileChatMessageHistory
)

class LangChainSessionManager:
    """
    基于 LangChain Memory 的会话管理器
    提供统一的会话管理接口，支持多种存储后端
    """

    def __init__(self, storage_backend: str = "memory", **storage_config):
        """
        初始化会话管理器

        Args:
            storage_backend: 存储后端类型 ("memory", "redis", "mongodb", "postgres", "file")
            **storage_config: 存储后端配置参数
        """
        self.storage_backend = storage_backend
        self.storage_config = storage_config
        self.sessions: Dict[str, Dict[str, Any]] = {}

        # 验证存储后端配置
        self._validate_storage_config()

        print(f"📱 会话管理器初始化完成，使用 {storage_backend} 后端")

    def _validate_storage_config(self):
        """验证存储后端配置"""
        if self.storage_backend == "redis":
            if "url" not in self.storage_config:
                self.storage_config["url"] = "redis://localhost:6379"
        elif self.storage_backend == "mongodb":
            if "connection_string" not in self.storage_config:
                self.storage_config["connection_string"] = "mongodb://localhost:27017"
        elif self.storage_backend == "postgres":
            if "connection_string" not in self.storage_config:
                self.storage_config["connection_string"] = "postgresql://localhost:5432"
        elif self.storage_backend == "file":
            if "storage_dir" not in self.storage_config:
                self.storage_config["storage_dir"] = "./chat_sessions"
                os.makedirs(self.storage_config["storage_dir"], exist_ok=True)

    def _create_memory_backend(self, session_id: str) -> ConversationBufferMemory:
        """
        根据存储后端创建 Memory 实例

        Args:
            session_id: 会话ID

        Returns:
            ConversationBufferMemory 实例
        """
        if self.storage_backend == "memory":
            # 内存存储（默认）
            return ConversationBufferMemory(
                memory_key="conversation_history",
                return_messages=True,
                output_key="response"
            )

        elif self.storage_backend == "redis":
            # Redis 存储
            history = RedisChatMessageHistory(
                session_id=session_id,
                url=self.storage_config["url"]
            )
            return ConversationBufferMemory(
                chat_memory=history,
                return_messages=True,
                output_key="response"
            )

        elif self.storage_backend == "mongodb":
            # MongoDB 存储
            history = MongoDBChatMessageHistory(
                session_id=session_id,
                connection_string=self.storage_config["connection_string"]
            )
            return ConversationBufferMemory(
                chat_memory=history,
                return_messages=True,
                output_key="response"
            )

        elif self.storage_backend == "postgres":
            # PostgreSQL 存储
            history = PostgresChatMessageHistory(
                session_id=session_id,
                connection_string=self.storage_config["connection_string"]
            )
            return ConversationBufferMemory(
                chat_memory=history,
                return_messages=True,
                output_key="response"
            )

        elif self.storage_backend == "file":
            # 文件存储
            file_path = os.path.join(
                self.storage_config["storage_dir"],
                f"{session_id}.json"
            )
            history = FileChatMessageHistory(file_path)
            return ConversationBufferMemory(
                chat_memory=history,
                return_messages=True,
                output_key="response"
            )

        else:
            raise ValueError(f"不支持的存储后端: {self.storage_backend}")

    def create_session(self, session_id: str = None) -> str:
        """
        创建新的会话

        Args:
            session_id: 会话ID，如果为None则自动生成

        Returns:
            会话ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        # 创建 Memory 后端
        memory = self._create_memory_backend(session_id)

        # 记录会话元数据
        self.sessions[session_id] = {
            "memory": memory,
            "created_at": time.time(),
            "last_activity": time.time(),
            "message_count": 0,
            "storage_backend": self.storage_backend
        }

        print(f"📱 Created new session: {session_id} (using {self.storage_backend} backend)")

        return session_id

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话信息

        Args:
            session_id: 会话ID

        Returns:
            会话信息字典
        """

        if session_id not in self.sessions:
            self.create_session(session_id)

        # 更新最后活动时间
        self.sessions[session_id]["last_activity"] = time.time()
        session = self.sessions[session_id]

        return session

    def get_memory(self, session_id: str) -> ConversationBufferMemory:
        """
        获取会话的 Memory 实例

        Args:
            session_id: 会话ID

        Returns:
            ConversationBufferMemory 实例
        """
        session = self.get_session(session_id)

        if not isinstance(session, dict):
            self.create_session(session_id)
            session = self.sessions[session_id]

        return session["memory"]

    def add_message(self, session_id: str, message: str, is_user: bool = True):
        """
        添加消息到会话历史

        Args:
            session_id: 会话ID
            message: 消息内容
            is_user: 是否为用户消息
        """
        session = self.get_session(session_id)

        if not isinstance(session, dict):
            print(f"❌ Error: session is not a dict for session_id {session_id}, type: {type(session)}")
            return

        memory = session["memory"]

        # 使用 LangChain 标准接口添加消息
        if is_user:
            memory.chat_memory.add_user_message(message)
        else:
            memory.chat_memory.add_ai_message(message)

        # 更新统计信息
        session["message_count"] += 1
        session["last_activity"] = time.time()

        print(f"📝 Session {session_id} added {'user' if is_user else 'AI'} message")

    def get_conversation_history(self, session_id: str) -> List[BaseMessage]:
        """
        获取对话历史（LangChain 标准格式）

        Args:
            session_id: 会话ID

        Returns:
            LangChain 消息列表
        """
        memory = self.get_memory(session_id)
        return memory.chat_memory.messages

    def get_conversation_context(self, session_id: str, max_messages: int = 10) -> List[Dict[str, Any]]:
        """
        获取对话上下文（带时间戳的格式）

        Args:
            session_id: 会话ID
            max_messages: 最大消息数量

        Returns:
            带时间戳的消息列表
        """
        messages = self.get_conversation_history(session_id)

        # 转换为带时间戳的格式
        formatted_messages = []
        for msg in messages[-max_messages:]:
            message_data = {
                "content": msg.content,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "is_user": isinstance(msg, HumanMessage),
                "message_type": msg.__class__.__name__
            }
            formatted_messages.append(message_data)

        return formatted_messages

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话详细信息

        Args:
            session_id: 会话ID

        Returns:
            会话信息字典
        """
        if session_id not in self.sessions:
            return {}

        session = self.sessions[session_id]

        if not isinstance(session, dict):
            return {}

        memory = session["memory"]

        return {
            "session_id": session_id,
            "message_count": len(memory.chat_memory.messages),
            "created_at": session["created_at"],
            "last_activity": session["last_activity"],
            "storage_backend": session["storage_backend"],
            "memory_type": type(memory).__name__
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        列出所有会话信息

        Returns:
            会话信息列表
        """
        return [
            self.get_session_info(session_id)
            for session_id in self.sessions.keys()
        ]

    def clear_session(self, session_id: str):
        """
        清空会话内容

        Args:
            session_id: 会话ID
        """
        if session_id in self.sessions:
            memory = self.sessions[session_id]["memory"]
            memory.clear()

            # 重置统计信息
            self.sessions[session_id]["message_count"] = 0
            self.sessions[session_id]["last_activity"] = time.time()

            print(f"🧹 Cleared session: {session_id}")

    def delete_session(self, session_id: str):
        """
        删除会话

        Args:
            session_id: 会话ID
        """
        if session_id in self.sessions:
            # 清空 Memory
            memory = self.sessions[session_id]["memory"]
            memory.clear()

            # 删除会话记录
            del self.sessions[session_id]

            print(f"🗑️ Deleted session: {session_id}")

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        清理过期会话

        Args:
            max_age_hours: 最大存活时间（小时）

        Returns:
            清理的会话数量
        """
        current_time = time.time()
        expired_sessions = []

        for session_id, session_data in self.sessions.items():
            age_hours = (current_time - session_data["last_activity"]) / 3600
            if age_hours > max_age_hours:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            self.delete_session(session_id)

        if expired_sessions:
            print(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")

        return len(expired_sessions)

    def get_conversation_summary(self, session_id: str) -> str:
        """
        获取对话摘要

        Args:
            session_id: 会话ID

        Returns:
            对话摘要
        """
        messages = self.get_conversation_history(session_id)

        if not messages:
            return "暂无对话记录"

        user_messages = [msg.content for msg in messages if isinstance(msg, HumanMessage)]
        ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]

        summary = f"对话摘要 (会话ID: {session_id})\n"
        summary += f"总消息数: {len(messages)}\n"
        summary += f"用户消息: {len(user_messages)}\n"
        summary += f"AI回复: {len(ai_messages)}\n"

        if user_messages:
            summary += f"最新用户消息: {user_messages[-1][:100]}...\n"

        return summary

    def export_session(self, session_id: str) -> Dict[str, Any]:
        """
        导出会话数据

        Args:
            session_id: 会话ID

        Returns:
            会话数据字典
        """
        if session_id not in self.sessions:
            return {}

        session = self.sessions[session_id]
        memory = session["memory"]

        return {
            "session_info": self.get_session_info(session_id),
            "messages": [
                {
                    "content": msg.content,
                    "type": msg.__class__.__name__,
                    "timestamp": datetime.now().isoformat()
                }
                for msg in memory.chat_memory.messages
            ]
        }

# 创建默认实例（使用内存存储）
default_session_manager = LangChainSessionManager()

# 便捷函数
def create_session(session_id: str = None) -> str:
    """创建新会话的便捷函数"""
    return default_session_manager.create_session(session_id)

def get_session(session_id: str) -> Dict[str, Any]:
    """获取会话信息的便捷函数"""
    return default_session_manager.get_session(session_id)

def add_message(session_id: str, message: str, is_user: bool = True):
    """添加消息的便捷函数"""
    default_session_manager.add_message(session_id, message, is_user)

def get_conversation_context(session_id: str, max_messages: int = 10) -> List[Dict[str, Any]]:
    """获取对话上下文的便捷函数"""
    return default_session_manager.get_conversation_context(session_id, max_messages)

def list_sessions() -> List[Dict[str, Any]]:
    """列出所有会话的便捷函数"""
    return default_session_manager.list_sessions()

def clear_session(session_id: str):
    """清空会话的便捷函数"""
    default_session_manager.clear_session(session_id)

def delete_session(session_id: str):
    """删除会话的便捷函数"""
    default_session_manager.delete_session(session_id)
