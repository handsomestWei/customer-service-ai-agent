"""
配置文件
包含系统运行所需的各种配置参数
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# OpenAI兼容API配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen3-8B")

# HTTP请求配置
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))
HTTP_MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "3"))
HTTP_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "MultiAgentCustomerService/1.0.0"
}

# 系统配置
SYSTEM_NAME = "多智能体客服系统"
VERSION = "1.0.0"

# 日志配置
LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}
