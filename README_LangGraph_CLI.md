# 🚀 使用 LangGraph CLI 部署多智能体客服系统

基于 [LangGraph CLI 官方文档](https://docs.langchain.com/langgraph-platform/cli#configuration-file)，系统支持通过 LangGraph CLI 进行部署和管理。

## 🎯 主要优势

- **标准化部署**: 使用官方推荐的 LangGraph CLI 工具
- **Docker 支持**: 自动构建和运行 Docker 容器
- **生产就绪**: 支持生产环境的部署和扩展
- **配置驱动**: 通过 `langgraph.json` 配置文件管理所有设置

## 📋 配置文件结构

### langgraph.json 配置说明

```json
{
  "dependencies": ["."],                                    // 项目依赖
  "graphs": {                                               // 工作流图定义
    "customer_service": "./multi_agent_customer_service.py:build_workflow"
  },
  "env": "./.env",                                          // 环境变量文件
  "python_version": "3.11",                                 // Python版本
  "base_image": "langchain/langgraph-api:0.2",             // 基础镜像
  "workflow": {                                             // 工作流详细配置
    "name": "多智能体客服系统",
    "description": "基于LangGraph的多智能体客服系统，支持产品咨询、技术支持、账单处理、投诉处理等",
    "entry_point": "classify_query",
    "nodes": {                                              // 节点配置
      "classify_query": {
        "type": "function",
        "description": "分类客户查询类型",
        "function": "classify_query_node"
      },
      "general_agent": {
        "type": "agent",
        "description": "综合客服智能体",
        "agent": "GeneralAgent",
        "expertise": ["信息查询", "基础服务", "问题转接"]
      },
      "final_response": {
        "type": "function",
        "description": "生成最终响应",
        "function": "final_response_node"
      }
    },
    "edges": {                                              // 边配置
      "conditional": {                                       // 条件边
        "classify_query": {
          "product_info": "product_agent",
          "technical_support": "tech_agent",
          "billing": "billing_agent",
          "complaint": "complaint_agent",
          "general_inquiry": "general_agent"
        }
      },
      "direct": [                                            // 直接边
        ["product_agent", "final_response"],
        ["tech_agent", "final_response"],
        ["billing_agent", "final_response"],
        ["complaint_agent", "final_response"],
        ["general_agent", "final_response"]
      ]
    },
    "end_point": "final_response"
  },
  "store": {                                                // 存储配置
    "ttl": {
      "refresh_on_read": true,                              // 读取时刷新TTL
      "sweep_interval_minutes": 60,                         // 清理间隔
      "default_ttl": 10080                                  // 默认TTL (7天)
    }
  },
  "checkpointer": {                                         // 检查点配置
    "ttl": {
      "strategy": "delete",                                 // 过期策略
      "sweep_interval_minutes": 10,                         // 检查间隔
      "default_ttl": 43200                                  // 默认TTL (30天)
    }
  },
  "http": {                                                 // HTTP服务器配置
    "port": 8123,                                           // 端口
    "host": "0.0.0.0"                                      // 主机
  }
}
```

## 🔧 安装和配置

### 1. 安装 LangGraph CLI

```bash
# 安装基础版本
pip install langgraph-cli

# 安装开发版本（支持 dev 命令）
pip install -U "langgraph-cli[inmem]"
```

### 2. 验证安装
在win环境中，langgraph-cli下载后，需要将`langgraph.exe`路径加入PATH环境变量。或者使用时直接带全路径，例`C:\Users\<yourName>\AppData\Roaming\Python\Python313\Scripts\langgraph.exe`
```bash
langgraph --help
```

### 3. 环境配置

确保你有以下文件：
- `langgraph.json` - LangGraph CLI 配置文件
- `.env` - 环境变量文件（包含API密钥）
- `requirements.txt` - Python 依赖文件
- `agents/` - 智能体模块目录
- `tools/` - 工具函数模块目录

**环境变量配置示例：**
```env
# OpenAI兼容API配置
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_MODEL=Qwen/Qwen3-8B

# HTTP配置
HTTP_TIMEOUT=30
HTTP_MAX_RETRIES=3

# 日志配置
LOG_LEVEL=INFO
```

## 🚀 使用方法

### 开发模式 (dev)

```bash
# 启动开发服务器（支持热重载）
langgraph dev

# 指定配置文件
langgraph dev -c langgraph.json

# 指定端口和主机
langgraph dev --port 8000 --host 0.0.0.0

# 启用调试
langgraph dev --debug-port 5678 --wait-for-client
```

### 构建 Docker 镜像

```bash
# 构建 Docker 镜像
langgraph build -t my-customer-service:latest

# 指定平台
langgraph build --platform linux/amd64,linux/arm64 -t my-customer-service:latest

# 使用本地镜像（不拉取最新版本）
langgraph build --no-pull -t my-customer-service:latest
```

### 启动服务 (up)

```bash
# 启动 LangGraph API 服务器
langgraph up

# 指定端口
langgraph up -p 8000

# 等待服务启动
langgraph up --wait

# 使用本地构建的镜像
langgraph up --image my-customer-service:latest
```

### 生成 Dockerfile

```bash
# 生成 Dockerfile
langgraph dockerfile -c langgraph.json Dockerfile

# 查看生成的 Dockerfile 内容
cat Dockerfile
```

## 🌐 API 端点

接口文档访问地址默认`http://127.0.0.1:2024/docs`，内嵌了js需要挂梯子。也可参考`https://langchain-ai.github.io/langgraph/cloud/reference/api/api_ref.html`

启动服务后，系统会提供以下主要的API端点：
### Assistants 管理
- POST /assistants - 创建助手
- POST /assistants/search - 搜索助手
- GET /assistants/{assistant_id} - 获取助手详情
- DELETE /assistants/{assistant_id} - 删除助手
- PATCH /assistants/{assistant_id} - 更新助手
### Threads 管理
- POST /threads - 创建线程
- POST /threads/search - 搜索线程
- GET /threads/{thread_id} - 获取线程详情
- DELETE /threads/{thread_id} - 删除线程
- PATCH /threads/{thread_id} - 更新线程
### Thread Runs 执行
- POST /threads/{thread_id}/runs - 在线程上执行运行
- POST /threads/{thread_id}/runs/stream - 流式执行
- GET /threads/{thread_id}/runs/{run_id} - 获取运行状态
- POST /threads/{thread_id}/runs/{run_id}/cancel - 取消运行
### Stateless Runs 执行
- POST /runs - 无状态执行
- POST /runs/stream - 无状态流式执行
- POST /runs/batch - 批量执行
### 系统状态
- GET /info - 服务器信息
- GET /ok - 健康检查
- GET /metrics - 系统指标

## 🔍 监控和调试

### 开发模式特性

- **热重载**: 代码修改后自动重启服务
- **调试支持**: 支持断点调试
- **实时日志**: 查看详细的执行日志
- **Studio 集成**: 自动连接到 LangGraph Studio

### 生产模式特性

- **Docker 容器**: 隔离的运行环境
- **健康检查**: 自动健康状态监控
- **日志管理**: 结构化的日志输出
- **性能监控**: 内置的性能指标

## 🐳 Docker 部署

### 构建镜像

```bash
# 构建生产镜像
langgraph build -t customer-service:latest

# 查看构建的镜像
docker images | grep customer-service
```

### 运行容器

```bash
# 运行容器
docker run -d \
  --name customer-service \
  -p 2024:2024 \
  -e OPENAI_API_KEY=your_key \
  customer-service:latest

# 查看容器状态
docker ps | grep customer-service

# 查看日志
docker logs customer-service
```

### Docker Compose

创建 `docker-compose.yml` 文件：

```yaml
version: '3.8'
services:
  customer-service:
    build: .
    ports:
      - "2024:2024"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./.env:/app/.env
    restart: unless-stopped
```

运行服务：

```bash
docker-compose up -d
```

## 📚 相关资源

- [LangGraph CLI 官方文档](https://docs.langchain.com/langgraph-platform/cli#configuration-file)
- [LangGraph Platform 概述](https://docs.langchain.com/langgraph-platform/)
- [使用langgraph创建模版项目](https://docs.langchain.com/langgraph-platform/local-server)
- [LangGraph Studio](https://smith.langchain.com/)
