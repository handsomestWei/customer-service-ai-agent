#!/usr/bin/env python3
"""
多智能体客服系统 - Web应用
基于 LangGraph API 接口，提供Web界面，支持持续的客户对话和会话管理
"""

import os
import json
import time
import requests
from typing import Dict, List, Any
from flask import Flask, render_template, request, jsonify, session, Response
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入配置
from config import *

app = Flask(__name__)

# LangGraph API 配置
LANGGRAPH_API_URL = "http://127.0.0.1:2024"
LANGGRAPH_GRAPH_NAME = "customer_service"

# 全局变量存储助手ID和线程ID
assistant_id = None
current_thread_id = None

# Flask 配置
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key-here")
app.config['SESSION_TYPE'] = 'filesystem'



def ensure_assistant_exists():
    """确保助手存在，如果不存在则创建"""
    global assistant_id

    try:
        # 首先尝试搜索现有的助手
        search_response = requests.post(
            f"{LANGGRAPH_API_URL}/assistants/search",
            json={
                "graph_id": LANGGRAPH_GRAPH_NAME,
                "limit": 1
            },
            timeout=10
        )

        if search_response.status_code == 200:
            assistants = search_response.json()
            if assistants:
                assistant_id = assistants[0]["assistant_id"]
                print(f"✅ 找到现有助手: {assistant_id}")
                return True

        # 如果没有找到，创建新的助手
        create_response = requests.post(
            f"{LANGGRAPH_API_URL}/assistants",
            json={
                "graph_id": LANGGRAPH_GRAPH_NAME,
                "name": "Customer Service Assistant",
                "description": "Multi-agent customer service system"
            },
            timeout=10
        )

        if create_response.status_code == 200:
            result = create_response.json()
            assistant_id = result["assistant_id"]
            print(f"✅ 创建新助手: {assistant_id}")
            return True
        else:
            print(f"❌ 创建助手失败: {create_response.status_code}")
            return False

    except Exception as e:
        print(f"❌ 确保助手存在时出错: {e}")
        return False

def ensure_thread_exists(session_id=None):
    """确保线程存在，如果不存在则创建"""
    global current_thread_id

    # 如果传入了session_id，优先使用它
    if session_id and session_id != 'default':
        # 验证这个session_id是否是有效的LangGraph线程ID
        try:
            # 尝试获取线程详情来验证ID是否有效
            thread_response = requests.get(
                f"{LANGGRAPH_API_URL}/threads/{session_id}",
                timeout=5
            )
            if thread_response.status_code == 200:
                # 这是一个有效的LangGraph线程ID
                current_thread_id = session_id
                return True
            else:
                print(f"⚠️ 会话ID {session_id} 不是有效的LangGraph线程ID，将创建新线程")
                # 清除无效的ID，创建新线程
                session_id = None
        except Exception as e:
            print(f"⚠️ 验证会话ID {session_id} 时出错: {e}")
            session_id = None

    # 如果已经有current_thread_id，直接返回
    if current_thread_id:
        return True

    # 只有在没有线程ID时才创建新的
    try:
        # 创建新线程
        response = requests.post(
            f"{LANGGRAPH_API_URL}/threads",
            json={},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            current_thread_id = result["thread_id"]
            print(f"✅ 创建新线程: {current_thread_id}")
            return True
        else:
            print(f"❌ 创建线程失败: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ 确保线程存在时出错: {e}")
        return False

def get_conversation_history(session_id: str):
    """获取对话历史"""
    if 'conversations' not in session:
        session['conversations'] = {}
    return session['conversations'].get(session_id, [])

def add_conversation_message(session_id: str, role: str, content: str):
    """添加对话消息"""
    history = get_conversation_history(session_id)
    history.append({
        'role': role,
        'content': content,
        'timestamp': time.time()
    })
    session['conversations'][session_id] = history

@app.route('/')
def index():
    """主页"""
    # 获取当前会话的对话历史
    current_session_id = session.get('current_session_id', 'default')
    conversation_history = get_conversation_history(current_session_id)

    return render_template('index.html',
                         conversation_history=conversation_history)

@app.route('/api/chat', methods=['POST'])
def chat():
    """处理聊天请求"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')

        if not user_message:
            return jsonify({'error': '消息不能为空'}), 400

        # 确保助手和线程存在
        if not ensure_assistant_exists():
            return jsonify({'error': '无法创建或找到助手'}), 500

        if not ensure_thread_exists(session_id):
            return jsonify({'error': '无法创建线程'}), 500

        # 在线程上执行运行
        response = requests.post(
            f"{LANGGRAPH_API_URL}/threads/{current_thread_id}/runs",
            json={
                "assistant_id": assistant_id,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_message
                        }
                    ],
                    # 添加额外的状态字段，确保工作流能正确访问
                    "customer_query": user_message,
                    "session_id": current_thread_id
                }
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            run_id = result["run_id"]

            # 等待运行完成
            run_status = "running"
            max_wait_time = 120  # 最大等待120秒
            wait_start = time.time()

            while run_status in ["running", "pending"]:
                if time.time() - wait_start > max_wait_time:
                    print(f"⚠️ 运行超时，已等待 {max_wait_time} 秒")
                    return jsonify({'error': '运行超时'}), 500

                time.sleep(0.5)
                status_response = requests.get(
                    f"{LANGGRAPH_API_URL}/threads/{current_thread_id}/runs/{run_id}",
                    timeout=10
                )
                if status_response.status_code == 200:
                    run_data = status_response.json()
                    run_status = run_data.get("status", "unknown")

                    if run_status in ["completed", "success"]:
                        # 获取线程状态以获取输出
                        thread_response = requests.get(
                            f"{LANGGRAPH_API_URL}/threads/{current_thread_id}/state",
                            timeout=10
                        )
                        if thread_response.status_code == 200:
                            thread_state = thread_response.json()
                            # 从状态中提取AI回复
                            ai_response = extract_ai_response(thread_state)
                            return jsonify({
                                'response': ai_response,
                                'session_id': current_thread_id,  # 返回LangGraph的线程ID
                                'thread_id': current_thread_id     # 同时返回线程ID，确保前端知道正确的ID
                            })
                        else:
                            print(f"❌ 获取线程状态失败: {thread_response.status_code}")
                            return jsonify({'error': '无法获取线程状态'}), 500
                    elif run_status in ["failed", "cancelled"]:
                        print(f"❌ 运行失败: {run_status}")
                        return jsonify({'error': f'运行失败: {run_status}'}), 500
                else:
                    print(f"❌ 获取运行状态失败: {status_response.status_code}")

            return jsonify({'error': '运行超时'}), 500
        else:
            print(f"❌ 创建运行失败: {response.status_code}")
            return jsonify({'error': f'调用失败: {response.status_code}'}), 500

    except Exception as e:
        print(f"❌ 聊天处理错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'内部错误: {str(e)}'}), 500

def extract_ai_response(thread_state):
    """从线程状态中提取AI回复"""
    try:
        # 从 values 字段中提取AI回复
        if "values" in thread_state and isinstance(thread_state["values"], dict):
            values = thread_state["values"]

            # 优先从 values.response 获取
            if "response" in values and values["response"]:
                return values["response"]

            # 从 values.messages 获取
            if "messages" in values:
                for message in values["messages"]:
                    if message.get("role") == "assistant":
                        content = message.get("content", "")
                        if content:
                            return content

        # 如果没有找到，返回默认回复
        return "抱歉，我无法理解您的问题。"

    except Exception as e:
        print(f"❌ 提取AI回复时出错: {e}")
        return "抱歉，处理您的请求时出现了错误。"

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """处理流式聊天请求"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')

        if not user_message:
            return jsonify({'error': '消息不能为空'}), 400

        # 确保助手和线程存在
        if not ensure_assistant_exists():
            return jsonify({'error': '无法创建或找到助手'}), 500

        if not ensure_thread_exists(session_id):
            return jsonify({'error': '无法创建线程'}), 500

        def generate():
            try:
                # 使用流式执行 - 注意：LangGraph 的流式执行是通过 /runs 端点实现的
                response = requests.post(
                    f"{LANGGRAPH_API_URL}/threads/{current_thread_id}/runs",
                    json={
                        "assistant_id": assistant_id,
                        "input": {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": user_message
                                }
                            ],
                            # 添加额外的状态字段，确保工作流能正确访问
                            "customer_query": user_message,
                            "session_id": current_thread_id
                        }
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    run_id = result.get("run_id")

                    if run_id:
                        # 等待运行完成并获取结果
                        run_status = "running"
                        while run_status in ["running", "pending"]:
                            time.sleep(0.5)
                            status_response = requests.get(
                                f"{LANGGRAPH_API_URL}/threads/{current_thread_id}/runs/{run_id}",
                                timeout=10
                            )
                            if status_response.status_code == 200:
                                run_data = status_response.json()
                                run_status = run_data.get("status", "unknown")

                                if run_status in ["completed", "success"]:
                                    # 获取线程状态以获取输出
                                    thread_response = requests.get(
                                        f"{LANGGRAPH_API_URL}/threads/{current_thread_id}/state",
                                        timeout=10
                                    )
                                    if thread_response.status_code == 200:
                                        thread_state = thread_response.json()
                                        ai_response = extract_ai_response(thread_state)
                                        yield f"data: {json.dumps({'content': ai_response, 'session_id': current_thread_id, 'thread_id': current_thread_id})}\n\n"
                                        break
                                elif run_status in ["failed", "cancelled"]:
                                    yield f"data: {json.dumps({'error': f'运行失败: {run_status}'})}\n\n"
                                    break
                        else:
                            yield f"data: {json.dumps({'error': '运行超时'})}\n\n"
                    else:
                        yield f"data: {json.dumps({'error': '无法获取运行ID'})}\n\n"
                else:
                    yield f"data: {json.dumps({'error': f'流式调用失败: {response.status_code}'})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'error': f'流式处理错误: {str(e)}'})}\n\n"

            yield "data: [DONE]\n\n"

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        print(f"❌ 流式聊天处理错误: {e}")
        return jsonify({'error': f'内部错误: {str(e)}'}), 500

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """获取会话列表"""
    try:
        # 调用 LangGraph 线程列表 API - 使用 POST 方法
        response = requests.post(f"{LANGGRAPH_API_URL}/threads/search", json={})

        if response.status_code == 200:
            threads = response.json()

            # 转换数据格式以匹配前端期望
            sessions = []
            for thread in threads:
                # 提取线程ID和创建时间
                thread_id = thread.get("thread_id", "")
                created_at = thread.get("created_at", time.time())

                # 确保创建时间是有效的时间戳
                if isinstance(created_at, str):
                    try:
                        # 尝试解析ISO格式时间
                        import datetime
                        dt = datetime.datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        created_at = dt.timestamp()
                    except:
                        # 如果解析失败，使用当前时间
                        created_at = time.time()
                elif not isinstance(created_at, (int, float)) or created_at <= 0:
                    # 如果时间戳无效，使用当前时间
                    created_at = time.time()

                # 尝试获取消息数量
                message_count = 0
                try:
                    # 获取线程状态以计算消息数量
                    state_response = requests.get(
                        f"{LANGGRAPH_API_URL}/threads/{thread_id}/state",
                        timeout=5
                    )
                    if state_response.status_code == 200:
                        state_data = state_response.json()

                        # 计算消息数量 - 优先从 values.conversation_history 获取
                        if "values" in state_data and isinstance(state_data["values"], dict):
                            values = state_data["values"]
                            if "conversation_history" in values and values["conversation_history"]:
                                # 优先从 conversation_history 获取完整消息数量
                                message_count = len(values["conversation_history"])
                            elif "messages" in values:
                                # 直接计算所有消息数量，不去重
                                message_count = len(values["messages"])
                            elif "response" in values and values["response"]:
                                # 如果有response，至少算1条消息
                                message_count = 1
                        elif "messages" in state_data:
                            # 直接计算所有消息数量，不去重
                            message_count = len(state_data["messages"])
                except Exception as e:
                    message_count = 0

                sessions.append({
                    "session_id": thread_id,
                    "created_at": created_at,
                    "message_count": message_count
                })

            return jsonify({'sessions': sessions})
        else:
            print(f"❌ 获取线程列表失败: {response.status_code}")
            return jsonify({'error': f'获取会话列表失败: {response.status_code}'}), 500

    except Exception as e:
        print(f"❌ 获取会话列表时出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """获取特定会话详情"""
    try:
        # 调用 LangGraph 线程详情 API
        response = requests.get(f"{LANGGRAPH_API_URL}/threads/{session_id}")

        if response.status_code == 200:
            thread_data = response.json()

            # 获取线程状态以获取对话历史
            conversation_history = []
            try:
                state_response = requests.get(
                    f"{LANGGRAPH_API_URL}/threads/{session_id}/state",
                    timeout=5
                )
                if state_response.status_code == 200:
                    state_data = state_response.json()

                    # 从状态中提取对话历史
                    # 优先从 values 字段获取对话历史
                    if "values" in state_data and isinstance(state_data["values"], dict):
                        values = state_data["values"]

                        # 优先从 values.conversation_history 获取完整的对话历史
                        if "conversation_history" in values and values["conversation_history"]:
                            for msg in values["conversation_history"]:
                                content = msg.get("content", "")
                                is_user = msg.get("is_user", False)
                                if content:
                                    conversation_history.append({
                                        "is_user": is_user,
                                        "content": content,
                                        "timestamp": time.time(),
                                        "role": "user" if is_user else "assistant"
                                    })

                        # 如果conversation_history为空，从 values.messages 获取
                        elif "messages" in values:
                            for message in values["messages"]:
                                role = message.get("role", "user")
                                content = message.get("content", "")
                                if content:
                                    # 根据角色判断是否为用户消息
                                    is_user = role == "user"
                                    conversation_history.append({
                                        "is_user": is_user,
                                        "content": content,
                                        "timestamp": time.time(),
                                        "role": role
                                    })

                        # 从 values.response 获取最新的AI回复（如果conversation_history中没有）
                        if "response" in values and values["response"]:
                            response_content = values["response"]
                            # 检查是否已经添加过这个回复
                            if not any(msg["content"] == response_content and not msg["is_user"] for msg in conversation_history):
                                conversation_history.append({
                                    "is_user": False,
                                    "content": response_content,
                                    "timestamp": time.time(),
                                    "role": "assistant"
                                })

                    # 备用方案：从 messages 字段获取
                    elif "messages" in state_data:
                        for message in state_data["messages"]:
                            role = message.get("role", "user")
                            content = message.get("content", "")
                            if content:
                                is_user = role == "user"
                                conversation_history.append({
                                    "is_user": is_user,
                                    "content": content,
                                    "timestamp": time.time(),
                                    "role": role
                                })

                else:
                    print(f"⚠️ 获取线程状态失败: {state_response.status_code}")
            except Exception as e:
                print(f"⚠️ 获取线程状态时出错: {e}")
                import traceback
                traceback.print_exc()

            # 构建会话数据
            session_data = {
                "session_id": session_id,
                "created_at": thread_data.get("created_at", time.time()),
                "conversation_history": conversation_history
            }

            return jsonify({'session': session_data})
        else:
            print(f"❌ 获取线程详情失败: {response.status_code}")
            return jsonify({'error': f'获取会话详情失败: {response.status_code}'}), 500

    except Exception as e:
        print(f"❌ 获取会话详情时出错: {e}")
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """删除会话"""
    try:
        # 调用 LangGraph 删除线程 API
        response = requests.delete(f"{LANGGRAPH_API_URL}/threads/{session_id}")

        if response.status_code == 200:
            # 同时清除本地会话数据
            if 'conversations' in session and session_id in session['conversations']:
                del session['conversations'][session_id]

            return jsonify({'message': '会话删除成功'})
        else:
            return jsonify({'error': f'删除会话失败: {response.status_code}'}), 500

    except Exception as e:
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

@app.route('/api/sessions/<session_id>/clear', methods=['POST'])
def clear_session(session_id):
    """清空会话"""
    try:
        # 调用 LangGraph 清空线程状态 API
        # 注意：LangGraph 没有直接的清空端点，我们通过删除并重新创建线程来实现
        response = requests.delete(f"{LANGGRAPH_API_URL}/threads/{session_id}")

        if response.status_code == 200:
            # 创建新的线程
            new_thread_response = requests.post(
                f"{LANGGRAPH_API_URL}/threads",
                json={},
                timeout=10
            )

            if new_thread_response.status_code == 200:
                new_thread_id = new_thread_response.json()["thread_id"]
                # 同时清除本地会话数据
                if 'conversations' in session and session_id in session['conversations']:
                    session['conversations'][session_id] = []

                return jsonify({
                    'message': '会话清空成功',
                    'new_thread_id': new_thread_id
                })
            else:
                return jsonify({'error': '创建新线程失败'}), 500
        else:
            return jsonify({'error': f'清空会话失败: {response.status_code}'}), 500

    except Exception as e:
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

@app.route('/api/new_session', methods=['POST'])
def create_new_session():
    """创建新会话"""
    try:
        import uuid
        new_session_id = str(uuid.uuid4())

        # 设置当前会话
        session['current_session_id'] = new_session_id

        # 初始化新会话的对话历史
        if 'conversations' not in session:
            session['conversations'] = {}
        session['conversations'][new_session_id] = []

        return jsonify({
            'session_id': new_session_id,
            'message': '新会话创建成功'
        })

    except Exception as e:
        return jsonify({'error': f'创建会话失败: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time()
    })

@app.route('/api/test')
def test_langgraph():
    """测试 LangGraph API 调用"""
    try:
        # LangGraph 服务存活检查（见 README_LangGraph_CLI：GET /ok）
        health_check_status = 0
        try:
            ok_response = requests.get(f"{LANGGRAPH_API_URL}/ok", timeout=5)
            health_check_status = ok_response.status_code
        except requests.exceptions.RequestException as e:
            print(f"⚠️ LangGraph GET /ok 失败: {e}")

        # 测试线程搜索
        threads_response = requests.post(f"{LANGGRAPH_API_URL}/threads/search", json={}, timeout=10)

        # 测试助手搜索
        assistants_response = requests.post(f"{LANGGRAPH_API_URL}/assistants/search", json={}, timeout=10)

        # 部分版本 /ok 可能不存在或非 2xx，但 API 实际可用：用 threads 结果兜底
        if not (200 <= health_check_status < 300) and (200 <= threads_response.status_code < 300):
            print("⚠️ LangGraph GET /ok 未成功，但 threads/search 正常，健康检查标记为通过")
            health_check_status = 200

        return jsonify({
            'status': 'test_completed',
            'health_check': health_check_status,
            'threads_search': threads_response.status_code,
            'assistants_search': assistants_response.status_code,
            'details': {
                'ok_response': 'OK' if 200 <= health_check_status < 300 else (health_check_status or 'unreachable'),
                'threads_response': threads_response.text if threads_response.status_code != 200 else 'OK',
                'assistants_response': assistants_response.text if assistants_response.status_code != 200 else 'OK'
            }
        })

    except Exception as e:
        print(f"❌ 测试 LangGraph API 时出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'测试失败: {str(e)}'}), 500

def main():
    """主函数"""
    print("🚀 多智能体客服系统 Web 应用")
    print("=" * 60)

    # 启动 Web 服务
    print("🌐 启动 Web 服务...")
    print(f"📱 访问地址: http://localhost:5000")
    print("💡 按 Ctrl+C 停止服务")
    print()

    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    import time
    main()
