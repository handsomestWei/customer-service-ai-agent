#!/usr/bin/env python3
"""
客服 Web 相关业务逻辑：LangGraph REST 调用、线程/运行、状态解析、会话列表拼装等。
与 Flask 路由解耦，便于单测与复用。
"""

from __future__ import annotations

import json
import os
import time
import datetime as _dt
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# 配置（可被环境变量覆盖）
# -----------------------------------------------------------------------------

LANGGRAPH_API_URL: str = os.getenv("LANGGRAPH_API_URL", "http://127.0.0.1:2024").rstrip("/")
LANGGRAPH_GRAPH_NAME: str = os.getenv("LANGGRAPH_GRAPH_NAME", "customer_service")

# LangGraph SDK 侧的助手与当前线程缓存（与原 web_app 行为一致）
_assistant_id: Optional[str] = None
_current_thread_id: Optional[str] = None


def get_assistant_id() -> Optional[str]:
    return _assistant_id


def get_current_thread_id() -> Optional[str]:
    return _current_thread_id


# -----------------------------------------------------------------------------
# 线程 state → 对话列表 / 侧栏预览
# -----------------------------------------------------------------------------

def append_turn_from_state(conversation_history: List[Dict[str, Any]], msg: Dict[str, Any]) -> None:
    """从状态中的单条消息追加到会话历史列表；仅在状态里带有 timestamp 时写入条目。"""
    content = msg.get("content", "") or ""
    if not content:
        return
    is_user = bool(msg.get("is_user", False))
    entry: Dict[str, Any] = {
        "is_user": is_user,
        "content": content,
        "role": "user" if is_user else "assistant",
    }
    ts = msg.get("timestamp")
    if ts is not None and ts != "":
        entry["timestamp"] = ts
    conversation_history.append(entry)


def conversation_history_from_state_data(state_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 LangGraph 线程 state JSON 解析对话列表。"""
    conversation_history: List[Dict[str, Any]] = []
    if not isinstance(state_data, dict):
        return conversation_history

    if "values" in state_data and isinstance(state_data["values"], dict):
        values = state_data["values"]

        source_turns = None
        filled_from_turn_list = False
        pd_raw = values.get("persisted_dialogue")
        ch_raw = values.get("conversation_history")
        if isinstance(pd_raw, list) and len(pd_raw) > 0:
            source_turns = pd_raw
        elif isinstance(ch_raw, list) and len(ch_raw) > 0:
            source_turns = ch_raw

        if source_turns is not None:
            filled_from_turn_list = True
            for msg in source_turns:
                if isinstance(msg, dict):
                    append_turn_from_state(conversation_history, msg)

        elif "messages" in values:
            for message in values["messages"]:
                role = message.get("role", "user")
                content = message.get("content", "")
                if content:
                    is_user = role == "user"
                    row: Dict[str, Any] = {
                        "is_user": is_user,
                        "content": content,
                        "role": role,
                    }
                    mt = message.get("timestamp")
                    if mt is not None and mt != "":
                        row["timestamp"] = mt
                    conversation_history.append(row)

        # 已有 persisted_dialogue / conversation_history 时不再追加 values.response：
        # 助手正文已在轮次里；final_response_node 还可能给 response 加前缀导致去重失败、出现双线助手气泡。
        if "response" in values and values["response"]:
            response_content = values["response"]
            if not filled_from_turn_list:
                if not any(
                    msg["content"] == response_content and not msg["is_user"]
                    for msg in conversation_history
                ):
                    conversation_history.append({
                        "is_user": False,
                        "content": response_content,
                        "role": "assistant"
                    })

    elif "messages" in state_data:
        for message in state_data["messages"]:
            role = message.get("role", "user")
            content = message.get("content", "")
            if content:
                is_user = role == "user"
                row = {
                    "is_user": is_user,
                    "content": content,
                    "role": role,
                }
                mt = message.get("timestamp")
                if mt is not None and mt != "":
                    row["timestamp"] = mt
                conversation_history.append(row)

    return conversation_history


def last_user_question_from_history(conversation_history: List[Dict[str, Any]]) -> str:
    """取最后一条用户消息的纯文本（用于侧栏预览）。"""
    for msg in reversed(conversation_history):
        if not msg.get("is_user"):
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content) if content is not None else ""
        s = content.strip()
        if s:
            return s
    return ""


def extract_ai_response(thread_state: Dict[str, Any]) -> str:
    """从线程状态中提取 AI 回复文本。"""
    try:
        if "values" in thread_state and isinstance(thread_state["values"], dict):
            values = thread_state["values"]

            if "response" in values and values["response"]:
                return str(values["response"])

            if "messages" in values:
                for message in values["messages"]:
                    if message.get("role") == "assistant":
                        content = message.get("content", "")
                        if content:
                            return str(content)

        return "抱歉，我无法理解您的问题。"

    except Exception as e:
        print(f"❌ 提取AI回复时出错: {e}")
        return "抱歉，处理您的请求时出现了错误。"


# -----------------------------------------------------------------------------
# 助手 / 线程
# -----------------------------------------------------------------------------

def ensure_assistant_exists() -> bool:
    """确保 LangGraph 助手存在，不存在则创建。"""
    global _assistant_id

    try:
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
                _assistant_id = assistants[0]["assistant_id"]
                print(f"✅ 找到现有助手: {_assistant_id}")
                return True

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
            _assistant_id = result["assistant_id"]
            print(f"✅ 创建新助手: {_assistant_id}")
            return True
        else:
            print(f"❌ 创建助手失败: {create_response.status_code}")
            return False

    except Exception as e:
        print(f"❌ 确保助手存在时出错: {e}")
        return False


def ensure_thread_exists(client_session_id: Optional[str] = None) -> bool:
    """
    确保有可用的 LangGraph 线程。
    client_session_id: 前端传入的 session_id；若是合法线程 ID 则复用。"""
    global _current_thread_id

    sid = client_session_id
    if sid and sid != 'default':
        try:
            thread_response = requests.get(
                f"{LANGGRAPH_API_URL}/threads/{sid}",
                timeout=5
            )
            if thread_response.status_code == 200:
                _current_thread_id = sid
                return True
            else:
                print(f"⚠️ 会话ID {sid} 不是有效的LangGraph线程ID，将创建新线程")
                sid = None
        except Exception as e:
            print(f"⚠️ 验证会话ID {sid} 时出错: {e}")
            sid = None

    if _current_thread_id:
        return True

    try:
        response = requests.post(
            f"{LANGGRAPH_API_URL}/threads",
            json={},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            _current_thread_id = result["thread_id"]
            print(f"✅ 创建新线程: {_current_thread_id}")
            return True
        else:
            print(f"❌ 创建线程失败: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ 确保线程存在时出错: {e}")
        return False


def _normalize_created_at(created_at: Any) -> float:
    if isinstance(created_at, str):
        try:
            dt = _dt.datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            return dt.timestamp()
        except Exception:
            return time.time()
    if isinstance(created_at, (int, float)) and created_at > 0:
        return float(created_at)
    return time.time()


def _message_count_from_state_data(state_data: Dict[str, Any]) -> int:
    if "values" in state_data and isinstance(state_data["values"], dict):
        values = state_data["values"]
        if "conversation_history" in values and values["conversation_history"]:
            return len(values["conversation_history"])
        if "messages" in values:
            return len(values["messages"])
        if "response" in values and values["response"]:
            return 1
    if "messages" in state_data:
        return len(state_data["messages"])
    return 0


def fetch_sessions_list() -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    拉取线程列表并拼装前端会话项。
    成功返回 (sessions, None)，失败返回 (None, error_message)。
    """
    try:
        response = requests.post(f"{LANGGRAPH_API_URL}/threads/search", json={})

        if response.status_code != 200:
            print(f"❌ 获取线程列表失败: {response.status_code}")
            return None, f'获取会话列表失败: {response.status_code}'

        threads = response.json()
        sessions: List[Dict[str, Any]] = []

        for thread in threads:
            thread_id = thread.get("thread_id", "")
            created_at = _normalize_created_at(thread.get("created_at", time.time()))

            message_count = 0
            last_user_question = ""
            try:
                state_response = requests.get(
                    f"{LANGGRAPH_API_URL}/threads/{thread_id}/state",
                    timeout=5
                )
                if state_response.status_code == 200:
                    state_data = state_response.json()
                    parsed_hist = conversation_history_from_state_data(state_data)
                    last_user_question = last_user_question_from_history(parsed_hist)
                    message_count = _message_count_from_state_data(state_data)
            except Exception:
                message_count = 0

            sessions.append({
                "session_id": thread_id,
                "created_at": created_at,
                "message_count": message_count,
                "last_user_question": last_user_question,
            })

        return sessions, None

    except Exception as e:
        print(f"❌ 获取会话列表时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, f'服务器错误: {str(e)}'


def fetch_session_detail(session_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """获取单个线程详情 + 对话历史。成功返回 (payload, None)。"""
    try:
        response = requests.get(f"{LANGGRAPH_API_URL}/threads/{session_id}")

        if response.status_code != 200:
            print(f"❌ 获取线程详情失败: {response.status_code}")
            return None, f'获取会话详情失败: {response.status_code}'

        thread_data = response.json()
        conversation_history: List[Dict[str, Any]] = []

        try:
            state_response = requests.get(
                f"{LANGGRAPH_API_URL}/threads/{session_id}/state",
                timeout=5
            )
            if state_response.status_code == 200:
                state_data = state_response.json()
                conversation_history = conversation_history_from_state_data(state_data)
            else:
                print(f"⚠️ 获取线程状态失败: {state_response.status_code}")
        except Exception as e:
            print(f"⚠️ 获取线程状态时出错: {e}")
            import traceback
            traceback.print_exc()

        session_data = {
            "session_id": session_id,
            "created_at": thread_data.get("created_at", time.time()),
            "conversation_history": conversation_history
        }
        return session_data, None

    except Exception as e:
        print(f"❌ 获取会话详情时出错: {e}")
        return None, f'服务器错误: {str(e)}'


def delete_remote_thread(thread_id: str) -> Tuple[bool, int]:
    """删除 LangGraph 线程。返回 (是否 2xx, status_code)。"""
    response = requests.delete(f"{LANGGRAPH_API_URL}/threads/{thread_id}")
    return response.status_code == 200, response.status_code


def clear_thread_and_create_new(thread_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    删除旧线程并在服务端新建线程。
    成功返回 (new_thread_id, None)，失败返回 (None, error)。
    """
    ok, status = delete_remote_thread(thread_id)
    if not ok:
        return None, f'清空会话失败: {status}'

    new_thread_response = requests.post(
        f"{LANGGRAPH_API_URL}/threads",
        json={},
        timeout=10
    )

    if new_thread_response.status_code != 200:
        return None, '创建新线程失败'

    new_thread_id = new_thread_response.json()["thread_id"]
    return new_thread_id, None


# -----------------------------------------------------------------------------
# 一次聊天运行（阻塞轮询）
# -----------------------------------------------------------------------------

def run_chat_sync(user_message: str, client_session_id: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    在当前线程上提交一轮用户消息并等待完成。
    client_session_id: 前端传入的会话 ID（可为 LangGraph 线程 ID）。
    返回 (ai_text, error_text, http_status_optional)。
    """
    global _assistant_id, _current_thread_id

    if not user_message.strip():
        return None, '消息不能为空', 400

    if not ensure_assistant_exists():
        return None, '无法创建或找到助手', 500

    if not ensure_thread_exists(client_session_id):
        return None, '无法创建线程', 500

    assert _assistant_id and _current_thread_id

    try:
        run_resp = requests.post(
            f"{LANGGRAPH_API_URL}/threads/{_current_thread_id}/runs",
            json={
                "assistant_id": _assistant_id,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_message.strip()
                        }
                    ],
                    "customer_query": user_message.strip(),
                    "session_id": _current_thread_id
                }
            },
            timeout=30
        )

        if run_resp.status_code != 200:
            print(f"❌ 创建运行失败: {run_resp.status_code}")
            return None, f'调用失败: {run_resp.status_code}', run_resp.status_code

        result = run_resp.json()
        run_id = result["run_id"]

        run_status = "running"
        max_wait_time = 120
        wait_start = time.time()

        while run_status in ["running", "pending"]:
            if time.time() - wait_start > max_wait_time:
                print(f"⚠️ 运行超时，已等待 {max_wait_time} 秒")
                return None, '运行超时', 500

            time.sleep(0.5)
            status_response = requests.get(
                f"{LANGGRAPH_API_URL}/threads/{_current_thread_id}/runs/{run_id}",
                timeout=10
            )
            if status_response.status_code != 200:
                print(f"❌ 获取运行状态失败: {status_response.status_code}")
                break

            run_data = status_response.json()
            run_status = run_data.get("status", "unknown")

            if run_status in ["completed", "success"]:
                thread_response = requests.get(
                    f"{LANGGRAPH_API_URL}/threads/{_current_thread_id}/state",
                    timeout=10
                )
                if thread_response.status_code == 200:
                    thread_state = thread_response.json()
                    ai_response = extract_ai_response(thread_state)
                    return ai_response, None, None
                else:
                    print(f"❌ 获取线程状态失败: {thread_response.status_code}")
                    return None, '无法获取线程状态', 500

            if run_status in ["failed", "cancelled"]:
                print(f"❌ 运行失败: {run_status}")
                return None, f'运行失败: {run_status}', 500

        return None, '运行超时', 500

    except Exception as e:
        print(f"❌ 聊天处理错误: {e}")
        import traceback
        traceback.print_exc()
        return None, f'内部错误: {str(e)}', 500


def stream_chat_events(user_message: str, client_session_id: Optional[str] = None) -> Iterable[str]:
    """
    生成 SSE data 行（含末尾 [DONE]），供 Flask Response 逐块写出。
    """
    global _assistant_id, _current_thread_id

    if not user_message.strip():
        yield f"data: {json.dumps({'error': '消息不能为空'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    if not ensure_assistant_exists():
        yield f"data: {json.dumps({'error': '无法创建或找到助手'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    if not ensure_thread_exists(client_session_id):
        yield f"data: {json.dumps({'error': '无法创建线程'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    assert _assistant_id is not None and _current_thread_id is not None

    try:
        response = requests.post(
            f"{LANGGRAPH_API_URL}/threads/{_current_thread_id}/runs",
            json={
                "assistant_id": _assistant_id,
                "input": {
                    "messages": [{"role": "user", "content": user_message.strip()}],
                    "customer_query": user_message.strip(),
                    "session_id": _current_thread_id
                }
            },
            timeout=30
        )

        if response.status_code != 200:
            yield f"data: {json.dumps({'error': f'流式调用失败: {response.status_code}'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        result = response.json()
        run_id = result.get("run_id")

        if not run_id:
            yield f"data: {json.dumps({'error': '无法获取运行ID'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        tid = _current_thread_id
        run_status = "running"
        while run_status in ["running", "pending"]:
            time.sleep(0.5)
            status_response = requests.get(
                f"{LANGGRAPH_API_URL}/threads/{tid}/runs/{run_id}",
                timeout=10
            )
            if status_response.status_code != 200:
                yield f"data: {json.dumps({'error': f'获取运行状态失败: {status_response.status_code}'})}\n\n"
                break

            run_data = status_response.json()
            run_status = run_data.get("status", "unknown")

            if run_status in ["completed", "success"]:
                thread_response = requests.get(
                    f"{LANGGRAPH_API_URL}/threads/{tid}/state",
                    timeout=10
                )
                if thread_response.status_code == 200:
                    thread_state = thread_response.json()
                    ai_response = extract_ai_response(thread_state)
                    yield f"data: {json.dumps({'content': ai_response, 'session_id': tid, 'thread_id': tid})}\n\n"
                else:
                    yield f"data: {json.dumps({'error': '无法获取线程状态'})}\n\n"
                break

            if run_status in ["failed", "cancelled"]:
                yield f"data: {json.dumps({'error': f'运行失败: {run_status}'})}\n\n"
                break
        else:
            yield f"data: {json.dumps({'error': '运行超时'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': f'流式处理错误: {str(e)}'})}\n\n"

    yield "data: [DONE]\n\n"


def langgraph_connectivity_test() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    探测 LangGraph 服务与搜索接口。
    成功返回 (result_dict, None)，失败返回 (None, error_message)。
    """
    try:
        health_check_status = 0
        try:
            ok_response = requests.get(f"{LANGGRAPH_API_URL}/ok", timeout=5)
            health_check_status = ok_response.status_code
        except requests.exceptions.RequestException as e:
            print(f"⚠️ LangGraph GET /ok 失败: {e}")

        threads_response = requests.post(f"{LANGGRAPH_API_URL}/threads/search", json={}, timeout=10)
        assistants_response = requests.post(f"{LANGGRAPH_API_URL}/assistants/search", json={}, timeout=10)

        if not (200 <= health_check_status < 300) and (200 <= threads_response.status_code < 300):
            print("⚠️ LangGraph GET /ok 未成功，但 threads/search 正常，健康检查标记为通过")
            health_check_status = 200

        return ({
            'status': 'test_completed',
            'health_check': health_check_status,
            'threads_search': threads_response.status_code,
            'assistants_search': assistants_response.status_code,
            'details': {
                'ok_response': 'OK' if 200 <= health_check_status < 300 else (health_check_status or 'unreachable'),
                'threads_response': threads_response.text if threads_response.status_code != 200 else 'OK',
                'assistants_response': assistants_response.text if assistants_response.status_code != 200 else 'OK'
            }
        }, None)

    except Exception as e:
        print(f"❌ 测试 LangGraph API 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, f'测试失败: {str(e)}'
