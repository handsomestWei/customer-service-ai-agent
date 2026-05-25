"""
技术支持专家智能体
专门负责技术问题诊断和解决
"""

from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import BaseAgent

class TechAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="技术支持专家",
            role="技术问题诊断和解决",
            expertise=["故障诊断", "系统优化", "软件配置", "硬件维修"]
        )

        # TODO: 技术解决方案应该从知识库获取，这里只是模拟数据
        # 实际应用中应该连接技术知识库或调用技术支持API服务
        self.tech_database = {
            "常见故障": {
                "无法开机": "检查电源连接 → 长按电源键10秒 → 检查电池状态 → 联系技术支持",
                "系统卡顿": "清理缓存 → 关闭后台应用 → 重启设备 → 系统优化",
                "网络连接": "检查WiFi设置 → 重启路由器 → 检查网络配置 → 联系网络服务商",
                "软件崩溃": "强制关闭应用 → 清除应用数据 → 重新安装 → 检查系统兼容性"
            },
            "系统优化": {
                "性能提升": "清理垃圾文件 → 优化启动项 → 更新驱动程序 → 系统维护",
                "存储管理": "删除无用文件 → 清理下载文件夹 → 使用云存储 → 定期备份",
                "安全设置": "更新安全补丁 → 配置防火墙 → 安装杀毒软件 → 定期扫描",
                "电池优化": "调整屏幕亮度 → 关闭无用功能 → 优化应用设置 → 检查电池健康"
            },
            "硬件问题": {
                "屏幕问题": "检查连接线 → 更新显卡驱动 → 调整分辨率 → 联系维修",
                "声音问题": "检查音频设置 → 测试不同设备 → 更新音频驱动 → 硬件检测",
                "散热问题": "清理灰尘 → 检查风扇 → 优化使用环境 → 更换散热器",
                "接口故障": "检查连接 → 测试不同设备 → 更新驱动 → 硬件维修"
            }
        }

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理技术支持查询"""
        customer_query = state["customer_query"]
        session_id = state.get("session_id", "default")

        # 对话轮次由 classify / 外层节点写入 persisted_dialogue
        conversation_context = self._get_conversation_context(session_id, state)

        # 从技术数据库中匹配相关信息
        matched_info = self._match_tech_info(customer_query)

        # 构建系统提示并增强对话上下文说明
        base_system_prompt = f"""你是{self.name}，专门负责{self.role}。
        你的专业领域包括：{', '.join(self.expertise)}

        请根据客户的技术问题提供专业的解决方案：
        1. 仔细分析问题的技术细节
        2. 提供清晰的解决步骤
        3. 说明可能的原因和预防措施
        4. 如果问题复杂，建议联系专业技术人员

        回答要专业、准确，技术术语要通俗易懂。如果问题超出你的专业范围，请说明并建议转接给相应的技术专家。"""

        system_prompt = self._enhance_system_prompt_with_context(base_system_prompt)

        # 构建消息列表
        messages = []

        # 添加对话历史上下文（如果有的话）
        if conversation_context:
            context_message = f"""对话历史上下文：
{conversation_context}

请基于以上对话历史和当前查询，提供连贯的解答。"""
            messages.append(SystemMessage(content=context_message))

        # 添加系统提示
        messages.append(SystemMessage(content=system_prompt))

        # 如果有匹配的技术信息，添加到上下文中
        if matched_info:
            tech_context = f"""技术解决方案：
{matched_info}

当前查询：{customer_query}"""
            messages.append(HumanMessage(content=tech_context))
        else:
            messages.append(HumanMessage(content=customer_query))

        # 调用LLM
        try:
            response = self.llm.invoke(messages)
            response_content = response.content
        except Exception as e:
            print(f"技术专家调用LLM时出错: {e}")
            response_content = "抱歉，处理您的技术问题时遇到系统错误，请稍后重试。"

        state["response"] = response_content
        state["current_agent"] = self.name
        state["tools_used"].append(f"{self.name}_processing")

        return state

    def _match_tech_info(self, query: str) -> str:
        """匹配查询中的技术信息"""
        query_lower = query.lower()
        matched_info = []

        # 精确匹配技术类型
        for category, solutions in self.tech_database.items():
            if any(keyword in query_lower for keyword in category.lower().split()):
                # 格式化技术信息
                info_text = f"""【{category}】\n"""
                for issue, solution in solutions.items():
                    info_text += f"• {issue}：{solution}\n"
                matched_info.append(info_text)

        # 如果没有精确匹配，尝试关键词匹配
        if not matched_info:
            for category, solutions in self.tech_database.items():
                if any(keyword in query_lower for keyword in ["问题", "故障", "无法", "怎么", "怎么办", "技术支持"]):
                    if category not in [info.split('【')[1].split('】')[0] for info in matched_info]:
                        info_text = f"""相关解决方案：{category}\n"""
                        # 只显示前2项解决方案
                        for i, (issue, solution) in enumerate(solutions.items()):
                            if i < 2:
                                info_text += f"• {issue}：{solution}\n"
                        info_text += "..."
                        matched_info.append(info_text)

        return "\n".join(matched_info) if matched_info else ""
