"""
投诉处理专家智能体
专门负责客户投诉和建议处理
"""

from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import BaseAgent

class ComplaintAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="投诉处理专家",
            role="客户投诉和建议处理",
            expertise=["问题记录", "解决方案", "补偿措施", "服务改进"]
        )

        # TODO: 投诉处理信息应该从客服系统获取，这里只是模拟数据
        # 实际应用中应该连接客服数据库或调用客服API服务
        self.complaint_database = {
            "服务问题": {
                "响应速度慢": "承诺24小时内响应，超时提供补偿",
                "服务态度差": "记录问题，安排专人跟进，提供道歉补偿",
                "专业能力不足": "安排专业培训，提供专家支持",
                "处理流程": "记录问题 → 分析原因 → 制定方案 → 执行解决 → 回访确认"
            },
            "产品质量": {
                "功能缺陷": "提供免费维修或更换，延长保修期",
                "外观瑕疵": "提供更换或折扣补偿",
                "性能不达标": "技术检测确认后，提供升级或退款",
                "补偿标准": "根据问题严重程度，提供10%-100%的补偿"
            },
            "物流配送": {
                "配送延迟": "超时提供运费补偿，加急配送",
                "包装破损": "拍照记录，提供更换或补偿",
                "配送错误": "免费重新配送，提供额外补偿",
                "紧急处理": "24小时内响应，48小时内解决"
            }
        }

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理投诉相关查询"""
        customer_query = state["customer_query"]
        session_id = state.get("session_id", "default")

        # 对话轮次由 classify / 外层节点写入 persisted_dialogue
        conversation_context = self._get_conversation_context(session_id, state)

        # 从投诉数据库中匹配相关信息
        matched_info = self._match_complaint_info(customer_query)

        # 构建系统提示并增强对话上下文说明
        base_system_prompt = f"""你是{self.name}，专门负责{self.role}。
        你的专业领域包括：{', '.join(self.expertise)}

        请以专业、耐心的态度处理客户投诉：
        1. 认真倾听客户的问题和不满
        2. 表达理解和歉意
        3. 提供具体的解决方案和时间承诺
        4. 如果问题复杂，说明后续处理流程

        回答要真诚、专业，体现对客户的重视。如果投诉超出你的处理权限，请说明并承诺转交给相关部门处理。"""

        system_prompt = self._enhance_system_prompt_with_context(base_system_prompt)

        # 构建消息列表
        messages = []

        # 添加对话历史上下文（如果有的话）
        if conversation_context:
            context_message = f"""对话历史上下文：
{conversation_context}

请基于以上对话历史和当前查询，提供连贯的处理方案。"""
            messages.append(SystemMessage(content=context_message))

        # 添加系统提示
        messages.append(SystemMessage(content=system_prompt))

        # 如果有匹配的投诉信息，添加到上下文中
        if matched_info:
            complaint_context = f"""投诉处理政策：
{matched_info}

当前查询：{customer_query}"""
            messages.append(HumanMessage(content=complaint_context))
        else:
            messages.append(HumanMessage(content=customer_query))

        # 调用LLM
        try:
            response = self.llm.invoke(messages)
            response_content = response.content
        except Exception as e:
            print(f"投诉专家调用LLM时出错: {e}")
            response_content = "抱歉，处理您的投诉时遇到系统错误，请稍后重试。"

        state["response"] = response_content
        state["current_agent"] = self.name
        state["tools_used"].append(f"{self.name}_processing")

        return state

    def _match_complaint_info(self, query: str) -> str:
        """匹配查询中的投诉信息"""
        query_lower = query.lower()
        matched_info = []

        # 精确匹配投诉类型
        for category, solutions in self.complaint_database.items():
            if any(keyword in query_lower for keyword in category.lower().split()):
                # 格式化投诉信息
                info_text = f"""【{category}】\n"""
                for issue, solution in solutions.items():
                    info_text += f"• {issue}：{solution}\n"
                matched_info.append(info_text)

        # 如果没有精确匹配，尝试关键词匹配
        if not matched_info:
            for category, solutions in self.complaint_database.items():
                if any(keyword in query_lower for keyword in ["投诉", "问题", "建议", "不满", "改进"]):
                    info_text = f"""相关处理：{category}\n"""
                    # 只显示前2项解决方案
                    for i, (issue, solution) in enumerate(solutions.items()):
                        if i < 2:
                            info_text += f"• {issue}：{solution}\n"
                    info_text += "..."
                    matched_info.append(info_text)

        return "\n".join(matched_info) if matched_info else ""
