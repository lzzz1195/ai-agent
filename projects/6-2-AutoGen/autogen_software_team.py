import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console


load_dotenv(Path(__file__).with_name(".env"), override=True)


def create_model_client():
    """通过兼容 OpenAI 的接口连接 DeepSeek。"""
    for name in ("LLM_API_KEY", "LLM_MODEL_ID", "LLM_BASE_URL"):
        if not os.getenv(name):
            raise ValueError(f".env 缺少配置：{name}")

    return OpenAIChatCompletionClient(
        model=os.environ["LLM_MODEL_ID"],
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        # 本案例只使用文本对话，不使用图片、工具调用或结构化输出。
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        },
    )


def create_product_manager(model_client):
    return AssistantAgent(
        name="ProductManager",
        model_client=model_client,
        system_message=(
            "你是一位产品经理，负责软件需求分析和开发规划。"
            "请根据用户需求输出：核心需求、功能划分、技术方案、验收标准。"
            "已有方案时，根据新反馈提出必要调整，避免重复分析。"
            "保持简洁，完成后说：请工程师开始实现。"
        ),
    )


def create_engineer(model_client):
    return AssistantAgent(
        name="Engineer",
        model_client=model_client,
        system_message=(
            "你是一位资深 Python 工程师，负责根据需求和反馈实现代码。"
            "使用 Streamlit 和 requests 构建应用，提供完整的 app.py 代码、"
            "依赖安装命令和启动命令。使用真实公开的行情 API，"
            "处理网络超时、HTTP 错误、缺失字段，并说明数据时间与单位。"
            "不要编造实时行情，不要声称已经执行过代码。"
            "收到修改意见后提供修正后的完整代码。"
            "完成后说：请代码审查员检查。"
        ),
    )


def create_code_reviewer(model_client):
    return AssistantAgent(
        name="CodeReviewer",
        model_client=model_client,
        system_message=(
            "你是一位代码审查员，检查代码是否满足需求。"
            "重点检查 API 返回字段、异常处理、刷新逻辑、可运行性和安全性。"
            "指出具体问题并给出修改建议；如有阻止运行的问题，给出修正代码。"
            "区分静态检查和实际运行结果，不要声称已经执行测试。"
            "最后提供简短的人工测试清单，并说：请用户运行并反馈。"
        ),
    )


def create_user_proxy():
    return UserProxyAgent(
        name="UserProxy",
        input_func=input,
        description=(
            "接收人类用户输入的测试结果、错误信息或修改要求。"
            "用户验收完成后输入 TERMINATE 结束协作。"
        ),
    )


async def main():
    model_client = create_model_client()

    try:
        # 四个角色按列表顺序依次发言。
        team = RoundRobinGroupChat(
            participants=[
                create_product_manager(model_client),
                create_engineer(model_client),
                create_code_reviewer(model_client),
                create_user_proxy(),
            ],
            # 仅用户代理发出的 TERMINATE 能触发这个停止条件。
            termination_condition=TextMentionTermination(
                "TERMINATE",
                sources=["UserProxy"],
            ),
            max_turns=20,
        )

        task = """
请团队协作开发一个比特币价格显示应用。

要求：
1. 使用 Streamlit 和 requests。
2. 显示 BTC 当前美元价格、24 小时涨跌幅和涨跌额。
3. 显示数据时间，提供手动刷新按钮。
4. 使用真实公开的行情 API；如果需要密钥，明确说明配置方式。
5. 添加加载状态和异常提示，界面使用中文。
6. 交付完整 app.py、依赖安装命令、启动命令和测试方法。

请从需求分析开始，依次完成开发和代码审查。
"""

        print("启动软件开发团队；轮到你发言时可输入反馈。")
        print("验收完成后输入 TERMINATE 结束。\n")

        await Console(team.run_stream(task=task))

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
