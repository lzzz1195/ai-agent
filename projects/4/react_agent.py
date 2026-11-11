import re

from HelloAgentsLLM import HelloAgentsLLM
from tools import ToolExecutor, search


REACT_PROMPT_TEMPLATE = """
你是一个可以调用外部工具的助手。

可用工具：
{tools}

每轮只输出一组 Thought 和 Action，格式如下：
Thought: 简短说明本轮行动的目的。
Action: 工具名称[工具输入]

规则：
1. 每轮只能执行一个 Action。
2. 需要实时信息时，使用 Search[搜索关键词]。
3. 工具结果由程序通过 Observation 提供，不要自行编造。
4. 获得足够信息后，使用 Finish[最终答案]。
5. Action 必须使用英文冒号和英文方括号。

用户问题：
{question}

历史行动和工具结果：
{history}
"""


class ReActAgent:
    def __init__(self, llm_client, tool_executor, max_steps=5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def _parse_action(self, response):
        # 提取第一条 Action，兼容多行最终答案。
        action_match = re.search(
            r"^[ \t]*Action:[ \t]*(.*?)"
            r"(?=^[ \t]*(?:Thought|Action|Observation):|\Z)",
            response,
            re.MULTILINE | re.DOTALL,
        )

        if not action_match:
            return None, None, None

        action_text = action_match.group(1).strip()

        command_match = re.fullmatch(
            r"(\w+)\[(.*)\]",
            action_text,
            re.DOTALL,
        )

        if not command_match:
            return action_text, None, None

        name, argument = command_match.groups()
        return action_text, name, argument.strip()

    def run(self, question):
        self.history = []

        for step in range(1, self.max_steps + 1):
            print(f"\n--- 第 {step} 步 ---")

            # 1. 把问题、工具说明和历史结果交给模型。
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=self.tool_executor.getAvailableTools(),
                question=question,
                history="\n".join(self.history),
            )

            response = self.llm_client.think(
                messages=[{"role": "user", "content": prompt}]
            )

            if not response:
                print("模型没有返回有效内容，请检查模型调用。")
                return None

            # 2. 解析模型想执行的行动。
            action_text, name, argument = self._parse_action(response)

            if name is None:
                self.history.append(
                    "Observation: 上一轮 Action 格式无效。"
                    "请使用 Search[关键词] 或 Finish[答案]。"
                )
                continue

            # 3. Finish 表示任务完成。
            if name == "Finish":
                print(f"\n最终答案：\n{argument}")
                return argument

            # 4. 找到工具，由 Python 执行。
            tool = self.tool_executor.getTool(name)

            if tool is None:
                observation = f"错误：不存在名为 {name} 的工具。"
            else:
                try:
                    observation = tool(argument)
                except Exception as exc:
                    observation = f"工具执行失败：{exc}"

            print(f"\nObservation：\n{observation}")

            # 5. 保存结果，让模型下一轮能够看到。
            self.history.extend([
                f"Action: {action_text}",
                f"Observation: {observation}",
            ])

        print("\n达到最大执行步数，尚未完成任务。")
        return None


if __name__ == "__main__":
    llm = HelloAgentsLLM()

    executor = ToolExecutor()
    executor.registerTool(
        "Search",
        "搜索互联网，获取最新消息、产品信息和事实资料。",
        search,
    )

    agent = ReActAgent(llm, executor, max_steps=10)
    agent.run(
        "华为最新发布的手机是哪一款？"
        "它的主要卖点是什么？请先搜索，并注明信息来源。"
    )
