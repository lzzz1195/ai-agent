import ast
import re

from HelloAgentsLLM import HelloAgentsLLM


PLANNER_PROMPT_TEMPLATE = """
你是一个任务规划助手。
请将用户的问题分解为简洁、有逻辑顺序的执行步骤。

要求：
1. 每个步骤描述一个明确的子任务。
2. 后面的步骤可以使用前面步骤的结果。
3. 只生成计划，不提前计算答案。
4. 最后一个步骤必须汇总结果，回答原始问题。
5. 只输出一个 Python 字符串列表，使用下面的格式：

```python
["步骤1", "步骤2", "步骤3"]
```

用户问题：
{question}
"""


EXECUTOR_PROMPT_TEMPLATE = """
你是一个任务执行助手。
请根据原始问题、完整计划和已有结果，完成当前步骤。

原始问题：
{question}

完整计划：
{plan}

已经完成的步骤及结果：
{history}

当前步骤：
{current_step}

请只输出当前步骤的结果，必要时注明单位。
不要执行后续步骤。
如果当前步骤是最终汇总，请直接回答原始问题。
"""


class Planner:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def plan(self, question: str) -> list[str]:
        print("\n--- 正在生成计划 ---")

        prompt = PLANNER_PROMPT_TEMPLATE.format(
            question=question
        )

        response = self.llm_client.think(
            messages=[{"role": "user", "content": prompt}]
        )

        if not response or not response.strip():
            print("规划失败：模型没有返回有效内容。")
            return []

        try:
            # 优先提取代码块；也兼容直接返回列表的情况。
            match = re.search(
                r"```(?:python)?\s*(.*?)```",
                response,
                re.DOTALL | re.IGNORECASE,
            )
            plan_text = (
                match.group(1).strip()
                if match
                else response.strip()
            )

            # 将列表文本解析为 Python 数据，不执行其中的代码。
            plan = ast.literal_eval(plan_text)

            if not isinstance(plan, list) or not plan:
                raise ValueError("计划必须是非空列表。")

            if not all(
                isinstance(step, str) and step.strip()
                for step in plan
            ):
                raise ValueError("每个步骤都必须是非空字符串。")

            plan = [step.strip() for step in plan]

            print("\n解析后的计划：")
            for index, step in enumerate(plan, start=1):
                print(f"{index}. {step}")

            return plan

        except (ValueError, SyntaxError) as exc:
            print(f"计划解析失败：{exc}")
            print(f"模型原始输出：\n{response}")
            return []


class Executor:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def execute(self, question: str, plan: list[str]):
        history = []
        final_answer = ""

        plan_text = "\n".join(
            f"{index}. {step}"
            for index, step in enumerate(plan, start=1)
        )

        print("\n--- 开始执行计划 ---")

        for index, step in enumerate(plan, start=1):
            print(f"\n步骤 {index}/{len(plan)}：{step}")

            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan_text,
                history="\n\n".join(history) or "暂无",
                current_step=step,
            )

            response = self.llm_client.think(
                messages=[{"role": "user", "content": prompt}]
            )

            if not response or not response.strip():
                print(f"步骤 {index} 没有得到有效结果，停止执行。")
                return None

            final_answer = response.strip()

            # 保存当前结果，下一轮将它放进提示词。
            history.append(
                f"步骤 {index}：{step}\n"
                f"结果：{final_answer}"
            )

        return final_answer


class PlanAndSolveAgent:
    def __init__(self, llm_client):
        self.planner = Planner(llm_client)
        self.executor = Executor(llm_client)

    def run(self, question: str):
        print(f"\n用户问题：{question}")

        # 先生成完整计划。
        plan = self.planner.plan(question)

        if not plan:
            print("没有有效计划，任务结束。")
            return None

        # 再按照计划逐步执行。
        answer = self.executor.execute(question, plan)

        if answer is None:
            return None

        print(f"\n--- 最终答案 ---\n{answer}")
        return answer


if __name__ == "__main__":
    llm = HelloAgentsLLM()
    agent = PlanAndSolveAgent(llm)

    question = (
        "一个水果店周一卖出了15个苹果。"
        "周二卖出的苹果数量是周一的两倍。"
        "周三卖出的数量比周二少了5个。"
        "请问这三天总共卖出了多少个苹果？"
    )

    agent.run(question)
