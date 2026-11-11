from HelloAgentsLLM import HelloAgentsLLM


INITIAL_PROMPT_TEMPLATE = """
你是一位 Python 程序员。
请根据下面的任务编写代码。

任务：
{task}

要求：
- 包含完整的函数签名和文档字符串。
- 正确处理边界情况。
- 只输出 Python 代码，不要附加解释或 Markdown 代码块。
"""


REFLECT_PROMPT_TEMPLATE = """
你是一位代码评审员。
请审查下面的代码，检查正确性、边界情况和算法效率。

原始任务：
{task}

待审查的代码：
{code}

如果发现明确需要修改的问题：
请列出具体问题，以及可操作的修改建议。
不要输出完整的新代码。

如果没有明确需要修改的问题：
请只输出四个字：无需改进
"""


REFINE_PROMPT_TEMPLATE = """
你是一位 Python 程序员。
请根据评审反馈，修改上一版代码。

原始任务：
{task}

上一版代码：
{code}

评审反馈：
{feedback}

请保留正确的部分，修复问题。
只输出完整的新版本 Python 代码，
不要附加解释或 Markdown 代码块。
"""


class Memory:
    """保存当前任务中的代码版本和评审反馈。"""

    def __init__(self):
        self.records = []

    def add_record(self, record_type, content):
        self.records.append({
            "type": record_type,
            "content": content,
        })
        print(f"\n记忆已更新：{record_type}")

    def get_last_execution(self):
        for record in reversed(self.records):
            if record["type"] == "execution":
                return record["content"]
        return None

    def get_trajectory(self):
        return "\n\n".join(
            f"记录类型：{record['type']}\n{record['content']}"
            for record in self.records
        )


class ReflectionAgent:
    def __init__(self, llm_client, max_iterations=2):
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.memory = Memory()

    def _get_llm_response(self, prompt):
        response = self.llm_client.think(
            messages=[{"role": "user", "content": prompt}]
        )

        if not response or not response.strip():
            raise RuntimeError("模型没有返回有效内容。")

        return response.strip()

    def run(self, task):
        # 每个新任务使用独立的短期记忆。
        self.memory = Memory()

        stop_reason = (
            "达到轮次上限。"
            "最新版本尚未经过下一轮评审。"
        )

        print(f"\n任务：{task}")

        try:
            # 1. 生成初稿。
            print("\n--- 初始生成 ---")
            initial_code = self._get_llm_response(
                INITIAL_PROMPT_TEMPLATE.format(task=task)
            )
            self.memory.add_record("execution", initial_code)

            # 2. 反复评审和修改。
            for index in range(1, self.max_iterations + 1):
                print(
                    f"\n--- 第 {index}/{self.max_iterations} 轮 ---"
                )

                last_code = self.memory.get_last_execution()

                print("\n正在评审……")
                feedback = self._get_llm_response(
                    REFLECT_PROMPT_TEMPLATE.format(
                        task=task,
                        code=last_code,
                    )
                )
                self.memory.add_record("reflection", feedback)

                # 严格匹配，避免把“并非无需改进”等句子误判。
                if feedback == "无需改进":
                    stop_reason = "模型评审认为无需改进。"
                    break

                print("\n正在根据反馈修改……")
                refined_code = self._get_llm_response(
                    REFINE_PROMPT_TEMPLATE.format(
                        task=task,
                        code=last_code,
                        feedback=feedback,
                    )
                )
                self.memory.add_record("execution", refined_code)

        except RuntimeError as exc:
            stop_reason = (
                f"调用中断：{exc}"
                "保留最近一次成功生成的版本。"
            )

        final_code = self.memory.get_last_execution()

        print(f"\n停止原因：{stop_reason}")

        if final_code is None:
            print("本次没有生成有效代码。")
            return None

        print(f"\n--- 最终生成的代码 ---\n{final_code}")
        return final_code


if __name__ == "__main__":
    llm = HelloAgentsLLM()
    agent = ReflectionAgent(llm, max_iterations=2)

    task = (
        "编写名为 find_primes(n) 的 Python 函数，"
        "返回闭区间 [1, n] 内所有素数组成的升序列表。"
        "当 n 小于 2 时返回空列表。"
        "不使用第三方库，兼顾正确性和算法效率。"
    )

    agent.run(task)
