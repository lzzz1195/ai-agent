import os
from pathlib import Path

from dotenv import load_dotenv
from serpapi import GoogleSearch


# 明确加载当前文件旁边的 .env。
load_dotenv(Path(__file__).with_name(".env"))


def search(query: str) -> str:
    """搜索网页，将结果整理成模型能够读取的文字。"""
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "错误：未配置 SERPAPI_API_KEY。"

    print(f"正在搜索：{query}")

    try:
        results = GoogleSearch({
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",
            "hl": "zh-cn",
        }).get_dict()

        if results.get("error"):
            return f"搜索失败：{results['error']}"

        # 优先返回搜索引擎提供的直接答案。
        answer_box = results.get("answer_box") or {}
        answer = answer_box.get("answer") or answer_box.get("snippet")
        if answer:
            return str(answer)

        knowledge = results.get("knowledge_graph") or {}
        if knowledge.get("description"):
            return knowledge["description"]

        # 没有直接答案时，返回前三条网页结果。
        items = results.get("organic_results", [])[:3]
        snippets = [
            f"[{i}] {item.get('title', '')}\n"
            f"{item.get('snippet', '')}\n"
            f"来源：{item.get('link', '')}"
            for i, item in enumerate(items, start=1)
        ]

        return "\n\n".join(snippets) or "没有找到相关搜索结果。"

    except Exception as exc:
        return f"搜索发生错误：{exc}"


class ToolExecutor:
    """保存工具名称、用途说明和对应的 Python 函数。"""

    def __init__(self):
        self.tools = {}

    def registerTool(self, name, description, func):
        self.tools[name] = {
            "description": description,
            "func": func,
        }

    def getTool(self, name):
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self):
        return "\n".join(
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        )


if __name__ == "__main__":
    executor = ToolExecutor()
    executor.registerTool(
        "Search",
        "搜索互联网，获取实时信息、事实和资料。",
        search,
    )

    print("可用工具：")
    print(executor.getAvailableTools())

    tool = executor.getTool("Search")
    print("\n搜索结果：")
    print(tool("华为最新发布的手机型号及发布日期"))
