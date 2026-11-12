from pprint import pprint
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END


# 1. 定义状态：整个流程需要保存哪些数据
class AgentState(TypedDict):
    current_task: str
    messages: list[str]
    final_answer: str


# 2. 定义规划节点：读取任务，返回需要更新的字段
def planner_node(state: AgentState):
    round_number = len(state["messages"]) // 2 + 1
    plan = f"第{round_number}轮计划：处理任务“{state['current_task']}”"

    print("\n>>> 正在执行 planner 节点")

    # 创建新列表，保留已有记录并追加计划
    return {
        "messages": state["messages"] + [plan],
    }


# 3. 定义执行节点：读取最新计划，返回执行结果
def executor_node(state: AgentState):
    latest_plan = state["messages"][-1]
    result = f"模拟执行完成：{latest_plan}"

    print("\n>>> 正在执行 executor 节点")

    return {
        "messages": state["messages"] + [result],
        "final_answer": result,
    }


# 4. 条件函数：读取状态，决定下一步路线
def should_continue(state: AgentState):
    message_count = len(state["messages"])

    if message_count < 3:
        print(f">>> 条件判断：已有 {message_count} 条消息，返回 planner")
        return "continue"

    print(f">>> 条件判断：已有 {message_count} 条消息，结束流程")
    return "finish"


def main():
    # 5. 创建图，并注册节点
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)

    # 6. 添加固定连线
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")

    # 7. 添加条件连线
    graph.add_conditional_edges(
        "executor",
        should_continue,
        {
            "continue": "planner",
            "finish": END,
        },
    )

    # 8. 编译成可运行的图
    app = graph.compile()

    # 9. 提供初始状态
    initial_state: AgentState = {
        "current_task": "比较 AutoGen 和 LangGraph",
        "messages": [],
        "final_answer": "",
    }

    # 10. 逐步输出完整状态
    final_state = initial_state

    for index, snapshot in enumerate(
        app.stream(initial_state, stream_mode="values"),
        start=0,
    ):
        print(f"\n--- 状态快照 {index} ---")
        pprint(snapshot, sort_dicts=False)
        final_state = snapshot

    print("\n最终答案：")
    print(final_state["final_answer"])


if __name__ == "__main__":
    main()