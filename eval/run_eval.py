#!/usr/bin/env python3
"""
Travel Agent 离线评估脚本

用途
----
用固定用例集跑端到端评估，输出：成功率、单例耗时、token 消耗（可选折算成本）、
工具调用次数与失败明细。回答"任务样本与成功率、耗时成本"这类问题。

说明
----
- 工具层直接复用项目自身的 `tool_registry` + `MCPToolManager`，不是另写一套；
- 编排层是精简版 ReAct（不依赖 Streamlit），因此测的是"工具链 + 模型输出质量"，
  不是 UI 全链路耗时；
- MCP 服务未配置时，工具会返回结构化错误（这是项目现有行为），评估结果会如实
  反映"无外部数据时还能不能给出可用方案"。

用法
----
    python eval/run_eval.py --selfcheck           # 只校验用例集，不调用模型
    python eval/run_eval.py --limit 3             # 先跑 3 条冒烟
    python eval/run_eval.py                       # 跑全量 20 条
    python eval/run_eval.py --price-in 0.0008 --price-out 0.002   # 折算成本

环境变量
--------
    DASHSCOPE_API_KEY   必填（qwen-plus 走 DashScope 兼容接口）
    DEEPSEEK_API_KEY    可选（r1_analysis 工具）
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CASES = ROOT / "eval" / "cases.json"
RESULT_DIR = ROOT / "eval" / "results"


# ----------------------------------------------------------------------------
# 用例加载
# ----------------------------------------------------------------------------
def load_cases(path: Path, limit: int = 0, ids: str = "") -> list:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    if ids:
        wanted = {x.strip() for x in ids.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in wanted]
    if limit > 0:
        cases = cases[:limit]
    return cases


def validate_cases(cases: list) -> list:
    problems = []
    if not cases:
        problems.append("用例集为空")
    seen = set()
    for i, c in enumerate(cases, 1):
        for key in ("id", "category", "query", "min_chars"):
            if key not in c:
                problems.append(f"第 {i} 条缺少字段 {key}")
        if c.get("id") in seen:
            problems.append(f"id 重复: {c.get('id')}")
        seen.add(c.get("id"))
        if not isinstance(c.get("expect_any", []), list):
            problems.append(f"{c.get('id')}: expect_any 必须是数组")
    return problems


# ----------------------------------------------------------------------------
# 异步桥：MCP 客户端是异步的，AgentExecutor 是同步的
# ----------------------------------------------------------------------------
class AsyncBridge:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coro, timeout: float = 90.0):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=timeout)

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)


# ----------------------------------------------------------------------------
# Token / 工具调用统计
# ----------------------------------------------------------------------------
def build_callback_handler():
    from langchain_core.callbacks import BaseCallbackHandler

    class Stats(BaseCallbackHandler):
        def __init__(self):
            self.input_tokens = 0
            self.output_tokens = 0
            self.tool_calls = 0
            self.tool_errors = []

        def on_llm_end(self, response, **kwargs):
            try:
                usage = (response.llm_output or {}).get("token_usage") or {}
                self.input_tokens += int(usage.get("prompt_tokens", 0) or 0)
                self.output_tokens += int(usage.get("completion_tokens", 0) or 0)
            except Exception:
                pass

        def on_tool_start(self, serialized, input_str, **kwargs):
            self.tool_calls += 1

        def on_tool_error(self, error, **kwargs):
            self.tool_errors.append(str(error)[:200])

    return Stats()


# ----------------------------------------------------------------------------
# 工具构建（复用项目自身的注册表与 MCP 管理器）
# ----------------------------------------------------------------------------
def build_tools(bridge: AsyncBridge, use_rag: bool = True):
    from langchain.tools import Tool
    from aggentic_RAG.travel_agent.tools.tool_registry import AVAILABLE_TOOLS
    from aggentic_RAG.travel_agent.tools.mcp_tools import MCPToolManager

    manager = MCPToolManager()
    bridge.run(manager.initialize(), timeout=120)

    tools = []

    # RAG（可选：向量库存在且配置了 embedding key 时才启用）
    if use_rag:
        try:
            import os as _os
            from langchain_chroma import Chroma
            from langchain_community.embeddings import DashScopeEmbeddings
            from langchain.tools.retriever import create_retriever_tool

            db_dir = ROOT / "aggentic_RAG" / "data" / "travel_vectordb"
            if db_dir.exists() and _os.getenv("DASHSCOPE_API_KEY"):
                vectordb = Chroma(
                    persist_directory=str(db_dir),
                    embedding_function=DashScopeEmbeddings(
                        model="text-embedding-v4",
                        dashscope_api_key=_os.getenv("DASHSCOPE_API_KEY"),
                    ),
                )
                retriever = vectordb.as_retriever(search_kwargs={"k": 3})
                tools.append(
                    create_retriever_tool(
                        retriever=retriever,
                        name="rag_search",
                        description="从本地旅游知识库检索攻略、景点与美食信息。输入城市或景点名。",
                    )
                )
        except Exception as e:
            print(f"⚠️  RAG 工具未启用: {type(e).__name__}: {e}")

    # MCP 工具：按注册表逐条包装
    for definition in AVAILABLE_TOOLS:
        if definition.tool_type != "mcp":
            continue

        def _make(defn):
            server = defn.server_name
            tool_name = defn.mcp_tool_name

            def _run(**kwargs):
                try:
                    return bridge.run(manager.call_tool(server, tool_name, **kwargs))
                except Exception as e:
                    return json.dumps(
                        {"error": f"{type(e).__name__}: {e}", "server": server},
                        ensure_ascii=False,
                    )

            return _run

        tools.append(
            Tool(
                name=definition.name,
                description=definition.description,
                func=_make(definition),
            )
        )

    return tools, manager


def build_llm(model: str, temperature: float):
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY，无法调用 qwen-plus")
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature,
    )


REACT_PROMPT = """你是一名专业的旅行规划助手。你可以使用以下工具：

{tools}

可用工具名：{tool_names}

规则：
1. 需要实时信息（车次、天气、酒店、景点、航班、黄历）时必须调用工具，不要编造。
2. 工具返回错误时，明确说明该项数据不可用，并用已有信息给出可执行方案，不要假装查到了。
3. 最终答案包含：行程安排、交通、住宿、预算（如用户提供）、注意事项。

使用格式：
Question: 用户的问题
Thought: 你的思考
Action: 工具名（必须是 [{tool_names}] 之一）
Action Input: 工具参数
Observation: 工具返回
...（可重复 Thought/Action/Action Input/Observation）
Thought: 我已经知道最终答案
Final Answer: 给用户的中文方案

Question: {input}
Thought:{agent_scratchpad}"""


def run_one(case: dict, executor, stats, timeout_s: float) -> dict:
    start = time.perf_counter()
    result = {
        "id": case["id"],
        "category": case.get("category", ""),
        "query": case["query"],
        "ok": False,
        "reason": "",
    }
    try:
        out = executor.invoke(
            {"input": case["query"]},
            config={"callbacks": [stats]},
        )
        text = (out or {}).get("output", "") if isinstance(out, dict) else str(out)
        result["chars"] = len(text)
        if len(text) < int(case.get("min_chars", 100)):
            result["reason"] = f"输出过短({len(text)} < {case.get('min_chars')})"
        else:
            expect = case.get("expect_any") or []
            if expect and not any(k in text for k in expect):
                result["reason"] = "关键词未命中: " + "/".join(expect)
            else:
                result["ok"] = True
                result["reason"] = "ok"
        result["preview"] = text[:200].replace("\n", " ")
    except Exception as e:
        result["reason"] = f"{type(e).__name__}: {str(e)[:200]}"
        result["traceback"] = traceback.format_exc()[-800:]
    result["elapsed_s"] = round(time.perf_counter() - start, 2)
    return result


def summarize(results: list, model: str, price_in: float, price_out: float) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r["ok"])
    latencies = [r["elapsed_s"] for r in results if r.get("elapsed_s")]
    tokens_in = sum(r.get("tokens", {}).get("input", 0) for r in results)
    tokens_out = sum(r.get("tokens", {}).get("output", 0) for r in results)
    cost = tokens_in / 1000 * price_in + tokens_out / 1000 * price_out
    by_cat = {}
    for r in results:
        c = r.get("category", "unknown")
        b = by_cat.setdefault(c, {"total": 0, "passed": 0})
        b["total"] += 1
        b["passed"] += 1 if r["ok"] else 0
    return {
        "model": model,
        "total": total,
        "passed": passed,
        "success_rate": round(passed / total, 3) if total else 0,
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "max_latency_s": max(latencies) if latencies else 0,
        "tokens": {"input": tokens_in, "output": tokens_out},
        "est_cost_cny": round(cost, 4) if (price_in or price_out) else None,
        "by_category": by_cat,
    }


def write_reports(payload: dict) -> tuple:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = RESULT_DIR / f"report-{stamp}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    s = payload["summary"]
    lines = [
        f"# Travel Agent 评估报告 {payload['timestamp']}",
        "",
        f"- 模型：`{s['model']}`",
        f"- 用例：{s['total']}（通过 {s['passed']}）",
        f"- 成功率：**{s['success_rate'] * 100:.1f}%**",
        f"- 平均耗时：{s['avg_latency_s']}s（最大 {s['max_latency_s']}s）",
        f"- Token：输入 {s['tokens']['input']} / 输出 {s['tokens']['output']}",
    ]
    if s.get("est_cost_cny") is not None:
        lines.append(f"- 估算成本：¥{s['est_cost_cny']}")
    lines += ["", "## 分类结果", "", "| 类别 | 通过/总数 |", "| --- | --- |"]
    for cat, b in s["by_category"].items():
        lines.append(f"| {cat} | {b['passed']}/{b['total']} |")
    lines += ["", "## 明细", "", "| 用例 | 类别 | 结果 | 耗时 | 说明 |", "| --- | --- | --- | --- | --- |"]
    for r in payload["cases"]:
        mark = "✅" if r["ok"] else "❌"
        lines.append(
            f"| {r['id']} | {r['category']} | {mark} | {r.get('elapsed_s', 0)}s | {r['reason'][:60]} |"
        )
    md_path = RESULT_DIR / f"report-{stamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Travel Agent 离线评估")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="")
    parser.add_argument("--model", default=os.getenv("EVAL_MODEL", "qwen-plus"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--price-in", type=float, default=0.0, help="每千 input token 价格(元)")
    parser.add_argument("--price-out", type=float, default=0.0, help="每千 output token 价格(元)")
    parser.add_argument("--selfcheck", action="store_true", help="只校验用例集")
    args = parser.parse_args()

    cases = load_cases(Path(args.cases), args.limit, args.ids)
    problems = validate_cases(cases)

    print("=" * 64)
    print("Travel Agent 评估")
    print("=" * 64)
    print(f"用例文件: {args.cases}")
    print(f"用例数量: {len(cases)}")
    by_cat = {}
    for c in cases:
        by_cat[c.get("category", "unknown")] = by_cat.get(c.get("category", "unknown"), 0) + 1
    print("分类分布: " + ", ".join(f"{k}={v}" for k, v in by_cat.items()))

    if problems:
        print("\n❌ 用例集校验失败：")
        for p in problems:
            print("  -", p)
        return 2

    if args.selfcheck:
        print("\n✅ 用例集校验通过（未调用模型）")
        return 0

    if not os.getenv("DASHSCOPE_API_KEY"):
        print("\n❌ 缺少 DASHSCOPE_API_KEY，无法运行。可先 export 后再跑。")
        return 2

    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_core.prompts import PromptTemplate

    bridge = AsyncBridge()
    executor = None
    try:
        print("\n初始化工具链（MCP + RAG）...")
        tools, manager = build_tools(bridge, use_rag=not args.no_rag)
        health = manager.get_health()
        print(f"工具数: {len(tools)} | MCP 已连接: {health['connected'] or '无'}")
        if health["failed"]:
            print("MCP 未连接:")
            for name, reason in health["failed"].items():
                print(f"  - {name}: {reason}")

        llm = build_llm(args.model, args.temperature)
        prompt = PromptTemplate.from_template(REACT_PROMPT)
        agent = create_react_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=20,
        )

        results = []
        print("\n开始评估...\n")
        for i, case in enumerate(cases, 1):
            stats = build_callback_handler()
            r = run_one(case, executor, stats, args.timeout)
            r["tokens"] = {
                "input": stats.input_tokens,
                "output": stats.output_tokens,
            }
            r["tool_calls"] = stats.tool_calls
            if stats.tool_errors:
                r["tool_errors"] = stats.tool_errors[:3]
            results.append(r)
            mark = "✅" if r["ok"] else "❌"
            print(
                f"[{i}/{len(cases)}] {mark} {r['id']:<12} {r['elapsed_s']:>6.1f}s "
                f"tools={r['tool_calls']:<2} {r['reason'][:48]}"
            )

        summary = summarize(results, args.model, args.price_in, args.price_out)
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "summary": summary,
            "cases": results,
        }
        json_path, md_path = write_reports(payload)

        print("\n" + "=" * 64)
        print(f"成功率: {summary['success_rate'] * 100:.1f}%  ({summary['passed']}/{summary['total']})")
        print(f"平均耗时: {summary['avg_latency_s']}s")
        print(f"Token: in={summary['tokens']['input']} out={summary['tokens']['output']}")
        if summary["est_cost_cny"] is not None:
            print(f"估算成本: ¥{summary['est_cost_cny']}")
        print(f"报告: {json_path}")
        print(f"      {md_path}")
        print("=" * 64)
        return 0
    finally:
        try:
            if executor is not None:
                pass
        finally:
            bridge.close()


if __name__ == "__main__":
    sys.exit(main())
