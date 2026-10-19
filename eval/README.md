# 离线评估（eval）

用固定用例集跑 Travel Agent 的端到端评估，输出**成功率、耗时、token 消耗**，用于回答"任务样本与成功率、耗时成本"这类问题，也用于改 Prompt / 换模型后的回归对比。

## 设计说明（重要）

- **工具层复用项目自身实现**：直接使用 `travel_agent/tools/tool_registry.py` 的工具定义与 `MCPToolManager`，不是另写一套假工具。
- **编排层是精简版 ReAct**：不加载 Streamlit，因此测的是"工具链 + 模型输出质量"，**不包含 UI 渲染耗时**。
- **不伪造外部数据**：MCP 服务未配置时，工具会返回项目原有的结构化错误，评估会如实反映"无实时数据时能否给出可用方案"。

## 用例集

`cases.json`：20 条，4 类各 5 条

| 类别 | 覆盖 |
| --- | --- |
| simple | 单目的地、只要景点/美食 |
| full | 出发地 + 目的地 + 天数 + 预算，要完整方案 |
| multi | 2~3 个城市的路线排序 |
| complex | 老人小孩、极低预算、团建、过敏忌口、极端天气 |

每条用例的判定规则：无异常 + 输出长度 ≥ `min_chars` + 命中 `expect_any` 任一关键词。

## 运行

```bash
# 1) 只校验用例集（不调用模型，不花钱）
python eval/run_eval.py --selfcheck

# 2) 冒烟 3 条
python eval/run_eval.py --limit 3

# 3) 全量 20 条
python eval/run_eval.py

# 4) 折算成本（价格按官方价目表自行填写，单位：元/千 token）
python eval/run_eval.py --price-in 0.0008 --price-out 0.002
```

依赖环境变量：`DASHSCOPE_API_KEY`（必填）；`DEEPSEEK_API_KEY`（可选，供 `r1_analysis` 使用）。

## 输出

- `eval/results/report-<时间戳>.json`：逐条明细 + 汇总（成功率/平均耗时/token/分类通过率）
- `eval/results/report-<时间戳>.md`：可直接贴进文档或发给面试官

## 已知限制

- 只测最终答案质量，不测多轮对话与 UI；
- `expect_any` 是关键词命中，属于粗粒度判定；对"事实正确性"不做核验；
- MCP 服务全部不可用时，成功率会显著偏低——这正是"无工具兜底能力"的真实反映，建议在报告里注明当时的数据源状态。
