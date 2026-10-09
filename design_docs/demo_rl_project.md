---
name: demo_rl_project
description: demo_rl 新项目 — 把 sp32_dri.py 拉格朗日松弛优化迭代成强化学习;含 SP32 结构与 RL 映射
metadata: 
  node_type: memory
  type: project
  originSessionId: 9d911d18-df4b-44da-868c-9eca5cbc0b91
---

# demo_rl 项目(2026-09-09 启动了解)

目标:把 dintl-driver-spark `strategy/odt/odt_daily_update/lib/sp32_dri.py` 的拉格朗日松弛优化(SP32.prob_solve)迭代成强化学习。
代码目录:`/Users/didi/workspace/python-project/demo_rl`(独立于 dintl-driver-spark,py3.9 venv,目前只有 `cafune_rl.py`)。

## 双AI分工协作(2026-09-17用户定,两实例共读此节)

| 线 | 主导 | 范围 |
|---|---|---|
| **数据流线** | GM实例(本memory作者) | cafune_rl/cafune_rl_time_series数据流梳理审计: base_v9源表事实(SQL排雷A~G悬案)/春蓉管线审计(她自知有问题但忙于上线流程无暇管)/融合层数据管道(OracleProvider/BudgetTracker数据侧) |
| **时序建模线** | 另一大模型实例 | 标量日总label契约下的 V0→GBDT→深度模型/回测协议 |

- **交叉review**: 每方里程碑产物给对方审,沿用label review格式(评审落`*_review.md`文件,作者回应附同文件);裁决原则=数学裁决认数学,判断裁决看证据等级
- 背景: 春蓉知道cafune_rl数据流有问题,在忙另外的上线流程没时间管——数据流梳理由用户侧主导,她的线并行;她的"已知问题清单"是最高价值情报,待用户去要
- 接口: 数据流线供=干净聚合表+《源表事实清单》+口径文档;建模线按[[demo_rl_forecast_integration_design]]的标量日总契约实现;建模发现的数据异常回流数据流线修

## SP32 核心结构(被替换的对象)

- 每天每城市:百万级冒泡(bubble),每个冒泡 K 个调价动作(treat_list,如 [-0.15..0.15] 11 档),uplift 模型(ecr/cr)给每档预测 call/trip/gmv/cost/gp/cr/tr 矩阵(mat_val,形状 n_bubble × K)。
- 优化:选每冒泡一档,最大化 target(gmv/gp/trip),满足约束(trip/tr/cr/cost/...)阈值。
- 解法:拉格朗日松弛。固定 λ → 每冒泡独立 argmax(target_reg + Σλ_i·constraint_i) → 统计约束增量 → 次梯度更新 λ(带 min_lambda=0.0001 下限) → 迭代至收敛或 iter_th=5000。backup 机制保留最优可行 λ;最终不可行则回滚。
- 痛点:step/λ 初始化靠启发式(λ·asp/step_parm)+ 大量国家/城市 hardcoded hack(CL/AR/CO/CR/DO/EG、55000207 等)、震荡收敛慢、超时 AutoErrorCode 16。
- 产出 coef=λ_dict 写 Apollo → 线上 bub 请求时本地用 λ+模型预测做 argmax。即线上决策=线性打分(λ 为参数),日更只产出 λ。

## cafune_rl.py 现状(数据探索脚本)

复用 BR/16/55000233 simu_v9 数据+tf/xgb/cf 模型重建每档预测矩阵(与 prob_solve 的 mat_val 同构),再从 bub 日志(lambdaGMV/lambdaTrip/lambdaGP/lambdaCr 等 13 列)取线上真实 λ 按天统计、与 simu 特征 merge。这正是在构造离线 RL 需要的 (context, λ, 各档预测结果) 数据/仿真器原料。

## RL 映射(两种 framing)

1. **λ 作为 action**(线上零改动): policy π_θ(λ|城市特征/阈值/base指标),历史 (城市,日,λ,实际结果) 做离线 RL/contextual bandit;或 learning-to-optimize 学更新规则替代次梯度。
2. **端到端策略**: action=每冒泡/每 bin(weekday×dist×hour) 的调价档,离线策略学习(预测矩阵即仿真器);需改线上。

相关: [[422_daily_update_execution_chain]](dri_only.py 链路产出这些 mat_val 的上游)
