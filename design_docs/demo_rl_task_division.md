---
name: demo_rl_task_division
description: "demo_rl 双线分工定案(2026-09-17): 数据流梳理由原设计者主导 / 时序GMV预估由Claude主导,两模块完成后交叉验证和review;label契约已对齐(标量日总版)"
metadata:
  node_type: memory
  type: project
  originSessionId: 9d911d18-df4b-44da-868c-9eca5cbc0b91
---

# demo_rl 双线任务分工(2026-09-17 用户定案)

用户安排两个并行任务,完成后**两模块交叉验证和 review**:

## 任务一:cafune_rl 数据流梳理(原设计者主导)

当前数据流有一些问题(用户已指出但未指明具体现象——见 [[demo_rl_gmv_forecast_discussion]] 的 2026-09-15 收工状态)。梳理对象 = cafune_rl.py 的 simu_v9 → 预测矩阵 → env 数据链路,与 [[demo_rl_forecast_integration_design]] 的插点改造衔接。

## 任务二:时序 GMV 预估模型(Claude 主导)

在另一 session(demo_rl 目录)执行。已对齐的 label 契约终稿(见 [[demo_rl_forecast_label_claude_review]] 回应段):

- **主 label = 城内归一化日总**(÷历史 weekday 中位日总,防大城市主导跨城 loss),三头 P10/P50/P90
- 样本 = (城市, 日, t₀) 条件化,同日 96 样本共享 label(诚实结构)
- **接口传日总分位数不传 rem**(尾盘 rem=两接近数相减,相对误差放大;used_ratio=已用/日总由消费端自算)
- 占比曲线保留为可选辅助头(窗口感知 th 升级路径+诊断),不进契约

动工顺序(原 integration_design 定稿): V0模板+回测管线(裁判先行)→ env 侧 dynamic_tr_th → 注偏差测反馈 → V1(iTransformer 多城) → 与春蓉接口契约。

数据状态: `odt_forecast_city_gmv_15min_day` 已上线(77城×314天×96窗=232万行);开放项=补 188 天(daily=01 孤岛,世界杯素材在缺口里)、dt_hm 时区验证(第 N 次顺延,决定+11h 修正)、weekday 归属同验。

## 交叉验证约定

- 任务二产出预估器时,任务一负责从 env 消费者侧 review(口径一致性:账C表口径 vs simu 口径混搭已在 [[demo_rl_forecast_label_claude_review]] 小旗子②,oracle 双跑显式观察)
- 任务一梳理完数据流后,任务二侧 review 预估器与 env 的衔接契约(接口签名标量日总版已统一)

相关: [[demo_rl_forecast_integration_design]] [[demo_rl_forecast_label_claude_review]] [[demo_rl_gmv_forecast_discussion]]
