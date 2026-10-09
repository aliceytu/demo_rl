---
name: demo_rl_v0_first_results
description: "Claude侧2026-09-17晚V0首跑结论:数据三关案(null剔除2天/时区bj口径/weekday半悬案)+裁判基线V0a=11.76%+V0b=5.17%,供数据流梳理侧与建模决策参考"
metadata:
  node_type: memory
  type: project
  originSessionId: 9d911d18-df4b-44da-868c-9eca5cbc0b91
---

# demo_rl 时序预估 V0 首跑结论(2026-09-17 晚, Claude 侧)

数据文件(本地 tarnet 目录)= BR 77 城 × 503 天 × 96 窗 = 3,718,176 行, 全 product 16。分析产物在 demo_rl/analysis/(v0_analysis.py / null_diagnose.py / track_check.py), 输出 analysis/output/。

## 数据三个关案(排查实录, 供你梳理数据流时对照)

1. **null 关案**: 全表 null 27,199 = 两个空白日 14,784(2025-05-17/08-22 各 77×96 全 null, 有 call 有冒泡但 gmv 缺失, 已定剔除) + 背景 12,415(工作日当地凌晨 2-4:30, 两轨 gmv 皆空而 call>0 占 86%, "无完单"的真实零, null→0 合法)。explore_gmv 从不 null 的原因=生成脚本 `sum(when(exp,gmv).otherwise(0))` 带兜底, all_gmv 裸 sum(gmv) 无兜底(SQL 写法差异)。**all_gmv 含 explore 轨**(F.sum("gmv") 全行求和); explore_ratio 是行级占比 29.4% vs GMV 占比 24.5% 的差=explore 单均 gmv 更小。
2. **时区关案**: window_idx 是 **bj 口径**。三证据: 日内占比峰值窗 16-20(=当地晚高峰17-19)/早高峰 72-80/null 集中 bj 13:15-15:30(=当地凌晨2:15-4:30)。与 RL env 切窗一致, 无需 +11h 修正。
3. **weekday 半悬案(唯一未关)**: weekday 列与 bj_date 派生的偏移仅 {0,6} 且按月变化(20-87%)——列按行用当地日期派生, 但 bj 日前 44 窗(bj 0-11点)属当地前一天 → 同一 (city,date) 内 weekday 不自洽。**修法**: 按当地营业日重划日界(window≥44 归当日, <44 归前一日), weekday 从当地日期派生。修完重跑 V0a 应再降。注意: 这影响预估器的"日"口径定义, 与 env 侧对齐时需确认(见 [[demo_rl_forecast_label_claude_review]] 小旗子③同款基准问题)。

## V0 裁判基线(建模入场线)

- V0a 裸模板(历史同weekday中位日总) WMAPE = **11.76%**
- V0b 模板×校准因子(实际前缀/历史同前缀中位): 全桶平均 **5.17%**; 分桶单调: 开盘2h 8.64%(-26.5%) → 中午 5.04%(-57%) → 尾盘 0.56%(-95%) ——"越晚越准"实证曲线
- V0a 的 P10-P90 穿越率 70.5%(偏窄, 残差厚尾, V1 带宽需校准)
- 城市差异: 最好 9.6%(55000439) / 最差 20.9%(55000548)、20.2%(55000082)
- 口径: explore_ratio 稳定 29.3%; avg_dp 月度漂移(2025-12 峰值 1.56 / 2026-07/08 低谷 1.25)

## 已定决策

- 剔除 2025-05-17/08-22(demo_rl/data_preprocess/clean_data.py 为训练管线共用加载入口, null→0 一并处理)
- 建模入场线 = 打不过 V0a(11.76%)的模型不该存在; 前缀特征(累计/与历史比值/近窗动量)是第一生产力
- 下一步候选: weekday 日界修复重跑 V0 / 尾部城市画像 / GBDT 表格版

相关: [[demo_rl_forecast_label_claude_review]] [[demo_rl_gmv_forecast_discussion]] [[demo_rl_task_division]]
