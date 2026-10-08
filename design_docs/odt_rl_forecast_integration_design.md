---
name: odt_rl_forecast_integration_design
description: "时序预估接入cafune_rl的定稿设计:env侧三插点(动态th/state加3维/step累计账本)+预估规格清单+动工顺序,用户与春蓉的接口契约"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9d911d18-df4b-44da-868c-9eca5cbc0b91
---

# 时序预估接入设计定稿(2026-09-14,基于用户已读懂env/agent边界后的精确化)

## 两份代码总结与1367公式血缘(2026-09-17定稿,常被回查)

**cafune_rl.py**(春蓉,可执行主线~4400行): 段落=数据与预测管线(simu_v9+3模型→每冒泡×每档预测矩阵)/线上λ分析/3.1日级λ回放baseline/3.2 v1原型/3.3 v2完整版(15min窗,20维state,27动作,Double DQN,main_run_example)。本质=把"λ每天离线解一次"改成"RL每15min在线调",拉格朗日argmax降级为环境执行机构,DQN学调度。

**cafune_rl_time_series.py**(用户,伪代码设计稿,#wjz恢复标记): 五模块=ForecastProvider/BudgetTracker/env三处改造/reward双保险/验证四步。对前者: agent零改动,state 20→23自动。依赖odt_forecast_city_gmv_15min_day表。

**与wjz_tr_control_v3的1367公式关系=直系血亲(四换一留)**:
- 不变的骨架: 修正量=(真实累计TR−simu累计TR)×(已用GMV/剩余GMV),阈值=基准∓修正量
- 尺度: 周→15min窗; 分子: Apollo回读→BudgetTracker; 基准: simu表→fixed_baseline_cache; **分母: simu表→时序预估器(用户支线的物理位置)**
- 符号: ≤约束调低收紧 → ≥约束调高收紧(减数对调,最易装反)
- 约束对象: odt_fee_tr→v_tr(GP口径,简化版;完整对齐需env补odt_fee列,待定项); gap_automatic/gap_handle/fluctuate clamp暂未移植(clamp建议补,防尾盘过激)
- 角色变化: v3里公式=生产控制器本身; 融合层里=降级为env reward规则(内环),RL外环学预判
- 系谱: sp32 λ(周尺度离线对偶)→1367(周尺度在线对偶,v3)→th(t)(窗尺度在线对偶,env)→RL(快尺度学习primal) = 三时间尺度primal-dual; 用户在v3写的控制器与在预估支线接的分母服务同一条公式的两个尺度

**用户理解的里程碑**(2026-09-16/17确认): P10~P90来源链条=历史相似情境多次实现→散布→排序取名次→分位数回归学成函数(pinball穿越率校准宽度); 远窗宽近窗窄=相似情境散布的宽窄; 分位数不可加→辅助总量头(逐窗P50可加,P10/P90不可); label定稿=all轨日内占比曲线(训练)/剩余GMV分位数(交付); 距离感知住在逐窗分位数宽度+总量头,不拆label(总量误差由距离盲的日因子主导); bandit批评→加bandit基线实验(现在赢=批评坐实,预算化后输=RL选型兑现)

## 架构定位:全部在 env 侧三个点,agent 零改动(除输入维度)

用 env/agent 边界透镜:env 攥着 _build_state(可观测性)+compute_rl_reward(目标)+step(账本),预估器恰好三处各插一刀。不动 argmax 打分、不动 agent 内部、预估器不参与梯度(env 的一部分不是策略)——保住用户/春蓉分工:她换一个 reward 函数+加三列 state,用户做成即插即拔的 env 侧插件。

**插点① reward 动态阈值(价值主体)**: compute_rl_reward(3299) 的固定 TR_TH=0 换成 1367 日内化:
tr_th(t) = 日目标TR − (真实累计TR(t) − simu累计TR(t)) × (已用GMV(t)/剩余GMV预估(t))
预估器供分母=修正杠杆;尾盘分母缩小杠杆变陡→预估误差尾盘被放大(精度要求的来源)。把逐窗口独立优化变成日内预算控制。

**插点② state 加 3 维**(_build_state 3460 附近,④⑤组之间): 累计TR偏差/GMV已用比例/剩余GMV预估P50或P10P90带宽。网络 20→23 维,QNet(state_dim=23) 一个数字=agent 侧全部代价。业务含义: agent 从看不见预算变成知道剩多少弹药/时间/预估可信度。

**插点③ step() 记两本累计账**(3497): 真实累计(agent的λ走出)与simu累计(固定λ基线)。后者不用新算——fixed_baseline_cache(3379)已存每窗固定λ指标,累加即可。

## 预估规格清单(**2026-09-17经独立评审修订:label改标量日总版**,详见[[odt_rl_forecast_label_claude_review]]含原设计者回应)

- **主label: 归一化日总GMV标量**(城内,÷历史weekday中位日总),三头P10/P50/P90。依据:恒等式rem(t₀)=日总−已用(已用为观测值),唯一未知量是日总;当前下游只吃标量
- **接口: provider返回日总分位数,不传rem**——尾盘rem=两接近数相减放大相对误差;used_ratio=已用/日总由消费端算,无消解
- 占比曲线降级为**可选辅助头**(窗口感知th升级路径+诊断+多任务正则),不进契约
- 样本: (城市,日,t₀)条件化,同日96样本共享label(诚实结构);总量监督的独立单元=城市日级(~3.5万/459天)
- 对象仍是基线GMV(all轨,OTD域,city×product,bj口径),不预估TR/策略后GMV/λ响应
- 模型: 逐城V0模板+全国多城联合(城市静态特征条件化)+V2预训练兜异常日;**V0与新label同构,不受修订影响**
- 评估: 滚动起点回测按t₀分桶;test双向held-out(未见城×未见日期)只看一次
- 账C口径混搭(simu TR账vs真实表GMV进度)=oracle双跑显式观察项;日目标TR的base-relative/绝对口径=插点①落地前第一决策

## 离线退化预期

离线"真实累计"也来自同套预测矩阵→偏差项恒0→公式退化纯前馈(日目标按预估曲线分摊)。分阶段: 离线验证前馈收益上限;反馈修正收益须注入人工偏差(真实累计乘±10~30%执行误差)测鲁棒性;真值线上小流量。

## 动工顺序

1. V0模板+回测管线(裁判先行,兼公式分母v0供货商)
2. env侧加flag开关的dynamic_tr_th(先用V0供分母,纯前馈跑通插点①③)
3. 注入偏差测反馈鲁棒性
4. V1(iTransformer多城)替换分母供货商,同协议对比
5. 与春蓉对接口契约: forecast_day_total_gmv(city, date, t₀)→{P50,P10,P90 日总标量},纯函数签名(2026-09-17评审后统一为标量日总版)

待办: 用户同意后,按老流程(展示草稿→review→落盘)写 算法理解/时序预估接入设计.txt 作为与春蓉对齐的设计稿底本。

相关: [[odt_rl_gmv_forecast_discussion]] [[odt_rl_project]]
