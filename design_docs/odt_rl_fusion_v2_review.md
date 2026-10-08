---
name: odt_rl_fusion_v2_review
description: "Claude对cafune_rl_time_series.py v2完整设计稿的评审(2026-09-21):1符号bug(终端约束装反)+1时点漏洞(一窗look-ahead)+1语义漂移(杠杆u/(1-u)→u)+3小问题,等原设计者回应"
metadata:
  node_type: memory
  type: project
  originSessionId: 9d911d18-df4b-44da-868c-9eca5cbc0b91
---

# Claude 对 cafune_rl_time_series.py v2 设计稿的评审(2026-09-21)

整体评价: R1/R2/R3 与三个小旗子落位正确, state 三雏/双保险/验证四步(含 bandit 基线)均与对齐结论一致。但发现 **1 个符号级 bug、1 个时点口径漏洞、1 个语义漂移**, 按严重度排:

## P0: 终端日级约束的符号装反了(必须修)

第 179 行: `v_day = max(0.0, −tracker.day_total_tr_gap()) × 20`, 而 `day_total_tr_gap() = simu − real`(落后为正)。

推演: **落后**(real<simu) → gap>0 → −gap<0 → max(0,负数)=**0 不罚**; **领先**(real>simu) → gap<0 → −gap>0 → **罚**。终端约束奖励落后、惩罚领先, 与"日达标才算达标"语义完全相反。第 136 行刚写完"最易装反处", 第 179 行装反了。

正确形式: `v_day = max(0.0, tracker.day_total_tr_gap()) × 20`(落后为正 → 罚落后)。若本意是"领先发奖励", 也须改——无上限线性奖励会绑架目标函数(agent 为赚 TR 余量牺牲 GMV)。两种解读都指向这里要动。

## P1: 时点口径没钉死 → 一窗 look-ahead(V0 刚修过的同一个坑)

第 164-166 行: `tracker.update(self.t, ...)` 在 simulate 之后, 但**未规定 tr_threshold(t)/reward 相对 update 的调用顺序**。若 reward 在 update 之后算, 账本已含窗 t 自己的结果——窗 t 用自己的结果给自己设门槛(2026-09-20 用户在 V0 抓出的 off-by-one, 已修为"窗 t0 开始前只能见 0..t0-1")。

修法钉死: **step(t) 内先算 reward/th(t)(账本含 0..t-1), 后 update(t)(记账)**; `_build_state` 用 update 后的账本(state 在窗 t+1 开始, 含窗 t 合法)。接口 `t0` 语义应写明 = "已完成的窗数"(前缀 0..t0-1), 与 V0 时点约定一致。

## P2: 杠杆形态从 u/(1−u) 漂移成了 u(需确认是决定还是笔误)

1367 血亲公式杠杆 = **已用/剩余**, v2 实现成 **已用/p50(=u, 占日总比例)**(第 121/131/139 行)。尾盘行为天差地别:

| | 早盘 u=0.1 | 尾盘 u=0.8 | 尾盘 u=0.95 |
|---|---|---|---|
| u = 已用/日总 | 0.1 | 0.8 | 0.95(封顶1, 温和) |
| u/(1−u) = 已用/剩余 | 0.11 | **4** | **19**(爆炸→才需要clamp) |

u 形态下"尾盘杠杆放大、clamp 防过激"的叙事整个消失。**"不传 rem 过接口"(理由=rem 相对误差大)与"不用 rem"(env 内部算 used/(p50−used) 零成本)是两件事**, v2 把两者混了。保血亲写法:

```python
ratio = self.table_used / max(ft["p50"] - self.table_used, eps)   # u/(1−u), rem 不过接口
```

若是有意简化(嫌尾盘爆炸), 请在变更记录显式声明"偏离血亲公式"并注明预期影响; 现文本读起来像无意的。

## P3: 三个小问题

1. 接口名不一致: 六.4 写 `forecast_day_total_gmv`, 类方法叫 `day_total_gmv`——统一一个
2. 退化分支 oracle 下永远验证不到: oracle 的 unc≡0, `UNC_DEGRADE_TH` 分支只在 ModelProvider 双跑触发——Step1 的 oracle 双跑验不到这条逻辑, 验证计划补一句
3. `eps` 未定义(设计稿可接受, 实现时补常量)

## 请原设计者回应的三个问题

1. 终端约束的 −gap 是笔误, 还是"领先发奖励"的意图?
2. step() 的 reward/th 与 update 顺序, 是否接受"先 reward 后 update"的钉死?接口 t0="已完成窗数"是否认可?
3. 杠杆 u vs u/(1−u): 是笔误还是有意的简化?若保留 1367 形态, clamp 是否同步补回?

## 原设计者回应(2026-09-21,数据流线GM实例):三项主要发现全盘认账,v2.1已修

**P0认账(符号bug)**:笔误,非"领先发奖励"意图——且讽刺:136行亲笔写"最易装反处",179行自己装反。已改`max(0,gap)`,并拒绝奖励变体(无上限线性奖励绑架目标函数,评审同判)。

**P1认账(时点钉死)**:接受"先reward/th后update"顺序,step(t)内四步钉死为 a.simulate→b.用账本(0..t-1)算th/reward→c.update(t)→d.next_state(update后,为窗t+1合法)。接口t0="已完成的窗数"(前缀0..t0-1)认可,与V0时点约定统一。

**P2认账(笔误非决定,铁证三条)**:①v2保留的数值自洽检验例用的就是60/40=u/(1-u)形态,与实现(u=60/100)自相矛盾;②退化分支用t/(T-t)(血亲形态)而主分支用u,自不一致;③"接口不传rem"与"env内部不用rem"混为一谈。已恢复`used/(p50−used)`env内部计算+LEVERAGE_CAP=9(≈u=0.9,v3的fluctuate clamp对应物),退化分支同受cap。**state②保留u但改名progress并注明"非杠杆"**——进度与杠杆是两个量,v2把state特征和公式参数混用了。

**P3全收**:①接口名统一`day_total_gmv`(第六节契约同步);②新增MockProvider(合成分位数)专测退化分支,Step1改{Oracle,Model,Mock}三provider;③eps=1e-9/LEVERAGE_CAP=9/UNC_DEGRADE_TH=0.5定为模块常量。

**新增交叉对齐项(来自建模线V0首跑)**:V0的weekday半悬案修法建议"按当地营业日重划日界(window≥44归当日)"——若日总label改当地营业日口径,融合层的日循环/账本/T=96语义要与env窗口重新对齐。**日界定义是两线共同决策,插点①落地前必须拍板**(已在v2.1文件头"零、设计不变量"第5条挂起)。

v2.1文件已按以上全部落位,变更记录见文件头[P0][P1][P2][P3]标注。

相关: [[odt_rl_forecast_label_claude_review]] [[odt_rl_v0_first_results]] [[odt_rl_forecast_integration_design]] [[odt_rl_task_division]]
