# wjz 恢复
# ============================================================================
# 时序预估 × 强化学习融合层 —— v3 可执行版 (2026-09-21)
# ----------------------------------------------------------------------------
# 版本史: v1伪代码初稿(09-15) → v2评审修订·label改日总标量(09-17) →
#         v2.1二轮流评审·符号/时点/杠杆修正(09-21) → v3 全量可执行(09-21)
#         → v3.1 Claude交叉修复(09-21): V0Provider兜底改先验(不碰真值)/
#         state_dim去reset副作用/空窗补记账C/demo参数配平/中位字典补product_id
# 设计文档与两轮评审记录: memory/demo_rl_forecast_label_claude_review.md,
#                        memory/demo_rl_fusion_v2_review.md
#
# 【本文件的可跑边界】
#   本地可跑(无需df_final): ForecastProvider家族(Oracle/V0/Mock) + BudgetTracker
#     + SyntheticBudgetEnv(表格驱动的合成弹性日循环) —— 即验证路径的
#     Step1(th曲线形状/退化分支/口径观察) 与 Step2(偏差注入反馈) 的机制部分
#   需春蓉环境(有df_final时): CafuneBudgetedEnv 挂到 PassengerLambda30MinEnv
#     上(依赖注入,不patch源码), 进入Step3/4(DQN与六格对比)
#
# 【v2.1评审落位】[P0]终端符号=罚落后 [P1]step内先reward/th(账本0..t-1)后update,
#   t0=已完成的窗数 [P2]杠杆=u/(1-u)血亲形态+LEVERAGE_CAP [P3]命名统一/eps常量化
# ============================================================================
import os
import sys
import math
import numpy as np
import pandas as pd

# ---- 数据入口: 复用建模线的清洗管线 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "data_preprocess"))
from clean_data import load_clean  # noqa: E402

# ---- 常量(v2.1定稿) ----
eps = 1e-9
T = 96                        # 15min窗数/天
LEVERAGE_CAP = 9.0            # 杠杆上限≈u=0.9(v3的fluctuate clamp对应物)
UNC_DEGRADE_TH = 0.5          # unc超过→杠杆退化时间进度
TRIP_TH = -0.01               # 与cafune_rl.py一致的约束阈值
TR_TH = 0.0
CR_TH = 0.0
REWARD_WEIGHTS = {"gmv": 1.0, "trip": 5.0, "tr": 5.0, "cr": 5.0, "smooth": 0.1}
TERMINAL_W = 20.0             # 终端日级约束权重(v2.1: 罚落后[P0])
REWARD_CLIP = 5.0
FUSION_OUT_DIR = os.path.join(_HERE, "analysis", "output")


def safe_div(a, b):
    return a / max(abs(b), eps)


# ============================================================================
# 一、ForecastProvider 家族(接口: t0 = 已完成的窗数, 前缀=窗0..t0-1)
# ============================================================================
class ForecastProvider:
    """统一接口: 返回当日总GMV分位数(表格口径all轨)。
    消费端派生: progress=已用/p50; leverage=已用/(p50−已用); unc=(p90−p10)/p50"""

    def day_total_gmv(self, city_id, product_id, bj_date, t0):
        raise NotImplementedError


class OracleProvider(ForecastProvider):
    """读真值日总(偷看未来=天花板)。unc≡0——退化分支验不到它,用Mock。"""

    def __init__(self, df_clean):
        dt = df_clean.groupby(["city_id", "product_id", "bj_date"], as_index=False)[
            "all_gmv"].sum().rename(columns={"all_gmv": "day_total"})
        self._dt = {(r.city_id, r.product_id, pd.Timestamp(r.bj_date)): r.day_total
                    for r in dt.itertuples()}

    def day_total_gmv(self, city_id, product_id, bj_date, t0):
        v = self._dt.get((city_id, product_id, pd.Timestamp(bj_date)))
        if v is None:
            raise KeyError("oracle无该(城,产品,日): %s %s %s" % (city_id, product_id, bj_date))
        return {"p10": v, "p50": v, "p90": v}


class V0Provider(ForecastProvider):
    """V0同构模型: day_total_hat = 历史同weekday中位日总 × 已用(t0)/历史同weekday同t0中位已用。
    分位带宽 = 该城历史残差比(actual/pred)按t0分桶的经验q10/q90因子(demo级:全样本内计算,
    严格回测的时序切分归建模线, 见memory/demo_rl_v0_first_results)。"""

    N_BUCKET = 8  # 96窗分8桶, 每桶12窗

    def __init__(self, df_clean):
        df = df_clean.copy()
        df["wk"] = df["bj_date"].dt.weekday
        # 日总与weekday中位
        day_tot = df.groupby(["city_id", "product_id", "bj_date", "wk"], as_index=False)[
            "all_gmv"].sum().rename(columns={"all_gmv": "day_total"})
        self._day_tot = {(r.city_id, r.product_id, pd.Timestamp(r.bj_date)): r.day_total
                         for r in day_tot.itertuples()}
        self._med_dt = day_tot.groupby(["city_id", "product_id", "wk"])["day_total"].median().to_dict()
        # 前缀序列(t0=已完成窗数 → 前缀和) 与 城×wk×t0 中位前缀
        df = df.sort_values(["city_id", "product_id", "bj_date", "window_idx"])
        df["prefix"] = df.groupby(["city_id", "product_id", "bj_date"])["all_gmv"].cumsum()
        # prefix[w] = 含窗w的累计; 前缀(0..t0-1) = shift后cumsum, t0=0时为0
        df["prefix_t0"] = df.groupby(["city_id", "product_id", "bj_date"])["prefix"].shift(1).fillna(0.0)
        self._prefix = {(r.city_id, r.product_id, pd.Timestamp(r.bj_date), int(r.window_idx)):
                        r.prefix_t0 for r in df.itertuples()}
        self._med_prefix = df.groupby(["city_id", "product_id", "wk", "window_idx"])["prefix_t0"].median().to_dict()
        # 残差比因子(城×桶): actual/pred 的 q10/q90
        probes = list(range(0, T, T // self.N_BUCKET))
        recs = []
        for r in day_tot.itertuples():
            key_p = (r.city_id, r.product_id)
            bdate = pd.Timestamp(r.bj_date)
            for t0 in probes:
                used = self._prefix.get((r.city_id, r.product_id, bdate, t0), 0.0)
                pred = self._predict(key_p[0], key_p[1], r.wk, bdate, t0)
                if pred is not None and pred > eps:
                    recs.append((r.city_id, t0 // (T // self.N_BUCKET),
                                 r.day_total / pred))
        rdf = pd.DataFrame(recs, columns=["city_id", "bucket", "ratio"])
        self._q = rdf.groupby(["city_id", "bucket"])["ratio"].quantile([0.1, 0.9]).unstack()
        self._probes = probes

    def _predict(self, city_id, product_id, wk, bj_date, t0):
        base = self._med_dt.get((city_id, product_id, wk))
        if base is None or base <= eps:
            return None
        if t0 == 0:
            return base
        med_used = self._med_prefix.get((city_id, product_id, wk, t0), 0.0)
        if med_used <= eps:
            return None
        used = self._prefix.get((city_id, product_id, bj_date, t0), 0.0)
        return base * used / med_used

    def day_total_gmv(self, city_id, product_id, bj_date, t0):
        ts = pd.Timestamp(bj_date)
        wk = ts.weekday()
        pred = self._predict(city_id, product_id, wk, ts, t0)
        if pred is None:
            # 冷启动兜底: 用先验(历史同weekday中位日总, V0a无校准版), 绝不碰真值
            # (v3.1: 原实现退回oracle真值会污染 oracle−model 差距度量)
            base = self._med_dt.get((city_id, product_id, wk))
            if base is None or base <= eps:
                raise KeyError("V0Provider无历史先验: %s %s weekday=%d"
                               % (city_id, product_id, wk))
            return {"p10": base * 0.85, "p50": base, "p90": base * 1.15}
        bucket = min(t0 // (T // self.N_BUCKET), self.N_BUCKET - 1)
        try:
            q10, q90 = self._q.loc[(city_id, bucket)]
        except KeyError:
            q10, q90 = 0.85, 1.15
        return {"p10": pred * q10, "p50": pred, "p90": pred * q90}


class MockProvider(ForecastProvider):
    """合成带宽: p10/p90 = p50×(1∓unc)——专测退化分支与unc传导。"""

    def __init__(self, base_provider, unc_value):
        self.base = base_provider
        self.unc = unc_value

    def day_total_gmv(self, city_id, product_id, bj_date, t0):
        ft = self.base.day_total_gmv(city_id, product_id, bj_date, t0)
        p50 = ft["p50"]
        return {"p10": p50 * (1 - self.unc), "p50": p50, "p90": p50 * (1 + self.unc)}


# ============================================================================
# 二、BudgetTracker(v2.1: 三本账 + progress/leverage分离[P2] + 时点纪律[P1])
# ============================================================================
class BudgetTracker:
    """每episode(天)一个。时点不变量: 所有读取方法只在update(t)之前调用合法
    (彼时账本恰好含窗0..t-1)。"""

    def __init__(self, forecast_provider, city_id, product_id, bj_date):
        self.real_gp = 0.0; self.real_gmv = 0.0      # 账A: agent的λ走出(env口径)
        self.simu_gp = 0.0; self.simu_gmv = 0.0       # 账B: fixed-λ baseline(env口径)
        self.table_used = 0.0                          # 账C: 表格口径已用
        self.forecast = forecast_provider
        self.keys = (city_id, product_id, bj_date)

    def update(self, metrics, fixed_metrics, table_row):
        """只在step(t)的reward/th计算之后调用。"""
        self.real_gp += metrics["gp_opt"]; self.real_gmv += metrics["gmv_opt"]
        self.simu_gp += fixed_metrics["gp_base"]; self.simu_gmv += fixed_metrics["gmv_base"]
        self.table_used += table_row["all_gmv"]

    def update_table_used(self, table_row):
        """空窗跳过时只记账C(表格口径进度拨盘); 账A/B无metrics不动。"""
        self.table_used += table_row["all_gmv"]

    def progress(self, t):
        ft = self.forecast.day_total_gmv(*self.keys, t0=t)
        return self.table_used / max(ft["p50"], eps)

    def leverage(self, t):
        """血亲杠杆 = 已用/(日总p50−已用) = u/(1-u), clamp防尾盘爆炸。
        自洽检验: 落后2%,u=0.6 → gap=2%×1.5=3% → 40%×3%=1.2pp=60%×2% ✓"""
        ft = self.forecast.day_total_gmv(*self.keys, t0=t)
        rem = max(ft["p50"] - self.table_used, eps)
        return min(self.table_used / rem, LEVERAGE_CAP)

    def state_features(self, t):
        """+3维: tr_gap / progress(u,非杠杆!) / unc(日总空间)"""
        ft = self.forecast.day_total_gmv(*self.keys, t0=t)
        tr_gap = safe_div(self.simu_gp, self.simu_gmv) - safe_div(self.real_gp, self.real_gmv)
        return [float(np.clip(tr_gap, -1, 1)),
                self.progress(t),
                (ft["p90"] - ft["p10"]) / max(ft["p50"], eps)]

    def tr_threshold(self, base_th, t):
        """≥约束: 真实累计TR落后(负gap项) → 门槛抬高 → 收紧。"""
        ft = self.forecast.day_total_gmv(*self.keys, t0=t)
        unc = (ft["p90"] - ft["p10"]) / max(ft["p50"], eps)
        if unc > UNC_DEGRADE_TH:  # 退化: 时间进度形态, 同受cap
            ratio = min(t / max(T - t, 1), LEVERAGE_CAP)
        else:
            ratio = self.leverage(t)
        self.last_ratio = ratio  # 日志用: 实际生效的杠杆(退化时≠血亲形态)
        tr_gap = safe_div(self.simu_gp, self.simu_gmv) - safe_div(self.real_gp, self.real_gmv)
        return base_th + tr_gap * ratio

    def day_total_tr_gap(self):
        """终端约束用(落后为正, 罚落后[P0])。"""
        return safe_div(self.simu_gp, self.simu_gmv) - safe_div(self.real_gp, self.real_gmv)


def compute_reward_budgeted(metrics, th_t, old_dev, new_dev):
    """融合层reward(合成环境版): gmv提升 − 动态th的v_tr − 平滑罚。
    与cafune_rl.compute_rl_reward同构, v_trip/v_cr在合成环境无定义故省。"""
    gmv_lift = safe_div(metrics["gmv_opt"] - metrics["gmv_base"], metrics["gmv_base"])
    v_tr = max(0.0, th_t - metrics["inc_tr"])
    smooth = ((new_dev - old_dev) / 0.05) ** 2
    raw = (REWARD_WEIGHTS["gmv"] * gmv_lift
           - REWARD_WEIGHTS["tr"] * v_tr
           - REWARD_WEIGHTS["smooth"] * smooth)
    return float(np.clip(raw, -REWARD_CLIP, REWARD_CLIP)), {
        "gmv_lift": gmv_lift, "v_tr": v_tr, "th_t": th_t, "smooth": smooth}


# ============================================================================
# 三、SyntheticBudgetEnv —— 表格驱动的日循环(本地可跑: Step1/Step2验证)
# ============================================================================
class SyntheticBudgetEnv:
    """用聚合表的all_gmv作每窗基准量, 合成TR与弹性(参数明示, 仅机制验证用):
      tr_base(t) = TR0×(1+AMP×sin) ; λ_dev: 松→gmv升/TR降(α/β弹性)
    [P1]顺序: a.政策读th(账本0..t-1) → b.simulate → c.th/reward → d.update → e.日志
    """

    def __init__(self, day_rows, provider, city_id, product_id, bj_date,
                 tr0=0.20, amp=0.30, alpha_gmv=0.5, beta_tr=0.5, bias=0.0,
                 lambda_cap=0.10):
        self.rows = day_rows.sort_values("window_idx").reset_index(drop=True)
        assert len(self.rows) == T, "当日窗数=%d 应=%d" % (len(self.rows), T)
        self.tracker = BudgetTracker(provider, city_id, product_id, bj_date)
        self.tr0, self.amp = tr0, amp
        self.alpha, self.beta, self.bias = alpha_gmv, beta_tr, bias
        self.lam_cap = lambda_cap

    def _tr_base(self, t):
        return self.tr0 * (1.0 + self.amp * math.sin(2 * math.pi * t / T))

    def run(self, policy="rule", step=0.01, verbose_every=8):
        lam = 0.0
        logs = []
        total_r = 0.0
        for t in range(T):
            # a. 政策(读当前th, 账本=0..t-1, 合法)
            th_for_policy = self.tracker.tr_threshold(TR_TH, t)
            if policy == "fixed":
                pass
            elif policy == "rule":  # 落后→收紧λ(护TR), 达标→缓慢放松(换GMV)
                inc_tr_prev = logs[-1]["inc_tr"] if logs else 0.0
                if inc_tr_prev < th_for_policy:
                    lam = max(lam - step, -self.lam_cap)
                else:
                    lam = min(lam + step * 0.5, self.lam_cap)
            old_lam = lam if t == 0 else logs[-1]["lam"]
            # b. simulate(合成弹性)
            gmv_base = float(self.rows.iloc[t]["all_gmv"])
            tr_b = self._tr_base(t)
            gp_base = gmv_base * tr_b
            gmv_opt = gmv_base * (1 + self.alpha * lam)
            tr_opt = tr_b * (1 - self.beta * lam)
            gp_opt = gmv_opt * tr_opt * (1 + self.bias)  # bias=Step2注入
            metrics = {"gmv_base": gmv_base, "gp_base": gp_base,
                       "gmv_opt": gmv_opt, "gp_opt": gp_opt,
                       "inc_tr": safe_div(gp_opt - gp_base, gmv_base)}
            fixed_metrics = {"gmv_base": gmv_base, "gp_base": gp_base}
            # c. th/reward(账本仍=0..t-1)
            th_t = th_for_policy
            r, rinfo = compute_reward_budgeted(metrics, th_t, old_lam, lam)
            total_r += r
            feat = self.tracker.state_features(t)
            # d. update(t) 记账
            self.tracker.update(metrics, fixed_metrics, self.rows.iloc[t])
            logs.append({"t": t, "lam": lam, "th_t": th_t, **rinfo,
                         "inc_tr": metrics["inc_tr"], "tr_gap": feat[0],
                         "progress": feat[1], "leverage": self.tracker.last_ratio,
                         "unc": feat[2], "gmv_base": gmv_base, "gmv_opt": gmv_opt})
            if verbose_every and t % verbose_every == 0:
                print("  t=%2d lam=%+.2f th=%+.4f inc_tr=%+.4f gap=%+.4f "
                      "u=%.2f lev=%.2f unc=%.3f r=%+.4f"
                      % (t, lam, th_t, metrics["inc_tr"], feat[0],
                         feat[1], logs[-1]["leverage"], feat[2], r))
        # 终端日级约束(罚落后[P0])
        day_gap = self.tracker.day_total_tr_gap()
        v_day = max(0.0, day_gap) * TERMINAL_W
        print("  [terminal] day_tr_gap=%+.4f v_day=%.4f | total_r=%.4f"
              % (day_gap, v_day, total_r - v_day))
        return pd.DataFrame(logs)


# ============================================================================
# 四、CafuneBudgetedEnv —— 挂接春蓉env(需df_final环境, 依赖注入不patch源码)
# ============================================================================
class CafuneBudgetedEnv:
    """把融合层挂到 PassengerLambda30MinEnv 上。用法(在能构造该env的环境):
        import cafune_rl as cr
        env = cr.PassengerLambda30MinEnv(df_day, keep_solution=False)
        budgeted = CafuneBudgetedEnv(env, cr.simulate_30min_window,
                                     cr.apply_lambda_action, provider, day_rows)
        state = budgeted.reset()   # 23维
        ... budgeted.step(action_idx)  # 动态th reward + 终端项, 内部P1顺序
    reward与cafune_rl.compute_rl_reward同构(v_tr改动态th, done步加v_day)。
    """

    def __init__(self, env, sim_fn, apply_fn, provider, day_rows,
                 terminal_w=TERMINAL_W):
        self.env = env
        self.sim_fn, self.apply_fn = sim_fn, apply_fn
        self.tracker = BudgetTracker(provider, env.df_day["city_id"].iloc[0],
                                     env.df_day["product_id"].iloc[0],
                                     env.df_day["rl_date"].iloc[0])
        self.day_rows = day_rows.sort_values("window_idx").set_index("window_idx")
        self.terminal_w = terminal_w
        self.last_metrics = None

    def reset(self):
        s = self.env.reset()
        self.tracker = BudgetTracker(self.tracker.forecast, *self.tracker.keys)
        return np.concatenate([s, np.array(self.tracker.state_features(0), dtype=np.float32)])

    def state_dim(self):
        # v3.1: 原实现 len(self.env.reset()) 每次查询都重置底层env(DQN训练中会被打断)
        return self.env.state_dim + 3

    def _reward(self, metrics, fixed_m, old_lam, new_lam, th_t, done):
        gmv_lift = safe_div(metrics["gmv_opt"] - fixed_m["gmv_opt"], fixed_m["gmv_opt"])
        v_trip = max(0.0, TRIP_TH - metrics["inc_trip"])
        v_tr = max(0.0, th_t - metrics["inc_tr"])
        v_cr = max(0.0, CR_TH - metrics["inc_cr"])
        smooth = sum(((new_lam[k] - old_lam[k]) / max(abs(old_lam[k]), eps)) ** 2
                     for k in new_lam)
        raw = (REWARD_WEIGHTS["gmv"] * gmv_lift - REWARD_WEIGHTS["trip"] * v_trip
               - REWARD_WEIGHTS["tr"] * v_tr - REWARD_WEIGHTS["cr"] * v_cr
               - REWARD_WEIGHTS["smooth"] * smooth)
        if done:  # [P0] 终端: 罚落后
            raw -= self.terminal_w * max(0.0, self.tracker.day_total_tr_gap())
        return float(np.clip(raw, -REWARD_CLIP, REWARD_CLIP))

    def step(self, action_idx):
        e = self.env
        if e.t >= len(e.window_list):
            return self.reset(), 0.0, True, {}
        window_id = e.window_list[e.t]
        old_lam = e.cur_lambda.copy()
        new_lam = self.apply_fn(cur_lambda=old_lam, init_lambda=e.init_lambda,
                                action_tuple=e.action_list[action_idx])
        df_win = e.df_day[e.df_day["window_id"] == window_id].copy()
        df_sol, metrics = self.sim_fn(df_win=df_win, cur_pax_lambda=new_lam)
        if metrics is None:            # 空窗跳过(cafune v2语义)
            # v3.1: 跳过也补记账C, 否则表格口径进度拨盘少记该窗
            self.tracker.update_table_used(self.day_rows.loc[window_idx_of(window_id)])
            e.cur_lambda = new_lam; e.t += 1
            done = e.t >= len(e.window_list)
            return self.reset() if done else self._state(), float("nan"), done, {"skipped": True}
        # [P1] 先th/reward(账本=0..t-1) 后update
        th_t = self.tracker.tr_threshold(TR_TH, e.t)
        fixed_m = e.fixed_baseline_cache[window_id]
        r = self._reward(metrics, fixed_m, old_lam, new_lam, th_t,
                         done=(e.t + 1 >= len(e.window_list)))
        self.tracker.update(metrics, fixed_m, self.day_rows.loc[window_idx_of(window_id)])
        e.cur_lambda = new_lam; e.prev_metrics = metrics
        e.t += 1
        done = e.t >= len(e.window_list)
        return self._state(), r, done, {"th_t": th_t, "metrics": metrics}

    # ---- 工具 ----
    def _state(self):
        s = self.env._build_state()
        return np.concatenate([s, np.array(self.tracker.state_features(self.env.t),
                                           dtype=np.float32)])


def window_idx_of(window_id):
    """cafune的window_id='YYYY-MM-DD HH:MM' → 窗序号(与聚合表对齐的桥)。"""
    ts = pd.Timestamp(window_id)
    return ts.hour * 4 + ts.minute // 15


# ============================================================================
# 五、main: Step1/Step2 机制验证demo(本地全量可跑)
# ============================================================================
if __name__ == "__main__":
    # ============ 参数区 ============
    CITY = None                  # None=自动选GMV最大的城
    BJ_DATE = None               # None=自动选数据中段的一个周三
    RUNS = [
        ("A_oracle_fixed_nobias",  "oracle",  "fixed", 0.0),   # 前馈基线: th应平
        ("B_oracle_fixed_bias-10", "oracle",  "fixed", -0.10),  # Step2前馈: 无补偿→th应单调升
        # v3.1参数配平: 补偿上限=β×lam_cap=5%, bias=-0.10时规则物理上补不满
        # (终gap≈0.055×tr_b>0.01假警报); 改-0.05使|gap|可收敛到<0.01
        ("C_oracle_rule_bias-05",  "oracle",  "rule",  -0.05),  # Step2闭环: gap应收敛
        ("D_mock_degrade",         "mock0.8", "rule",  -0.10),  # 退化分支: 杠杆=时间形态
    ]
    # ==============================

    print("== 加载清洗数据(共用入口, 带缓存) ==")
    df = load_clean()
    df["rl_date"] = df["bj_date"].dt.strftime("%Y-%m-%d")
    if CITY is None:
        CITY = df.groupby("city_id")["all_gmv"].sum().idxmax()
    sub = df[df.city_id == CITY]
    PRODUCT = sub["product_id"].iloc[0]
    if BJ_DATE is None:
        wed = sub[sub.bj_date.dt.weekday == 2]["bj_date"].drop_duplicates()
        BJ_DATE = wed.iloc[len(wed) // 2].strftime("%Y-%m-%d")
    day_rows = sub[sub.rl_date == BJ_DATE]
    print("demo: city=%s product=%s date=%s rows=%d day_total=%.1f"
          % (CITY, PRODUCT, BJ_DATE, len(day_rows), day_rows["all_gmv"].sum()))

    print("\n== 构建providers ==")
    oracle = OracleProvider(df)
    v0 = V0Provider(df)
    print("V0 sanity(该日): " + str({k: round(v, 1) for k, v in
          v0.day_total_gmv(CITY, PRODUCT, BJ_DATE, t0=48).items()}))
    print("  [注意] V0 sanity 为全样本内统计(含预测日), 数值偏乐观, 仅作接口连通性检查;"
          "严格回测时序切分归建模线")

    os.makedirs(FUSION_OUT_DIR, exist_ok=True)
    for name, prov_kind, policy, bias in RUNS:
        prov = {"oracle": oracle, "mock0.8": MockProvider(oracle, 0.8)}[prov_kind]
        print("\n===== RUN %s (provider=%s policy=%s bias=%.2f) =====" % (name, prov_kind, policy, bias))
        env = SyntheticBudgetEnv(day_rows, prov, CITY, PRODUCT, BJ_DATE, bias=bias)
        log = env.run(policy=policy)
        out = os.path.join(FUSION_OUT_DIR, "fusion_demo_%s_%s_%s.csv" % (name, CITY, BJ_DATE))
        log.to_csv(out, index=False)
        print("  log saved -> %s" % out)
        # 机制判读(按run类型给判据)
        th_head, th_tail = log["th_t"].iloc[0], log["th_t"].iloc[-16:].mean()
        if policy == "fixed" and bias < 0:
            ok = th_tail > th_head and log["th_t"].iloc[-1] > 0
            print("  [check] 前馈: th %.4f→%.4f 应单调收紧且尾盘>0: %s"
                  % (th_head, th_tail, "YES" if ok else "NO!"))
        elif policy == "rule" and bias < 0 and "mock" not in prov_kind:
            flips = (log["tr_gap"].iloc[1:] * log["tr_gap"].iloc[:-1] < 0).sum()
            print("  [check] 闭环: gap符号翻转%d次(先落后后超前=补偿过头), "
                  "terminal_gap=%+.4f(应|·|小于无补偿的尾盘gap): %s"
                  % (flips, log["tr_gap"].iloc[-1],
                     "YES" if abs(log["tr_gap"].iloc[-1]) < 0.01 else "CHECK"))
        elif "mock" in prov_kind:
            t_probe = 48  # log每窗一行, iloc[t]即窗t
            print("  [check] 退化: unc=%.2f>%.1f, t=%d时eff_leverage=%.2f "
                  "(时间形态=%.2f): %s"
                  % (log["unc"].iloc[-1], UNC_DEGRADE_TH, t_probe,
                     log["leverage"].iloc[t_probe],
                     t_probe / (T - t_probe),
                     "YES" if abs(log["leverage"].iloc[t_probe]
                                  - t_probe / (T - t_probe)) < 0.3 else "NO!"))
