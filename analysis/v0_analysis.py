#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ********************************************************************#
##author: zhaowenjiao
##create time: 2026-09-17
##desc: GMV 时序预估支线 V0 分析(裁判先行)
##  1) 基础体检: 数据量级/product分布/null日期分布/weekday语义
##  2) 形状图: 日内占比曲线 → 兼做 window_idx 时区判别
##  3) V0 滚动回测: V0a 裸模板(历史同weekday中位日总) vs
##     V0b 模板×校准因子(实际前缀/历史同前缀中位), 按 t0 分桶
##     报 WMAPE/带宽穿越率, 严格只用历史(无上帝视角)
##  4) 口径温度计: explore_ratio / avg_dp 按月趋势
##输出: analysis/output/ 下 CSV + PNG
# ********************************************************************#
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============ 参数区 ============
DATA_PATH = "/Users/didi/workspace/data/sp_features/tarnet/BR#2025-05-01_2026-09-15_gmv_15min_day.txt"
OUT_DIR = "/Users/didi/workspace/python-project/odt_rl/analysis/output"
N_BUCKETS = 12                  # t0 分桶数, 每桶 96/N_BUCKETS=8 窗
MIN_HIST_SAME_WEEKDAY = 4       # 模板最少需要的历史同weekday天数(预热)
EXAMPLE_CITIES = ["55000233", "55000185", "55000207"]   # 形状图示例城
EXCLUDE_CITIES = ["55000195", "55000082", "55000412"]   # 小体量城(日总中位<13万), 相对波动大, 排出裁判口径
# ==============================

dtype_map = {
    "city_id": "string", "product_id": "string", "bj_date": "string",
    "country_code": "string", "year": "string", "month": "string", "day": "string",
}


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_data(path):
    df = pd.read_csv(path, sep="\t", compression="gzip", dtype=dtype_map)
    df["bj_date"] = pd.to_datetime(df["bj_date"])
    return df


def basic_check(df):
    section("1. 基础体检")
    print("rows: {}, cols: {}".format(len(df), df.shape[1]))
    print("date range: {} ~ {}".format(df.bj_date.min().date(), df.bj_date.max().date()))
    print("city count: {}, product values: {}".format(df.city_id.nunique(), df.product_id.unique().tolist()))

    # 每城 product 构成(确认'16'过滤是否丢城)
    prod_per_city = df.groupby("city_id")["product_id"].nunique()
    print("product_id nunique per city: max={}, cities with !=1: {}".format(
        prod_per_city.max(), (prod_per_city != 1).sum()))

    # 每城: 天数/行数/每窗null率(对照盘点SQL)
    g = df.groupby("city_id").agg(
        days=("bj_date", "nunique"), rows=("bj_date", "size"),
        null_ratio=("all_gmv", lambda s: s.isna().mean()))
    g["rows_per_day"] = g.rows / g.days
    print("rows_per_day min/max: {:.2f} / {:.2f}".format(g.rows_per_day.min(), g.rows_per_day.max()))
    print("null_ratio 分档: median={:.4f}, p90={:.4f}, max={:.4f} (city {})".format(
        g.null_ratio.median(), g.null_ratio.quantile(0.9), g.null_ratio.max(), g.null_ratio.idxmax()))
    g.to_csv(os.path.join(OUT_DIR, "per_city_overview.csv"))

    # null 日期分布(判别: 是否集中在某段时期)
    null_by_ym = df[df.all_gmv.isna()].groupby(["year", "month"]).size()
    print("\nnull 按月分布(空=该月无null):")
    print(null_by_ym.to_string())
    # null 按窗口分布
    null_by_win = df.groupby("window_idx")["all_gmv"].apply(lambda s: s.isna().mean())
    print("\nnull 按窗口分布: min_win={} max_win={}".format(
        null_by_win.idxmin(), null_by_win.idxmax()))
    print("null 按窗口(前10高):")
    print(null_by_win.nlargest(10).to_string())

    # weekday 语义: 与 bj_date 派生的 weekday 是否一致(透传关案判别)
    wdf = df.groupby("bj_date")["weekday"].first()
    exp = pd.Series(wdf.index.weekday, index=wdf.index)
    mismatch = (wdf != exp)
    print("\nweekday 与 bj_date 派生不一致的天数: {} / {} ({:.1%})  [=0则透传关案]".format(
        mismatch.sum(), len(wdf), mismatch.mean()))
    offset = (wdf - exp) % 7
    print("weekday 偏移分布(0=一致, 6=提前一天, 1=推后一天): {}".format(
        offset.value_counts().sort_index().to_dict()))
    mm_ym = mismatch.groupby([wdf.index.year, wdf.index.month]).mean()
    print("weekday 不一致率按月(%):")
    print((mm_ym * 100).round(1).to_string())


def build_day_table(df):
    section("2. 日总表构建")
    day_df = df.groupby(["city_id", "bj_date"], as_index=False).agg(
        all_gmv_daysum=("all_gmv", lambda s: s.fillna(0).sum()),
        explore_gmv_daysum=("explore_gmv", lambda s: s.fillna(0).sum()),
        null_win_cnt_daysum=("all_gmv", lambda s: s.isna().sum()),
        weekday=("weekday", "first"),
    )
    zero_days = (day_df.all_gmv_daysum <= 0).sum()
    print("日级样本: {} (城市×日), 其中 all_gmv_daysum<=0 的天数: {}".format(len(day_df), zero_days))
    print("日总分布: median={:,.0f}, mean={:,.0f}, p99={:,.0f}".format(
        day_df.all_gmv_daysum.median(), day_df.all_gmv_daysum.mean(), day_df.all_gmv_daysum.quantile(0.99)))
    day_df.to_csv(os.path.join(OUT_DIR, "day_total_table.csv"), index=False)
    return day_df


def shape_plots(df, day_df):
    section("3. 日内形状图(兼时区判别)")
    # 全局日内占比曲线
    share_all = df.groupby("window_idx")["all_gmv"].sum() / df["all_gmv"].sum()
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes[0, 0].plot(share_all.index, share_all.values)
    axes[0, 0].set_title("overall intraday share")
    axes[0, 0].set_xlabel("window_idx (bj?)"); axes[0, 0].set_ylabel("share")
    for ax, c in zip(axes.flat[1:], EXAMPLE_CITIES):
        sub = df[df.city_id == c]
        if len(sub) == 0:
            continue
        s = sub.groupby("window_idx")["all_gmv"].sum() / sub["all_gmv"].sum()
        ax.plot(s.index, s.values)
        ax.set_title("city {}".format(c))
        ax.set_xlabel("window_idx"); ax.set_ylabel("share")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "intraday_shape.png"), dpi=100)
    plt.close(fig)
    print("saved intraday_shape.png")
    print("判别口径: 若 window_idx 是 bj 时间, BR当地(UTC-3)早晚高峰应落在窗 16-24(当地17-19点)与 72-80(当地7-9点)")
    print("         若 window_idx 是当地口径, 高峰应落在窗 28-36(早)与 68-76(晚)。看图说话。")
    top3 = share_all.nlargest(3)
    print("全局占比曲线 top3 峰值窗: {}".format({k: round(v, 4) for k, v in top3.items()}))
    for c in EXAMPLE_CITIES:
        sub = df[df.city_id == c]
        if len(sub) == 0:
            continue
        s = sub.groupby("window_idx")["all_gmv"].sum() / sub["all_gmv"].sum()
        print("city {} top3 峰值窗: {}".format(c, {k: round(v, 4) for k, v in s.nlargest(3).items()}))

    # weekday 效应: 各城归一化日总(÷自身中位)按 weekday 均值
    norm = day_df.groupby("city_id")["all_gmv_daysum"].median()
    day_df["norm_total"] = day_df.all_gmv_daysum / day_df.city_id.map(norm)
    wd_eff = day_df[day_df.all_gmv_daysum > 0].groupby("weekday")["norm_total"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(wd_eff.index, wd_eff.values)
    ax.set_title("weekday effect (day_total / city median)")
    ax.set_xlabel("weekday"); ax.set_ylabel("ratio")
    fig.savefig(os.path.join(OUT_DIR, "weekday_effect.png"), dpi=100)
    plt.close(fig)
    print("saved weekday_effect.png; weekday 效应: {}".format(wd_eff.round(3).to_dict()))


def run_v0_backtest(df, day_df):
    section("4. V0 滚动回测 (V0a 裸模板 vs V0b 模板×校准因子)")

    # 前缀累计矩阵: (city, date) x 96 窗
    df2 = df.sort_values(["city_id", "bj_date", "window_idx"])
    df2["gmv_nz"] = df2["all_gmv"].fillna(0)
    df2["cum_gmv"] = df2.groupby(["city_id", "bj_date"])["gmv_nz"].cumsum()
    cum_piv = df2.pivot_table(index=["city_id", "bj_date"], columns="window_idx",
                              values="cum_gmv", aggfunc="first")
    print("cum 矩阵: {}".format(cum_piv.shape))

    bucket_unit_size = 96 // N_BUCKETS
    # 按桶累计: [sum|err_a|, sum|err_b|, sum_actual, n, cover_a, cover_b]
    acc = np.zeros((N_BUCKETS, 6))
    city_stats = []
    n_warmup_skip = 0

    cities = [c for c in day_df.city_id.unique() if c not in EXCLUDE_CITIES]
    print("参与回测城市: {} (排除小体量城: {})".format(len(cities), EXCLUDE_CITIES))
    for ci, city in enumerate(cities):
        day_df_city = day_df[day_df.city_id == city].sort_values("bj_date").reset_index(drop=True)
        day_gmv_array = day_df_city["all_gmv_daysum"].to_numpy(dtype=float)
        W = day_df_city["weekday"].to_numpy()
        cum_piv_city_values = cum_piv.loc[city].reindex(day_df_city["bj_date"].values).to_numpy()  # (n_days, 96)
        if np.isnan(cum_piv_city_values).any():
            cum_piv_city_values = np.nan_to_num(cum_piv_city_values)  # 缺窗兜底
        city_err_a = city_gmv_sum = 0.0
        for wd in range(7):
            idx = np.where(W == wd)[0]
            if len(idx) <= MIN_HIST_SAME_WEEKDAY:
                n_warmup_skip += 1
                continue
            for j in range(MIN_HIST_SAME_WEEKDAY, len(idx)):
                i = idx[j]
                hist = idx[:j]
                mask = day_gmv_array[hist] > 0
                day_gmv_hist = day_gmv_array[hist][mask]
                if len(day_gmv_hist) == 0:
                    continue
                cum_piv_city_hist = cum_piv_city_values[hist][mask]
                # 预测日 i 的前缀曲线: 96维数组, [t0]=截至窗t0的累计GMV
                cum_piv_city_today = cum_piv_city_values[i]
                day_gmv_today = day_gmv_array[i]
                if day_gmv_today <= 0:
                    continue
                v0a = np.median(day_gmv_hist)
                q10, q90 = np.quantile(day_gmv_hist, [0.1, 0.9])
                cum_piv_hist_median = np.median(cum_piv_city_hist, axis=0)
                # 时点约定: 预测t0发生在窗t0开始前(agent此时定λ, 窗t0结果不存在),
                # 只能用已完成的窗 0..t0-1 → 分子分母同步移一位;
                # t0=0(开盘前)无前缀 → calib=1, 退化为V0a
                prefix_before_t0 = np.concatenate([[0.0], cum_piv_city_today[:-1]])
                hist_before_t0 = np.concatenate([[0.0], cum_piv_hist_median[:-1]])
                with np.errstate(invalid="ignore", divide="ignore"):
                    calib = np.where((hist_before_t0 > 0) & (prefix_before_t0 > 0),
                                     prefix_before_t0 / hist_before_t0, 1.0)
                v0b = v0a * calib
                err_a = np.abs(v0a - day_gmv_today)
                err_b = np.abs(v0b - day_gmv_today)
                cover_a = 1.0 if q10 <= day_gmv_today <= q90 else 0.0
                cover_b = ((q10 * calib <= day_gmv_today) & (day_gmv_today <= q90 * calib)).astype(float)
                # 按桶累计
                for t0 in range(96):
                    b = min(t0 // bucket_unit_size, N_BUCKETS - 1)
                    acc[b, 0] += err_a
                    acc[b, 1] += err_b[t0]
                    acc[b, 2] += day_gmv_today
                    acc[b, 3] += 1
                    acc[b, 4] += cover_a
                    acc[b, 5] += cover_b[t0]
                city_err_a += err_a
                city_gmv_sum += day_gmv_today
        if city_gmv_sum > 0:
            # 城市级只报 V0a(尾盘 t0=95 的 V0b 校准因子≈day_gmv_today/v0a, 退化自预测, 无意义)
            city_stats.append([city, city_err_a / city_gmv_sum])
        if (ci + 1) % 20 == 0:
            print("  {} cities done...".format(ci + 1))

    # 分桶报表
    res = pd.DataFrame(acc, columns=["sum_err_a", "sum_err_b", "sum_act", "n", "cover_a", "cover_b"])
    res["bucket"] = np.arange(N_BUCKETS)
    res["t0_range"] = ["{}-{}".format(b * bucket_unit_size, min((b + 1) * bucket_unit_size - 1, 95)) for b in range(N_BUCKETS)]
    res["wmape_a"] = res.sum_err_a / res.sum_act
    res["wmape_b"] = res.sum_err_b / res.sum_act
    res["improve%"] = (res.wmape_a - res.wmape_b) / res.wmape_a * 100
    res["cover_a"] = res.cover_a / res.n
    res["cover_b"] = res.cover_b / res.n
    res = res[["bucket", "t0_range", "n", "wmape_a", "wmape_b", "improve%", "cover_a", "cover_b"]]
    print("\n分桶报表(V0b=带校准因子; improve%=V0b相对V0a的WMAPE降幅):")
    print(res.round(4).to_string(index=False))
    res.to_csv(os.path.join(OUT_DIR, "v0_backtest_by_bucket.csv"), index=False)

    cstat = pd.DataFrame(city_stats, columns=["city_id", "wmape_a"])
    cstat = cstat.sort_values("wmape_a")
    print("\n城市排名(按 V0a WMAPE, 前10/后10):")
    print(cstat.round(4).to_string(index=False, max_rows=20))
    cstat.to_csv(os.path.join(OUT_DIR, "v0_backtest_by_city.csv"), index=False)
    print("\noverall V0a WMAPE: {:.4f}, V0b 全桶平均 WMAPE: {:.4f}(含尾盘退化桶, 看分桶表为准), 预热跳过组数: {}".format(
        acc[:, 0].sum() / acc[:, 2].sum(), acc[:, 1].sum() / acc[:, 2].sum(), n_warmup_skip))


def gauge_trends(df):
    section("5. 口径温度计(按月)")
    t = df.groupby(["year", "month"]).agg(
        explore_ratio=("explore_ratio", "mean"),
        avg_dp=("avg_dp", "mean"),
        rows=("bj_date", "size"))
    print(t.round(4).to_string())
    t.to_csv(os.path.join(OUT_DIR, "monthly_gauge.csv"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    basic_check(df)
    day_df = build_day_table(df)
    shape_plots(df, day_df)
    run_v0_backtest(df, day_df)
    gauge_trends(df)
    section("完成")
    print("输出目录: {}".format(OUT_DIR))


if __name__ == "__main__":
    main()
