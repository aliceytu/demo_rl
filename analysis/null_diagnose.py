#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ********************************************************************#
##author: zhaowenjiao
##create time: 2026-09-17
##desc: null 专项排查 — 2025-05/08 两个峰值月的 null
##  1) 按城分布: 是不是所有城都有, 还是集中在少数城
##  2) 按天分布: 散布整月, 还是集中在某几天(平台事件)
##  3) 按窗分布: 是否仍是深夜窗(53-62)形态, 还是另类
##  4) 交叉列: null 窗里 call/bubble/explore 是否也空 → 无单 vs 归因失败
# ********************************************************************#
import os
import numpy as np
import pandas as pd

DATA_PATH = "/Users/didi/workspace/data/sp_features/tarnet/BR#2025-05-01_2026-09-15_gmv_15min_day.txt"
OUT_DIR = "/Users/didi/workspace/python-project/odt_rl/analysis/output"

dtype_map = {
    "city_id": "string", "product_id": "string", "bj_date": "string",
    "country_code": "string", "year": "string", "month": "string", "day": "string",
}


def section(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


def month_sub(df, y, m):
    return df[(df.bj_date.dt.year == y) & (df.bj_date.dt.month == m)]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH, sep="\t", compression="gzip", dtype=dtype_map)
    df["bj_date"] = pd.to_datetime(df["bj_date"])

    # ============ 1. 按城分布 ============
    section("1. null 按城分布(2025-05 / 2025-08 / 基线对照 2026-05)")
    rows = []
    for name, sub in [("2025-05", month_sub(df, 2025, 5)),
                      ("2025-08", month_sub(df, 2025, 8)),
                      ("2026-05(对照)", month_sub(df, 2026, 5))]:
        nulls = sub[sub.all_gmv.isna()]
        pc = nulls.groupby("city_id").size()
        total = len(sub)
        rows.append([name, len(nulls), pc.min() if len(pc) else 0,
                     pc.median(), pc.max(), pc.idxmax() if len(pc) else "-",
                     len(pc), df.city_id.nunique()])
    print(pd.DataFrame(rows, columns=["月份", "null总数", "每城min", "每城中位",
                                      "每城max", "max城", "有null的城数", "总城数"]).to_string(index=False))
    # 同一批城在两个月都坏吗?
    m5 = month_sub(df, 2025, 5)
    m8 = month_sub(df, 2025, 8)
    pc5 = m5[m5.all_gmv.isna()].groupby("city_id").size()
    pc8 = m8[m8.all_gmv.isna()].groupby("city_id").size()
    both = pd.concat([pc5, pc8], axis=1, keys=["may", "aug"]).fillna(0)
    print("\n各城两月 null 数(前15):")
    print(both.sort_values("may", ascending=False).head(15).to_string())

    # ============ 2. 按天分布 ============
    section("2. null 按天分布(2025-05 和 2025-08, 每天全表理论窗数=77*96=7392)")
    for name, sub in [("2025-05", month_sub(df, 2025, 5)),
                      ("2025-08", month_sub(df, 2025, 8))]:
        per_day = sub[sub.all_gmv.isna()].groupby(sub.bj_date.dt.date).size()
        print("\n{}: 日null数 min={} 中位={} max={}  (散布整月→各日接近; 事件爆发→个别日尖峰)".format(
            name, per_day.min(), int(per_day.median()), per_day.max()))
        print(per_day.to_string())

    # ============ 3. 按窗分布 ============
    section("3. null 按窗分布(两月合计 vs 全表对照)")
    m_peak = df[(df.bj_date.dt.year == 2025) & (df.bj_date.dt.month.isin([5, 8]))]
    w_peak = m_peak[m_peak.all_gmv.isna()].groupby("window_idx").size()
    w_all = df[df.all_gmv.isna()].groupby("window_idx").size()
    w_peak = w_peak / w_peak.sum()
    w_all = w_all / w_all.sum()
    comp = pd.concat([w_peak, w_all], axis=1, keys=["两月", "全表"]).fillna(0)
    print("两月 top5 窗: {}".format(comp["两月"].nlargest(5).round(4).to_dict()))
    print("全表 top5 窗: {}".format(comp["全表"].nlargest(5).round(4).to_dict()))

    # ============ 4. 交叉列判别 ============
    section("4. null 窗的交叉列(无单 vs 归因失败)")
    for name, sub in [("两月", m_peak), ("全表", df), ("基线月2026-05", month_sub(df, 2026, 5))]:
        nn = sub[sub.all_gmv.isna()]
        if len(nn) == 0:
            print("{}: 无 null".format(name))
            continue
        call_nn = nn.all_call_cnt.isna().mean()
        call0 = (nn.all_call_cnt.fillna(-1) == 0).mean()
        bubble0 = (nn.all_bubble_cnt.fillna(-1) == 0).mean()
        expl_nn = nn.explore_gmv.isna().mean()
        print("{}: null窗数={} | call也null={:.1%} | call==0={:.1%} | bubble==0={:.1%} | explore也null={:.1%}".format(
            name, len(nn), call_nn, call0, bubble0, expl_nn))

    section("完成")


if __name__ == "__main__":
    main()
