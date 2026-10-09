#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ********************************************************************#
##author: zhaowenjiao
##create time: 2026-09-17
##desc: 轨构成排查 — 回答"all_gmv null 的窗是不是 explore 单撑起来的"
##  1) 背景 null 窗(剔除 05-17/08-22 空白日)的 explore_gmv 量级分布
##     —— 有没有 explore_gmv 很大而 all_gmv null 的窗
##  2) 背景 null 窗的 call 构成: all_call vs explore_call
##     —— "只有explore单" vs "主轨有单但gmv失败"
##  3) 全表: all_gmv 到底含不含 explore 轨(sum(explore)/sum(all) vs explore_ratio列)
# ********************************************************************#
import os
import numpy as np
import pandas as pd

DATA_PATH = "/Users/didi/workspace/data/sp_features/tarnet/BR#2025-05-01_2026-09-15_gmv_15min_day.txt"
OUT_DIR = "/Users/didi/workspace/python-project/demo_rl/analysis/output"

dtype_map = {
    "city_id": "string", "product_id": "string", "bj_date": "string",
    "country_code": "string", "year": "string", "month": "string", "day": "string",
}


def section(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH, sep="\t", compression="gzip", dtype=dtype_map)
    df["bj_date"] = pd.to_datetime(df["bj_date"])

    null_mask = df.all_gmv.isna()
    outage_days = pd.to_datetime(["2025-05-17", "2025-08-22"])
    is_outage = df.bj_date.isin(outage_days)
    # 背景 null: 全表 null 减去空白日
    bg_null = null_mask & ~is_outage

    # ============ 1. 背景 null 窗的 explore_gmv 量级 ============
    section("1. 背景 null 窗(剔除空白日)的 explore_gmv 分布")
    bg = df[bg_null]
    print("背景 null 窗总数: {}".format(len(bg)))
    print("其中 explore_gmv > 0 的比例: {:.1%}".format((bg.explore_gmv > 0).mean()))
    print("其中 explore_gmv > 100 的比例: {:.1%}".format((bg.explore_gmv > 100).mean()))
    print("explore_gmv 分位: min={:.1f} p50={:.1f} p90={:.1f} max={:.1f}".format(
        bg.explore_gmv.min(), bg.explore_gmv.median(), bg.explore_gmv.quantile(0.9), bg.explore_gmv.max()))
    top = bg.nlargest(10, "explore_gmv")[["city_id", "bj_date", "window_idx", "explore_gmv", "all_call_cnt", "explore_call_cnt"]]
    print("\nexplore_gmv 最大的 10 个 null 窗:")
    print(top.to_string(index=False))

    # ============ 2. 背景 null 窗的 call 轨构成 ============
    section("2. 背景 null 窗的 call 构成(all_call vs explore_call)")
    call = bg[["all_call_cnt", "explore_call_cnt", "all_bubble_cnt", "explore_bubble_cnt"]]
    both_pos = (bg.all_call_cnt > 0) & (bg.explore_call_cnt > 0)
    only_explore = (bg.all_call_cnt.fillna(0) == 0) & (bg.explore_call_cnt > 0)
    all_eq_explore = (bg.all_call_cnt == bg.explore_call_cnt)
    main_call_exists = bg.all_call_cnt > bg.explore_call_cnt
    print("all_call_cnt > 0 的比例: {:.1%}".format((bg.all_call_cnt > 0).mean()))
    print("explore_call_cnt > 0 的比例: {:.1%}".format((bg.explore_call_cnt > 0).mean()))
    print("只有explore单(all_call=0且explore>0): {:.1%}".format(only_explore.mean()))
    print("all_call 恰好 == explore_call: {:.1%}".format(all_eq_explore.mean()))
    print("主轨有单(all_call > explore_call): {:.1%}  ← 有主轨单却gmv null = 归因失败信号".format(main_call_exists.mean()))
    print("call 全为0: {:.1%}".format((bg.all_call_cnt.fillna(0) == 0).mean()))

    # ============ 3. 全表: all_gmv 含不含 explore 轨 ============
    section("3. all_gmv 是否包含 explore 轨(全表口径)")
    s_all = df.all_gmv.sum()
    s_expl = df.explore_gmv.sum()
    print("sum(explore_gmv) / sum(all_gmv) = {:.3%}".format(s_expl / s_all))
    print("explore_ratio 列全表均值 = {:.3%}".format(df.explore_ratio.mean()))
    both = df[df.all_gmv.notna() & (df.all_gmv > 0)]
    gt = (both.explore_gmv > both.all_gmv).sum()
    print("explore_gmv > all_gmv 的窗数(两列都非null): {} / {} ({:.3%})".format(
        gt, len(both), gt / len(both)))
    # 若 all 含 explore, 两列比应 ≈ explore_ratio 且 explore 永不 > all
    # 若 all 只含主轨, 两列比应 ≈ explore_ratio/(1-explore_ratio) ≈ 41%

    section("完成")


if __name__ == "__main__":
    main()
