#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ********************************************************************#
##author: zhaowenjiao
##create time: 2026-09-17
##desc: 77 城 × 96 窗的日内 GMV 占比, 按日平均(每天等权)落 CSV
##     每城一行, 每窗一列; 行和应=1(校验)
##     与 v0_analysis.shape_plots 的"总和相除"不同: 那是按量加权(大日主导),
##     这里是先算每日占比再跨日平均
# ********************************************************************#
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocess.clean_data import load_clean

OUT_PATH = "/Users/didi/workspace/python-project/odt_rl/analysis/output/intraday_share_avg_by_city.csv"


def main():
    df = load_clean()
    # 每天每城的日总
    day = df.groupby(["city_id", "bj_date"])["all_gmv"].sum().rename("day_sum")
    # 每天每城每窗的 GMV
    win = df.groupby(["city_id", "bj_date", "window_idx"])["all_gmv"].sum().rename("win_sum")
    # 每日占比 = 窗GMV / 当日总
    share = (win / day).rename("share").reset_index()
    # 跨日平均(每天等权)
    avg = share.groupby(["city_id", "window_idx"])["share"].mean().rename("avg_share")
    wide = avg.reset_index().pivot(index="city_id", columns="window_idx", values="avg_share")
    wide.to_csv(OUT_PATH)
    print("shape: {} 城 × {} 窗".format(*wide.shape))
    print("每城占比合计(min/max, 应≈1): {:.4f} / {:.4f}".format(
        wide.sum(axis=1).min(), wide.sum(axis=1).max()))
    print("saved: {}".format(OUT_PATH))


if __name__ == "__main__":
    main()
