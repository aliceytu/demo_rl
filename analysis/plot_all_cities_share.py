#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ********************************************************************#
##author: zhaowenjiao
##create time: 2026-09-17
##desc: 77 城日内占比曲线同图展示(横轴=window_idx, 纵轴=日均GMV占比),
##     一条城一条线, 灰虚线=均匀线 1/96 参照
# ********************************************************************#
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "/Users/didi/workspace/python-project/demo_rl/analysis/output/intraday_share_avg_by_city.csv"
OUT_PATH = "/Users/didi/workspace/python-project/demo_rl/analysis/output/all_cities_intraday_share.png"


def main():
    df = pd.read_csv(CSV_PATH, index_col="city_id")
    x = df.columns.astype(int)
    fig, ax = plt.subplots(figsize=(16, 8))
    for city, row in df.iterrows():
        ax.plot(x, row.values, alpha=0.5, linewidth=0.6)
    ax.axhline(1 / 96, color="gray", linestyle="--", linewidth=0.8, label="uniform 1/96")
    ax.set_xlabel("window_idx (15min)")
    ax.set_ylabel("intraday gmv share (daily avg)")
    ax.set_title("77 cities intraday GMV share curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=120)
    print("saved: {}".format(OUT_PATH))


if __name__ == "__main__":
    main()
