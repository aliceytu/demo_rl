#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ********************************************************************#
##author: zhaowenjiao
##create time: 2026-09-21
##desc: 城市级 V0 报告 — 合并回测分(V0a WMAPE) + 数据质量(null率/天数),
##     全 77 城落 CSV + 排名柱状图; 顺带算 null率 vs WMAPE 相关系数
##     (回答: 尾部高误差城市是不是数据质量差)
# ********************************************************************#
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/Users/didi/workspace/python-project/demo_rl/analysis/output"
DL = "/Users/didi/Downloads"


def main():
    bt = pd.read_csv(os.path.join(OUT, "v0_backtest_by_city.csv"), dtype={"city_id": str})
    ov = pd.read_csv(os.path.join(OUT, "per_city_overview.csv"), dtype={"city_id": str})
    m = bt.merge(ov, on="city_id", how="left")
    m["wmape_pct"] = (m.wmape_a * 100).round(2)
    m["null_ratio_pct"] = (m.null_ratio * 100).round(3)
    m = m.sort_values("wmape_a").reset_index(drop=True)
    m = m[["city_id", "wmape_pct", "null_ratio_pct", "days"]]
    m.to_csv(os.path.join(OUT, "city_report.csv"), index=False)
    m.to_csv(os.path.join(DL, "city_report.csv"), index=False)

    corr = m["wmape_pct"].corr(m["null_ratio_pct"])
    print("77城 WMAPE: min={:.2f}% median={:.2f}% max={:.2f}%".format(
        m.wmape_pct.min(), m.wmape_pct.median(), m.wmape_pct.max()))
    print("null率 vs V0a WMAPE 相关系数: {:.3f}".format(corr))
    print("\n前10(最好):")
    print(m.head(10).to_string(index=False))
    print("\n后10(最差):")
    print(m.tail(10).to_string(index=False))

    # 全城排名柱状图(最好在上)
    fig, ax = plt.subplots(figsize=(10, 14))
    y = range(len(m))
    colors = ["#c0392b" if w > 15 else "#7f8c8d" for w in m.wmape_pct]
    ax.barh(y, m.wmape_pct[::-1], color=colors[::-1], height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(m.city_id[::-1], fontsize=7)
    ax.set_xlabel("V0a WMAPE (%)")
    ax.set_title("per-city V0a WMAPE (red: >15%)")
    ax.axvline(11.76, color="blue", linestyle="--", linewidth=1, label="overall 11.76%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "city_wmape_ranking.png"), dpi=120)
    fig.savefig(os.path.join(DL, "city_wmape_ranking.png"), dpi=120)
    plt.close(fig)
    print("\nsaved: city_report.csv / city_wmape_ranking.png (output + Downloads)")


if __name__ == "__main__":
    main()
