#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ********************************************************************#
##author: zhaowenjiao
##create time: 2026-09-17
##desc: 数据清洗 — 训练管线共用加载入口
##  1) 剔除 2025-05-17 / 2025-08-22:
##     全表 77城×96窗 all_gmv 全 null 的空白日(有call有冒泡但gmv字段缺失),
##     占全表 null 的 54%, 判定为数据缺陷(见 analysis/null_diagnose.py)
##  2) 背景 null→0: 深夜无完单的真实零(工作日当地凌晨, 两轨gmv皆空,
##     explore聚合带otherwise(0)而all裸sum致null, SQL写法差异)
##  3) 体检打印: 清洗前后行数 / 清洗后 day_total==0 天数(应=0, 全0天恰为两空白日×77城)
##用法: 其他脚本 from data_preprocess.clean_data import load_clean
# ********************************************************************#
import os
import pickle
import numpy as np
import pandas as pd

DATA_PATH = "/Users/didi/workspace/data/sp_features/tarnet/BR#2025-05-01_2026-09-15_gmv_15min_day.txt"
CACHE_DIR = "/Users/didi/workspace/python-project/demo_rl/analysis/output"
CLEAN_VERSION = "v2"   # 清洗逻辑变更时 +1, 旧缓存自动失效


def _cache_path():
    """缓存键 = 清洗版本 + 源文件mtime: 数据重导或逻辑变更自动失效"""
    mtime = int(os.path.getmtime(DATA_PATH))
    return os.path.join(CACHE_DIR, "df_clean_{}_{}.pkl".format(CLEAN_VERSION, mtime))
BAD_DATES = ["2025-05-17", "2025-08-22"]
GMV_COLS = ["all_gmv", "explore_gmv"]

dtype_map = {
    "city_id": "string", "product_id": "string", "bj_date": "string",
    "country_code": "string", "year": "string", "month": "string", "day": "string",
}


def load_raw(path=DATA_PATH):
    df = pd.read_csv(path, sep="\t", compression="gzip", dtype=dtype_map)
    df["bj_date"] = pd.to_datetime(df["bj_date"])
    return df


KEY_COLS = {"city_id", "product_id", "bj_date", "window_idx", "weekday",
            "country_code", "year", "month", "day"}


def clean(df):
    """剔除空白日 + 全列 null 规范化, 返回清洗后副本:
    - 填0组: gmv列/计数列/explore_ratio(空窗=无事件, 真实0)
    - 城级中位填组: 均值类指标(空窗无值, 填0会污染语义, 用城级中位兜底)
    - 关键列(城市/日期/窗/星期)若null: 直接丢行, 不瞎填"""
    n0 = len(df)
    bad = pd.to_datetime(BAD_DATES)
    df2 = df[~df.bj_date.isin(bad)].copy()

    # 先普查(填之前的原始 null 面貌)
    census = df2.isna().sum()
    census = census[census > 0]
    if len(census):
        print("clean 前 null 普查(行数>0 的列):")
        print(census.to_string())

    # 丢弃高缺失列(78% null): 填中位=造假特征, 一期不用(二期有完整天气源再回填)
    drop_cols = ["astro_avg_pax_call_price_avg", "avg_temperature", "avg_precipitation"]
    df2 = df2.drop(columns=drop_cols)
    print("丢弃高缺失列(78% null): {}".format(drop_cols))

    # 填0组
    zero_fill = GMV_COLS + [c for c in df2.columns if c.endswith("_cnt")] + ["explore_ratio"]
    zero_fill = [c for c in zero_fill if c in df2.columns]
    df2[zero_fill] = df2[zero_fill].fillna(0.0)
    if zero_fill:
        print("填0: {}".format(zero_fill))

    # 城级中位填组
    median_fill = [c for c in df2.columns
                   if c not in KEY_COLS and c not in zero_fill and df2[c].isna().any()]
    for c in median_fill:
        df2[c] = df2[c].fillna(df2.groupby("city_id")[c].transform("median"))
    if median_fill:
        print("城级中位填: {}".format(median_fill))

    # 关键列兜底: 丢行
    for c in ["city_id", "bj_date", "window_idx", "weekday"]:
        if df2[c].isna().any():
            n = df2[c].isna().sum()
            df2 = df2[df2[c].notna()]
            print("关键列 {} 有 {} null, 已丢行".format(c, n))

    print("clean: rows {} -> {} (剔除 {} 行), 剩余 null 总数: {}".format(
        n0, len(df2), n0 - len(df2), df2.isna().sum().sum()))
    return df2


def verify(df):
    """体检: 清洗后不应再有全天0GMV的(城,日)"""
    day0 = df.groupby(["city_id", "bj_date"])["all_gmv"].sum().reset_index()
    zero = day0[day0.all_gmv <= 0]
    print("清洗后 day_total<=0 的(城,日)数: {} (应=0)".format(len(zero)))
    return zero


def load_clean(cache=True):
    path = _cache_path()
    if cache and os.path.exists(path):
        print("load cache: {}".format(path))
        with open(path, "rb") as f:
            return pickle.load(f)
    df = clean(load_raw())
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(df, f)
        print("saved cache: {}".format(path))
    return df


if __name__ == "__main__":
    df = load_clean(cache=True)   # 顺手存 pickle 缓存, 后续脚本直接读
    verify(df)
