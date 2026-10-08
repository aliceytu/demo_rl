#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 任务类型: py_spark
# ********************************************************************#
##author:zhaowenjiao
##create time:2026-09-16
##desc: 从 Hive 表 odt_forecast_city_gmv_15min_day(日分区版) 直接抽取数据,
##       全量城市落 HDFS csv（samples_city_local/BR/，后缀 _gmv_15min_day）,
##       供 GMV 时序预估支线初步数据分析拉取。
##       不做 product_id 过滤: 盘点显示 BR 每城恰好一个 product(96行/天),
##       全量导出 79 城;若未来出现多 product, 需改按 (city, product) 粒度处理。
##       与 export_broad_cnt_assistant 的落盘路径/格式保持一致。
##       注意: 表分区是 (country_code, year, month, day), bj_date 是普通列,
##       按 (year, month) 分区对枚举过滤即可剪枝(day级留给bj_date条件), 否则全表扫描。
##remind:请在资源引用中添加需要引用的资源
# ********************************************************************#
import datetime
from pyspark.sql import SparkSession
import pyspark.sql.functions as F


def build_ym_cond(bj_start, bj_end):
    """枚举 (year, month) 分区对, 保证分区剪枝。
    month 为两位字符串('05'), 与建表时 substring(bj_date, 6, 2) 写入口径一致。"""
    pairs = []
    cur = datetime.date(int(bj_start[:4]), int(bj_start[5:7]), 1)
    end = datetime.date(int(bj_end[:4]), int(bj_end[5:7]), 1)
    while cur <= end:
        pairs.append("(year='{}' and month='{}')".format(cur.strftime("%Y"), cur.strftime("%m")))
        if cur.month == 12:
            cur = datetime.date(cur.year + 1, 1, 1)
        else:
            cur = datetime.date(cur.year, cur.month + 1, 1)
    return "(" + " or ".join(pairs) + ")"


if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("export_odt_forecast_city_gmv_15min") \
        .enableHiveSupport() \
        .getOrCreate()
    spark.sql("use international_rider_strategy")

    # ============ 参数区 ============
    country_code = 'BR'
    # product_id 不做过滤: 全量导出 79 城(每城恰好一个 product, product 列仍在 select * 里)
    bj_start = "2025-05-01"
    bj_end = "2026-09-15"      # 改这里: 表回填到哪天填哪天; 当天数据不完整建议取昨天
    # ==============================

    ym_cond = build_ym_cond(bj_start, bj_end)

    # 城市列表从表里动态取(实际有数据的城市), 避免硬编码 70 城遗漏
    city_df = spark.sql("""
        select distinct city_id
        from odt_forecast_city_gmv_15min_day
        where country_code = '{c}'
          and {ym}
          and bj_date between '{s}' and '{e}'
        order by city_id
    """.format(c=country_code, ym=ym_cond, s=bj_start, e=bj_end))
    city_list = [row.city_id for row in city_df.collect()]
    print("city count: {}, cities: {}".format(len(city_list), city_list))

    # 从 Hive 表抽取(全量城市, 不做 product 过滤)
    export_sql = f"""
        select *
        from odt_forecast_city_gmv_15min_day
        where country_code = '{country_code}'
          and {ym_cond}
          and bj_date between '{bj_start}' and '{bj_end}'
    """
    res_df = spark.sql(export_sql).cache()
    print("total rows: {}".format(res_df.count()))
    print("columns: {}".format(res_df.columns))

    # 每城盘点(不落盘, 供城市可用性盘点: 天数/窗口完整率)
    res_df.groupBy("city_id") \
        .agg(F.countDistinct("bj_date").alias("days"), F.count("*").alias("rows")) \
        .orderBy("days", ascending=False) \
        .show(100, truncate=False)

    # 全量城市落 HDFS csv（repartition(1) 单 part 文件, 与生成脚本口径一致）
    saved_path_dir = "hdfs://DClusterUS1/user/root_international_algo_dev/international_rider_strategy/hive/international_rider_strategy/samples_city_local/" + country_code + "/"
    city_save_path = saved_path_dir + country_code + "#" + bj_start + "_" + bj_end + "_gmv_15min_day.txt"

    res_df.repartition(1).write.csv(city_save_path, compression='gzip', header=True, sep="\t", mode='overwrite')
    spark.stop()
