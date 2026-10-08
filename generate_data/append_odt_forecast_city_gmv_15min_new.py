#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 任务类型: py_spark
# ********************************************************************#
##author:zhaowenjiao
##create time:2026-09-16 10:30:00
##desc:迁移脚本(一次性)。把 odt_forecast_city_gmv_15min(分区country_code,year,month)
##     的内容搬到 odt_forecast_city_gmv_15min_day(分区country_code,year,month,day)。
##     day分区值 = bj_date第9-10位(与generate_city_gmv_15min_new.py的推导一致)。
##     自带DDL自建表;默认OVERWRITE=True幂等重跑(只覆盖本次数据涉及的分区);
##     迁移后同一日期范围不要再跑generate_city_gmv_15min_new.py,防止双写。
##remind:请在资源引用中添加需要引用的资源
# ********************************************************************#
import datetime
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

SRC_TABLE = "international_rider_strategy.odt_forecast_city_gmv_15min"
DST_TABLE = "international_rider_strategy.odt_forecast_city_gmv_15min_day"

# 新表数据列(不含分区列),与generate_city_gmv_15min_new.py的OUT_DATA_COLS一致
OUT_DATA_COLS = [
    "city_id", "product_id", "bj_date", "window_idx", "weekday",
    "all_gmv", "all_finish_cnt", "all_call_cnt", "all_bubble_cnt",
    "explore_gmv", "explore_finish_cnt", "explore_call_cnt", "explore_bubble_cnt",
    "explore_ratio",
    "avg_dp", "avg_price", "avg_road_dist",
    "mkt_avg_eta", "mkt_avg_ar", "mkt_avg_answer10", "mkt_avg_bubble_per_driver",
    "mkt_avg_unique_empty_drivers", "mkt_avg_bubbles", "mkt_avg_order10",
    "mkt_avg_wait_pick_time",
    "mkt_avg_fgrid_call_rate", "mkt_avg_fgrid_finish_rate",
    "mkt_avg_tgrid_call_rate", "mkt_avg_tgrid_finish_rate",
    "astro_avg_pax_cnt_call_28", "astro_avg_pax_cnt_finish_28",
    "astro_avg_pax_call_price_avg",
    "avg_temperature", "avg_precipitation",
]

# 与generate_city_gmv_15min_new.py的DDL一致——自建表,解除对该脚本执行顺序的依赖
DDL = """
create table if not exists international_rider_strategy.odt_forecast_city_gmv_15min_day (
    city_id          string  comment '城市ID',
    product_id       string  comment '产品ID',
    bj_date          string  comment '北京日期 yyyy-MM-dd',
    window_idx       int     comment '15min窗序号0-95',
    weekday          int     comment '星期0-6,模板分组键',
    all_gmv          double  comment '全量轨:sum(gmv)完单口径——主label',
    all_finish_cnt   bigint  comment '全量轨:完单量',
    all_call_cnt     bigint  comment '全量轨:呼叫量',
    all_bubble_cnt   bigint  comment '全量轨:冒泡数(兼盘点)',
    explore_gmv      double  comment 'explore轨GMV(随机派档,政策无关)',
    explore_finish_cnt bigint comment 'explore轨完单量',
    explore_call_cnt bigint  comment 'explore轨呼叫量',
    explore_bubble_cnt bigint comment 'explore轨冒泡数',
    explore_ratio    double  comment 'explore占比(口径温度计)',
    avg_dp           double  comment '平均动调倍率(市场松紧)',
    avg_price        double  comment '平均预估价(派档前)',
    avg_road_dist    double  comment '平均里程(需求结构侦探)',
    mkt_avg_eta      double  comment '平均等待匹配时长',
    mkt_avg_ar       double  comment '平均应答率',
    mkt_avg_answer10 double  comment '平均10min应答',
    mkt_avg_bubble_per_driver double comment '平均供需比',
    mkt_avg_unique_empty_drivers double comment '平均空闲司机数(供给侧)',
    mkt_avg_bubbles  double  comment '平均网格冒泡密度',
    mkt_avg_order10  double  comment '平均10min完单',
    mkt_avg_wait_pick_time double comment '平均等待接驾时长',
    mkt_avg_fgrid_call_rate   double comment '出发网格呼叫率代表',
    mkt_avg_fgrid_finish_rate double comment '出发网格完单率代表',
    mkt_avg_tgrid_call_rate   double comment '目的网格呼叫率代表',
    mkt_avg_tgrid_finish_rate double comment '目的网格完单率代表',
    astro_avg_pax_cnt_call_28   double comment '用户活跃度代表',
    astro_avg_pax_cnt_finish_28 double comment '用户质量代表',
    astro_avg_pax_call_price_avg double comment '用户价格敏感代表(源列仅3天版)',
    avg_temperature  double  comment '均温(二期协变量)',
    avg_precipitation double comment '均降水(二期协变量)'
)
comment 'ODT时序预估底层表v1(日分区版,由月分区版迁移):城市x产品x15min窗双轨聚合'
partitioned by (country_code string, year string, month string, day string)
stored as parquet
"""


def insert_df_hdfs(spark, tb_name, df, partition_str, overwrite=False):
    """动态分区写入。overwrite=False 时 append(insert into),重复跑会重复写。"""
    spark.conf.set("spark.sql.hive.manageFilesourcePartitions", "false")
    spark.sql("set hive.exec.dynamic.partition.mode = nonstrict")
    spark.sql("set hive.exec.dynamic.partition=true")
    spark.sql("use international_rider_strategy")
    df.repartition(1).createOrReplaceTempView("df_tb_for_insert")
    op = "insert overwrite" if overwrite else "insert into"
    spark.sql("{op} table {tb_name} "
              "partition({partition_str}) "
              "select * from df_tb_for_insert".format(op=op, tb_name=tb_name,
                                                     partition_str=partition_str))


def build_migrate_df(spark, country_code, ym_start, ym_end):
    """读老表(分区country_code,year,month,剪枝过滤),补day列凑齐新表分区。
    select顺序 = 数据列(表内列序) + country_code,year,month,day(与partition子句顺序一致)。"""
    src = spark.table(SRC_TABLE).where(
        (F.col("country_code") == country_code) &
        F.concat(F.lpad(F.col("year"), 4, "0"), F.lpad(F.col("month"), 2, "0")).between(ym_start, ym_end)
    )
    return (src.select(*OUT_DATA_COLS, "country_code", "year", "month")
               .withColumn("day", F.substring("bj_date", 9, 2)))


def run_sanity(spark, country_code, ym_start, ym_end):
    """迁移前后行数对比(人工复核用)。"""
    ym_cond = "concat(lpad(year,4,'0'), lpad(month,2,'0')) between '{s}' and '{e}'" \
        .format(s=ym_start, e=ym_end)
    print("==== Sanity1: 老表行数 ====")
    src_cnt = spark.sql("select count(*) as cnt from {tb} where country_code='{c}' and {ym}"
                        .format(tb=SRC_TABLE, c=country_code, ym=ym_cond)).first()["cnt"]
    print("source rows = %d" % src_cnt)
    print("==== Sanity2: 新表该区间行数(分区country_code/year/month/day) ====")
    dst_cnt = spark.sql("select count(*) as cnt from {tb} where country_code='{c}' and {ym}"
                        .format(tb=DST_TABLE, c=country_code, ym=ym_cond)).first()["cnt"]
    print("target rows = %d" % dst_cnt)
    print("==== Sanity3: day分区样例分布(应为01-31且分布合理) ====")
    spark.sql("""
        select year, month, day, count(*) as cnt
        from {tb}
        where country_code='{c}' and {ym}
        group by year, month, day order by year, month, day
    """.format(tb=DST_TABLE, c=country_code, ym=ym_cond)).show(40)


if __name__ == '__main__':
    # ============ 参数区(按月循环跑时改这里) ============
    OVERWRITE = True           # 默认True=幂等(覆盖本次数据涉及的分区,重跑不重复);
                               # 注意:动态分区overwrite只覆盖有数据的分区,区间内无数据的旧分区不会被清
    COUNTRY_CODE = "BR"
    YM_START = "202505"        # 老表year-month分区剪枝范围
    YM_END = "202505"
    # ====================================================

    spark = SparkSession.builder \
        .appName("append_odt_forecast_city_gmv_15min_day") \
        .enableHiveSupport() \
        .getOrCreate()

    # 自建表(幂等)
    spark.sql(DDL)

    out = build_migrate_df(spark, COUNTRY_CODE, YM_START, YM_END)
    insert_df_hdfs(spark, DST_TABLE, out, "country_code, year, month, day",
                   overwrite=OVERWRITE)

    run_sanity(spark, COUNTRY_CODE, YM_START, YM_END)

    spark.stop()
