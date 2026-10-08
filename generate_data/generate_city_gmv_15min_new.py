#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 任务类型: py_spark
# ********************************************************************#
##author:zhaowenjiao
##create time:2026-09-16 10:00:00
##desc:ODT时序预估底层表生成(日分区版)。源表sp_feature_bub_order_pay_base_v9,按
##     city×product×bj_date×15min窗(0-95)双轨聚合(all_/explore_),空窗grid
##     补齐(NULL=无数据,0=有流量为零)。目标表odt_forecast_city_gmv_15min_day。
##     与generate_city_gmv_15min.py差异:分区为country_code,year,month,day
##     (原版为country_code,year,month)。
##remind:请在资源引用中添加需要引用的资源
##     dt_hm格式已确认(2026-09-15): 'yyyy-MM-dd HH:mm'
##     gmv完单语义若存疑,置CHECK_MODE=True复核后再生成
# ********************************************************************#
import datetime
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# ============================================================================
# 源表列(与desc对齐,小写)
# ============================================================================
SRC_TABLE = "international_rider_strategy.sp_feature_bub_order_pay_base_v9"
DST_TABLE = "international_rider_strategy.odt_forecast_city_gmv_15min_day"

KEY_COLS = ["city_id", "product_id", "bj_date", "dt_hm", "weekday"]
TRACK_COLS = ["gmv", "is_call", "is_finish", "hit_sp_explore"]
BG_COLS = ["dynamic_times", "pre_total_fee_dp", "road_dist_km"]
MKT_COLS = ["eta", "ar", "answer10", "bubble_per_driver", "uniqueemptydrivers",
            "bubbles", "order10", "waitpicktime",
            # 网格率列经2026-09-15报错列清单核实: 完单率只有交叉命名,无fgrid_finish_rate_fgrid
            "fgrid_call_rate_fgrid_14_days_14", "fgrid_finish_rate_tgrid_14_days_14",
            "tgrid_call_rate_tgrid_14_days_14", "tgrid_finish_rate_tgrid_14_days_14"]
ASTRO_COLS = ["astro_pax_cnt_call_28", "astro_pax_cnt_finish_28", "astro_pax_call_price_avg_3"]
WEATHER_COLS = ["realtime_temperature", "realtime_precipitation"]

# 建表列序(数据列,不含分区列)——insert时select顺序必须与此一致
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

DDL = """
create table if not exists international_rider_strategy.odt_forecast_city_gmv_15min_day (
    city_id          string  comment '城市ID',
    product_id       string  comment '产品ID',
    bj_date          string  comment '北京日期 yyyy-MM-dd',
    window_idx       int     comment '15min窗序号0-95 = hour*4+floor(minute/15)',
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
comment 'ODT时序预估底层表v1(日分区版):城市x产品x15min窗双轨聚合.空窗grid补齐:NULL=无数据,0=有流量为零.结果列分all/explore轨,背景列不分.宽进严出:mkt_/astro_为候选列,建模侧筛选'
partitioned by (country_code string, year string, month string, day string)
stored as parquet
"""


def insert_df_hdfs(spark, tb_name, df, partition_str):
    spark.conf.set("spark.sql.hive.manageFilesourcePartitions", "false")
    spark.sql("set hive.exec.dynamic.partition.mode = nonstrict")
    spark.sql("set hive.exec.dynamic.partition=true")
    spark.sql("use international_rider_strategy")
    df.repartition(1).createOrReplaceTempView("df_tb_for_insert")
    spark.sql("insert overwrite table {tb_name} "
              "partition({partition_str}) "
              "select * from df_tb_for_insert".format(tb_name=tb_name, partition_str=partition_str))


def run_check(spark, bj_start, bj_end, country_code, product_id):
    """Step1 口径确认: gmv完单语义(及时间列样例复核)。"""
    df = spark.table(SRC_TABLE).where(
        (F.col("country_code") == country_code) &
        (F.col("product_id") == product_id) &
        (F.col("bj_date").between(bj_start, bj_end))
    ).select("dt_hm", "bubble_time", "bj_date", "bj_hour", "hour", "weekday",
             "is_call", "is_finish", "gmv", "call_gmv", "finish_gmv")
    print("==== Step1a: dt_hm / 时间列样例 ====")
    df.show(20, truncate=False)
    print("==== Step1b: gmv语义检查(应为0: is_finish=0但gmv非零的行数) ====")
    df.where((F.col("is_finish") == 0) &
             (F.col("gmv").isNotNull()) & (F.col("gmv") != 0)).count()
    print("is_finish=0 且 gmv非0/非NULL 的行数 = %d" %
          df.where((F.col("is_finish") == 0) &
                   (F.col("gmv").isNotNull()) & (F.col("gmv") != 0)).count())


def parse_hh_mi(df):
    """已确认(2026-09-15 hive验证): dt_hm='yyyy-MM-dd HH:mm' → 小时第12-13位/分钟第15-16位。
    例: '2025-04-10 02:28' → hh=02, mi=28 → window_idx = 2*4+1 = 9
    (不用源表hour列: bj/local口径未标注,统一从dt_hm取)"""
    return (df.withColumn("hh", F.substring("dt_hm", 12, 2).cast("int"))
              .withColumn("mi", F.substring("dt_hm", 15, 2).cast("int"))
              .withColumn("window_idx", (F.col("hh") * 4 + F.floor(F.col("mi") / 15)).cast("int"))
              .drop("hh", "mi", "dt_hm"))


def build_agg(spark, bj_start, bj_end, country_code, product_id):
    # 分区剪枝: country_code/product_id是分区列(剪枝主力); bj_date是普通列剪不了,
    # 故补year-month分区条件辅助剪枝。长回填时建议按月循环跑(每次单月分区,剪枝最彻底)
    ym_start = bj_start[:7].replace("-", "")
    ym_end = bj_end[:7].replace("-", "")
    src = spark.table(SRC_TABLE).where(
        (F.col("country_code") == country_code) &
        (F.col("product_id") == product_id) &
        (F.col("bj_date").between(bj_start, bj_end)) &
        F.concat(F.lpad(F.col("year"), 4, "0"), F.lpad(F.col("month"), 2, "0")).between(ym_start, ym_end)
    ).select(KEY_COLS + TRACK_COLS + BG_COLS + MKT_COLS + ASTRO_COLS + WEATHER_COLS)

    df = parse_hh_mi(src)

    exp = F.col("hit_sp_explore")
    agg_exprs = [
        F.max("weekday").alias("weekday"),
        F.sum("gmv").alias("all_gmv"),
        F.sum("is_finish").alias("all_finish_cnt"),
        F.sum("is_call").alias("all_call_cnt"),
        F.count(F.lit(1)).alias("all_bubble_cnt"),
        F.sum(F.when(exp, F.col("gmv")).otherwise(F.lit(0))).alias("explore_gmv"),
        F.sum(F.when(exp, F.col("is_finish")).otherwise(F.lit(0))).alias("explore_finish_cnt"),
        F.sum(F.when(exp, F.col("is_call")).otherwise(F.lit(0))).alias("explore_call_cnt"),
        F.count(F.when(exp, F.lit(1))).alias("explore_bubble_cnt"),
        F.avg(exp.cast("int")).alias("explore_ratio"),
        F.avg("dynamic_times").alias("avg_dp"),
        F.avg("pre_total_fee_dp").alias("avg_price"),
        F.avg("road_dist_km").alias("avg_road_dist"),
        F.avg("eta").alias("mkt_avg_eta"),
        F.avg("ar").alias("mkt_avg_ar"),
        F.avg("answer10").alias("mkt_avg_answer10"),
        F.avg("bubble_per_driver").alias("mkt_avg_bubble_per_driver"),
        F.avg("uniqueemptydrivers").alias("mkt_avg_unique_empty_drivers"),
        F.avg("bubbles").alias("mkt_avg_bubbles"),
        F.avg("order10").alias("mkt_avg_order10"),
        F.avg("waitpicktime").alias("mkt_avg_wait_pick_time"),
        F.avg("fgrid_call_rate_fgrid_14_days_14").alias("mkt_avg_fgrid_call_rate"),
        F.avg("fgrid_finish_rate_tgrid_14_days_14").alias("mkt_avg_fgrid_finish_rate"),
        F.avg("tgrid_call_rate_tgrid_14_days_14").alias("mkt_avg_tgrid_call_rate"),
        F.avg("tgrid_finish_rate_tgrid_14_days_14").alias("mkt_avg_tgrid_finish_rate"),
        F.avg("astro_pax_cnt_call_28").alias("astro_avg_pax_cnt_call_28"),
        F.avg("astro_pax_cnt_finish_28").alias("astro_avg_pax_cnt_finish_28"),
        F.avg("astro_pax_call_price_avg_3").alias("astro_avg_pax_call_price_avg"),
        F.avg("realtime_temperature").alias("avg_temperature"),
        F.avg("realtime_precipitation").alias("avg_precipitation"),
    ]
    return df.groupBy("city_id", "product_id", "bj_date", "window_idx").agg(*agg_exprs)


def build_grid_and_join(spark, agg, country_code):
    """96格日历grid补空窗: NULL=无数据(缺口mask原料), 0=有流量为零。
    注意: spark必须显式传入——Spark3.2的DataFrame没有.sparkSession属性(3.4才加)。"""
    windows = spark.range(0, 96) \
        .withColumnRenamed("id", "window_idx").withColumn("window_idx", F.col("window_idx").cast("int"))
    # weekday不进grid: agg侧也有weekday,join后select会歧义;统一从agg侧取(空窗为NULL,与其他指标语义一致)
    day_keys = agg.select("city_id", "product_id", "bj_date").distinct()
    grid = day_keys.crossJoin(windows)

    out = (grid.join(agg, ["city_id", "product_id", "bj_date", "window_idx"], "left")
               .select(*OUT_DATA_COLS)
               .withColumn("country_code", F.lit(country_code))
               .withColumn("year", F.substring("bj_date", 1, 4))
               .withColumn("month", F.substring("bj_date", 6, 2))
               .withColumn("day", F.substring("bj_date", 9, 2)))
    return out


def run_sanity(spark, country_code):
    """Step3 成版三查,结果人工复核。双峰检查自动选数据量最大的城市作代表(全城混合会抹平形态)。"""
    print("==== Sanity1: 覆盖率(每天窗数应≈96) ====")
    spark.sql("""
        select city_id, count(distinct bj_date) as days,
               count(*) / count(distinct bj_date) as avg_windows_per_day
        from {tb} where country_code='{c}'
        group by city_id order by days desc limit 20
    """.format(tb=DST_TABLE, c=country_code)).show()
    print("==== Sanity2: 双峰形态样例(自动选数据量最大的城市) ====")
    spark.sql("""
        select window_idx, sum(all_gmv) as gmv
        from {tb}
        where country_code='{c}'
          and city_id = (select city_id from {tb} where country_code='{c}'
                         group by city_id order by sum(all_bubble_cnt) desc limit 1)
        group by window_idx order by window_idx
    """.format(tb=DST_TABLE, c=country_code)).show(100)


if __name__ == '__main__':
    # ============ 参数区(按月循环跑时改这里) ============
    CHECK_MODE = False         # dt_hm格式已确认,直接False生成
    COUNTRY_CODE = "BR"
    PRODUCT_ID = "16"          # 分区列过滤(剪枝提速的关键,当前只做16)
    BJ_START = "2025-05-01"    # 回填起始(先用短区间试跑,通过后放开/按月循环)
    BJ_END = "2025-05-07"
    # ====================================================

    spark = SparkSession.builder \
        .appName("odt_forecast_city_gmv_15min_day") \
        .enableHiveSupport() \
        .getOrCreate()

    if CHECK_MODE:
        run_check(spark, BJ_START, BJ_END, COUNTRY_CODE, PRODUCT_ID)
        spark.stop()
        exit(0)

    # Step0: 建表
    spark.sql(DDL)

    # Step2: 聚合 + grid + 动态分区写入
    agg = build_agg(spark, BJ_START, BJ_END, COUNTRY_CODE, PRODUCT_ID)
    out = build_grid_and_join(spark, agg, COUNTRY_CODE)
    insert_df_hdfs(spark, DST_TABLE, out, "country_code, year, month, day")

    # Step3: sanity
    run_sanity(spark, COUNTRY_CODE)

    spark.stop()
