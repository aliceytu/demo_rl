-- ============================================================================
-- ODT 时序预估底层表生成脚本
-- 目标表: international_rider_strategy.odt_forecast_city_gmv_15min
-- 源表:   international_rider_strategy.sp_feature_bub_order_pay_base_v9
-- 粒度:   city_id x product_id x bj_date x window_idx (15min, 0-95)
-- 口径:   bj_date(北京日期); 结果列分 all_/explore_ 双轨; 空窗由grid补齐(NULL=无数据,0=有流量为零)
-- 设计文档: memory/demo_rl_gmv_forecast_discussion.md + 算法理解/拉格朗日理解.txt
-- 待确认: Step 1 的 dt_hm 格式(影响 window_idx 解析,见 ★ 标记处)
-- ============================================================================


-- ============================================================================
-- Step 0: 建表 (已建则跳过)
-- 注意: country_code 只作为分区列,不在数据列中
-- ============================================================================
create table if not exists international_rider_strategy.odt_forecast_city_gmv_15min (
    -- ===== 键 =====
    city_id          string  comment '城市ID',
    product_id       string  comment '产品ID',
    bj_date          string  comment '北京日期 yyyy-MM-dd',
    window_idx       int     comment '15min窗序号0-95 = hour*4+floor(minute/15)',
    weekday          int     comment '星期0-6,由源表weekday透传,模板分组键',
    -- ===== 全量轨(策略口径,label所在) =====
    all_gmv          double  comment '全量轨:sum(gmv)完单口径——主label',
    all_finish_cnt   bigint  comment '全量轨:完单量(前缀特征/漏斗诊断)',
    all_call_cnt     bigint  comment '全量轨:呼叫量(前缀特征/漏斗诊断)',
    all_bubble_cnt   bigint  comment '全量轨:冒泡数(兼盘点行数)',
    -- ===== explore轨(随机派档,政策无关) =====
    explore_gmv      double  comment 'explore轨GMV(形状学习/口径温度计)',
    explore_finish_cnt bigint comment 'explore轨完单量',
    explore_call_cnt bigint  comment 'explore轨呼叫量',
    explore_bubble_cnt bigint comment 'explore轨冒泡数',
    -- ===== 口径监控 =====
    explore_ratio    double  comment 'explore占比(=explore_bubble_cnt/all_bubble_cnt)',
    -- ===== 核心背景协变量(派档上游,不分轨) =====
    avg_dp           double  comment '平均动调倍率(市场松紧,dynamic_times)',
    avg_price        double  comment '平均预估价(pre_total_fee_dp,派档前)',
    avg_road_dist    double  comment '平均里程(需求结构侦探)',
    -- ===== mkt_市场状态(供给侧+匹配,宽进候选,建模侧做方差+相关筛选) =====
    mkt_avg_eta      double  comment '平均等待匹配时长',
    mkt_avg_ar       double  comment '平均应答率',
    mkt_avg_answer10 double  comment '平均10min应答',
    mkt_avg_bubble_per_driver double comment '平均供需比(冒泡/司机)',
    mkt_avg_unique_empty_drivers double comment '平均空闲司机数(供给侧)',
    mkt_avg_bubbles  double  comment '平均网格冒泡密度(需求密度代理)',
    mkt_avg_order10  double  comment '平均10min完单',
    mkt_avg_wait_pick_time double comment '平均等待接驾时长(waitpicktime)',
    -- ===== mkt_网格家族代表(每族只取一个14天版) =====
    mkt_avg_fgrid_call_rate   double comment '出发网格呼叫率代表',
    mkt_avg_fgrid_finish_rate double comment '出发网格完单率代表',
    mkt_avg_tgrid_call_rate   double comment '目的网格呼叫率代表',
    mkt_avg_tgrid_finish_rate double comment '目的网格完单率代表',
    -- ===== astro构成代表(异常日mix突变候选,宽进) =====
    astro_avg_pax_cnt_call_28   double comment '用户活跃度代表(28天版)',
    astro_avg_pax_cnt_finish_28 double comment '用户质量代表(28天版)',
    astro_avg_pax_call_price_avg double comment '用户价格敏感代表(源列仅3/7天版,取7)',
    -- ===== 天气保险(二期协变量) =====
    avg_temperature  double  comment '均温',
    avg_precipitation double comment '均降水'
)
comment 'ODT时序预估底层表v1:城市x产品x15min窗双轨聚合.空窗grid补齐:NULL=无数据,0=有流量为零.分轨判据:结果列分all/explore,背景列不分.宽进严出:mkt_/astro_为候选列,建模侧筛选.所有特征取用时刻<=t0'
partitioned by (country_code string, year string, month string)
stored as parquet;


-- ============================================================================
-- Step 1: 口径确认 (★必须先跑,结果决定 Step 2 中 window_idx 的解析式★)
-- ============================================================================
-- 1a. dt_hm 格式 + gmv 完单语义 + 时间列对比
-- select dt_hm, bubble_time, bj_date, bj_hour, local_date, hour, weekday,
--        is_call, is_finish, gmv, call_gmv, finish_gmv
-- from international_rider_strategy.sp_feature_bub_order_pay_base_v9
-- where country_code='BR' and product_id='16' and year='2025' and month='05' and day='01'
-- limit 20;
--
-- 确认三件事:
--   (1) dt_hm 长什么样 → 决定下面 ★★ 处的 substr 位置
--   (2) gmv 在 is_finish=0 的行是否为 0/NULL → 是则 sum(gmv) 即完单口径,否则改用
--       sum(if(is_finish=1, gmv, 0))
--   (3) bubble_time 是北京口径还是当地口径(若 dt_hm 不可用需走备选)


-- ============================================================================
-- Step 2: 全量回填 (动态分区写入; 按月循环跑请见文件末尾"按月模板"说明)
-- ============================================================================
insert overwrite table international_rider_strategy.odt_forecast_city_gmv_15min
partition (country_code, year, month)

with src as (
    select
        city_id,
        product_id,
        bj_date,
        weekday,
        -- ★★ 待Step1确认: 假设 dt_hm = 'yyyyMMddHHmm' ★★
        --   小时 = 第9-10位, 分钟 = 第11-12位
        --   若 dt_hm='yyyy-MM-dd HH:mm' → 改 substr(..,12,2) 和 substr(..,15,2)
        --   若用 bubble_time='yyyy-MM-dd HH:mm:ss' → 改 substr(bubble_time,12,2)/substr(bubble_time,15,2)
        --   (不用源表hour列: 其bj/local口径未标注,统一从dt_hm取)
        cast(substr(dt_hm, 9, 2) as int)  as hh,
        cast(substr(dt_hm, 11, 2) as int) as mi,
        gmv, is_call, is_finish, hit_sp_explore,
        dynamic_times, pre_total_fee_dp, road_dist_km,
        eta, ar, answer10, bubble_per_driver, uniqueemptydrivers,
        bubbles, order10, waitpicktime,
        fgrid_call_rate_fgrid_14_days_14, fgrid_finish_rate_fgrid_14_days_14,
        tgrid_call_rate_tgrid_14_days_14, tgrid_finish_rate_tgrid_14_days_14,
        astro_pax_cnt_call_28, astro_pax_cnt_finish_28, astro_pax_call_price_avg_7,
        realtime_temperature, realtime_precipitation
    from international_rider_strategy.sp_feature_bub_order_pay_base_v9
    where bj_date >= '2025-05-01'          -- ← 回填起始日,按需修改
      and bj_date <= '2026-09-13'          -- ← 回填截止日,按需修改
      and country_code in ('BR')
      and product_id = 16
      and fp_version = "FP_01"
      and driverpretotalfee>0
      and dynamic_times >= 1
),

agg as (
    select
        city_id, product_id, bj_date,
        hh * 4 + floor(mi / 15)            as window_idx,
        max(weekday)                       as weekday,
        sum(gmv)                           as all_gmv,
        sum(is_finish)                     as all_finish_cnt,
        sum(is_call)                       as all_call_cnt,
        count(*)                           as all_bubble_cnt,
        sum(if(hit_sp_explore, gmv, 0))      as explore_gmv,
        sum(if(hit_sp_explore, is_finish, 0)) as explore_finish_cnt,
        sum(if(hit_sp_explore, is_call, 0))   as explore_call_cnt,
        count(if(hit_sp_explore, 1, null))    as explore_bubble_cnt,
        avg(cast(hit_sp_explore as int))  as explore_ratio,
        avg(dynamic_times)                as avg_dp,
        avg(pre_total_fee_dp)             as avg_price,
        avg(road_dist_km)                 as avg_road_dist,
        avg(eta)                          as mkt_avg_eta,
        avg(ar)                           as mkt_avg_ar,
        avg(answer10)                     as mkt_avg_answer10,
        avg(bubble_per_driver)            as mkt_avg_bubble_per_driver,
        avg(uniqueemptydrivers)           as mkt_avg_unique_empty_drivers,
        avg(bubbles)                      as mkt_avg_bubbles,
        avg(order10)                      as mkt_avg_order10,
        avg(waitpicktime)                 as mkt_avg_wait_pick_time,
        avg(fgrid_call_rate_fgrid_14_days_14)   as mkt_avg_fgrid_call_rate,
        avg(fgrid_finish_rate_fgrid_14_days_14) as mkt_avg_fgrid_finish_rate,
        avg(tgrid_call_rate_tgrid_14_days_14)   as mkt_avg_tgrid_call_rate,
        avg(tgrid_finish_rate_tgrid_14_days_14) as mkt_avg_tgrid_finish_rate,
        avg(astro_pax_cnt_call_28)        as astro_avg_pax_cnt_call_28,
        avg(astro_pax_cnt_finish_28)      as astro_avg_pax_cnt_finish_28,
        avg(astro_pax_call_price_avg_7)   as astro_avg_pax_call_price_avg,
        avg(realtime_temperature)         as avg_temperature,
        avg(realtime_precipitation)       as avg_precipitation
    from src
    group by city_id, product_id, bj_date, hh * 4 + floor(mi / 15)
),

-- 96格日历grid: 空窗显式保留(NULL=无数据),缺口mask的原料
grid as (
    select a.city_id, a.product_id, a.bj_date, g.window_idx, a.weekday
    from (select distinct city_id, product_id, bj_date, weekday from agg) a
    lateral view explode(sequence(0, 95)) g as window_idx
)

select
    -- 34个数据列,顺序严格对齐建表列序
    g.city_id, g.product_id, g.bj_date, g.window_idx, g.weekday,
    a.all_gmv, a.all_finish_cnt, a.all_call_cnt, a.all_bubble_cnt,
    a.explore_gmv, a.explore_finish_cnt, a.explore_call_cnt, a.explore_bubble_cnt,
    a.explore_ratio,
    a.avg_dp, a.avg_price, a.avg_road_dist,
    a.mkt_avg_eta, a.mkt_avg_ar, a.mkt_avg_answer10, a.mkt_avg_bubble_per_driver,
    a.mkt_avg_unique_empty_drivers, a.mkt_avg_bubbles, a.mkt_avg_order10,
    a.mkt_avg_wait_pick_time,
    a.mkt_avg_fgrid_call_rate, a.mkt_avg_fgrid_finish_rate,
    a.mkt_avg_tgrid_call_rate, a.mkt_avg_tgrid_finish_rate,
    a.astro_avg_pax_cnt_call_28, a.astro_avg_pax_cnt_finish_28,
    a.astro_avg_pax_call_price_avg,
    a.avg_temperature, a.avg_precipitation,
    -- 动态分区列
    'BR'                as country_code,
    substr(g.bj_date, 1, 4) as year,
    substr(g.bj_date, 6, 2) as month
from grid g
left join agg a
       on  g.city_id    = a.city_id
       and g.product_id = a.product_id
       and g.bj_date    = a.bj_date
       and g.window_idx = a.window_idx
;

-- [按月模板说明] 若一次回填16个月风险大(单任务失败全重跑),可按月循环:
--   把 where 条件改为单月 (如 year='2025' and month='06' 分区过滤更快),
--   分区写入改为静态: partition (country_code='BR', year='2025', month='06'),
--   并删掉 select 末尾的三个动态分区列。数梦上配置循环调度即可。


-- ============================================================================
-- Step 3: 成版 sanity 三查 (跑完 Step 2 后执行,结果贴回给 Claude 复核)
-- ============================================================================
-- ① 覆盖率: 每天窗口数应≈96
-- select city_id, count(distinct bj_date) as days,
--        count(*) / count(distinct bj_date) as avg_windows_per_day
-- from international_rider_strategy.odt_forecast_city_gmv_15min
-- where country_code='BR'
-- group by city_id order by days desc;
--
-- ② 双峰形态: 挑一个普通周三看96窗曲线 (bj_date 换成实际存在的周三)
-- select window_idx, sum(all_gmv) as gmv
-- from international_rider_strategy.odt_forecast_city_gmv_15min
-- where country_code='BR' and city_id='55000233' and bj_date='2025-05-07'
-- group by window_idx order by window_idx;
--
-- ③ 周末 vs 工作日日总量
-- select case when weekday in (5,6) then 'weekend' else 'weekday' end as typ,
--        count(distinct bj_date) as days, sum(all_gmv)/count(distinct bj_date) as gmv_per_day
-- from international_rider_strategy.odt_forecast_city_gmv_15min
-- where country_code='BR' and city_id='55000233'
-- group by case when weekday in (5,6) then 'weekend' else 'weekday' end;
