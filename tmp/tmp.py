select estimate_id, hit_sp_explore,hit_driver_sp_explore,
ecrlist,crlist,erlist
from sp_feature_bub_order_pay_base_v9
where country_code = 'BR'
and product_id = 16
and concat_ws('-',year,month,day) = '2026-09-01'
and city_id = '55000063'
and hit_sp_explore = True

select
     count(*)                                                        as total_cnt,
    sum(case when ecrlist is null or ecrlist = '' then 1 else 0 end) as ecrlist_missing_cnt,
    sum(case when crlist  is null or crlist  = '' then 1 else 0 end) as crlist_missing_cnt,
    sum(case when (ecrlist is null or ecrlist='') and hit_sp_explore then 1 else 0 end) as ecr_miss_explore_cnt,
    sum(case when (ecrlist is null or ecrlist='') and not hit_sp_explore then 1 else 0 end) as ecr_miss_noexplore_cnt,
    sum(case when (crlist is null or crlist='') and hit_sp_explore then 1 else 0 end) as cr_miss_explore_cnt,
    sum(case when (crlist is null or crlist='') and not hit_sp_explore then 1 else 0 end) as cr_miss_noexplore_cnt
 from sp_feature_bub_order_pay_base_v9
 where country_code = 'BR'
 and product_id = 16
 and concat_ws('-',year,month,day) = '2026-09-01'
 and city_id = '55000063'



select city_id,count(*) cnt
from odt_forecast_city_gmv_15min_day
group by city_id


select city_id,
         count(distinct bj_date)                            as days,
         count(*)                                           as rows_cnt,
         count(*) / count(distinct bj_date)                 as rows_per_day,   -- 应恰好=96
         sum(case when all_gmv is null then 1 else 0 end) / count(*) as null_window_ratio
  from odt_forecast_city_gmv_15min_day
  group by city_id
  order by days desc;

select year,month,count(*)
from odt_forecast_city_gmv_15min_day
group by year,month
order by year,month asc

