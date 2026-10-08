# 1. 初始化环境 读取离线数据
# import
import findspark

findspark.init()  # 初始化spark的环境
import json
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import cvxpy as cp
import os
import logging
from sklearn.metrics import roc_auc_score
import sys
import math
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
from causalmldp.inference.tree import UpliftRandomForestClassifier

sys.path.append("/home/xiaoju/chenchunrong/project/BR")
np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)
plt.style.use('ggplot')

from pyspark.sql import SparkSession
from datetime import datetime
from datetime import timedelta
from pyspark.sql.types import *

os.environ['PYSPARK_SUBMIT_ARGS'] = """ --queue root.international_algo_dev
                                        --master yarn
                                        --driver-memory 10g
                                        --executor-cores 8
                                        --executor-memory 8G
                                        --num-executors 16
                                        pyspark-shell"""

# 关闭所有已存在的SparkContext
try:
    sc = SparkContext.getOrCreate()
    sc.stop()
except:
    pass

spark = SparkSession.builder \
    .master("local") \
    .appName("BR_sp_feature_grid14") \
    .enableHiveSupport() \
    .getOrCreate()

sc = spark.sparkContext
spark.sql("use international_rider_strategy")

date_all = ["2026-05-01"]


def get_dt_by_delta(dt_ori, delta_days):
    """
    得到dt_ori的delta_days天后的datetime形式
    """
    dt_target = (datetime.strptime(dt_ori, "%Y-%m-%d") + timedelta(delta_days)).strftime("%Y-%m-%d")
    return dt_target


def get_dt_by_delta_2(dt_ori, delta_days):
    """
    得到dt_ori的delta_days天前的datetime形式
    """
    dt_target = (datetime.strptime(dt_ori, "%Y-%m-%d") - timedelta(delta_days)).strftime("%Y-%m-%d")
    return dt_target

cur_date = date_all[0]
e_data = get_dt_by_delta(cur_date,37)
s_data = cur_date
s_data,e_data


# import os
country_code = 'BR'
product_id = "16"
city=55000233

#从数仓拉数据

os.system("mkdir -p /home/xiaoju/chenchunrong/project/BR/data/{country_code}_{s_data}_{e_data}_DATA".format(s_data=s_data,e_data=e_data,country_code=country_code))
# os.system("""  hdfs dfs -get /user/root_international_algo_dev/international_rider_strategy/hive/international_rider_strategy/samples_city_local/{country_code}/{country_code}#{product_id}#{city}_{s_data}_{e_data}_simu_v9.txt
# cd {country_code}#{product_id}#{city}_{s_data}_{e_data}_simu_v9.txt && mv *.gz /home/xiaoju/chenchunrong/project/BR/data/{country_code}_{s_data}_{e_data}_DATA/{country_code}#{product_id}#{city}_{s_data}_{e_data}_simu_v9.txt
# rm -rf /home/xiaoju/chenchunrong/project/BR/code/{country_code}#{product_id}#{city}_{s_data}_{e_data}_simu_v9.txt""".format(city = city,country_code = country_code,product_id = product_id,s_data=s_data,e_data=e_data))



# 函数

def get_bin_df(df, bin_cols, bin_nums, bin_cut_points={}, keep_digits=6):
    for index, x in enumerate(bin_cols):
        tmp_bin = x + '_bin'
        if x in bin_cut_points and len(bin_cut_points[x]) >= 3:
            cur_cut_points = np.array(bin_cut_points[x])
        else:
            cur_cut_points = np.nanpercentile(df[x], np.linspace(0, 100, bin_nums[index] + 1))

#             print("ori {}".format(x), cur_cut_points)
        cur_cut_points = np.floor(np.array(cur_cut_points) * (10 ** keep_digits)) / (10 ** keep_digits)
        # print("floor cur_cut_points", cur_cut_points)
        cur_cut_points[0] = cur_cut_points[0] - 10 ** (-1 * keep_digits)
        cur_cut_points[-1] = cur_cut_points[-1] + 10 ** (-1 * keep_digits)
#             print("digit {}".format(x), cur_cut_points)
        cur_cut_points = np.unique(cur_cut_points)
#             print("unique {}".format(x), cur_cut_points)
#
        bin_name_list = [x + '_' + '{:f}'.format(cur_cut_points[i]) + '_' + '{:f}'.format(cur_cut_points[i + 1]) for i
                         in range(len(cur_cut_points) - 1)]
#
        df[tmp_bin] = pd.cut(df[x], cur_cut_points, right=True, include_lowest=False, duplicates='drop',
                             labels=bin_name_list).astype(str)

    return df


def get_large_df(data_path, bj_start,bj_end,keep_cols):
    df = None
    data_iter = 0
    batch_size = int(1e6)
#         print('batch size is {}:'.format(batch_size))
    raw_df = pd.read_csv(data_path, sep='\t', chunksize=batch_size, low_memory=True,
                         error_bad_lines=False, compression='gzip')
    for chunk_df in raw_df:
#             print('=====  Iteration: {}  ====='.format(data_iter))
#             print('RAW input_df shape is: {}'.format(chunk_df.shape))
        chunk_df= chunk_df[keep_cols]
        chunk_df = chunk_df[chunk_df["bj_date"].between("2025-09-29", "2025-10-01")]
        if df is None:
            df = chunk_df
        else:
            df = pd.concat([df, chunk_df])
        data_iter = data_iter + 1

    return df


def get_ecr_opt(x):
    if x["pax_opt"] == 0:
        return x["pred_ecr_0"]
    else:
        return x["pred_ecr_" + str(x["pax_opt"])]


def judge_range(x):
    if x == -0.1:
        return "-0.1"
    elif x == -0.07:
        return "-0.07"
    elif x == -0.05:
        return "-0.05"
    elif x == 0:
        return "0"
    elif x == 0.05:
        return "0.05"
    elif x == 0.07:
        return "0.07"
    elif x == 0.1:
        return "0.1"
    else:
        return ""

# 配置

country_code = "BR"
product_id = "16"
city_id = city
feature_version = 'v9'
file_start = bj_start = s_data
file_end = bj_end = e_data

# 下面都不用动
start_dist_km = 4
# bj_start = "2023-12-01"55000199
minprice_sp_type = "0"
minprice_dsp_type = "1"

ecr_treat_list = [-0.15, -0.13, -0.1, -0.07, -0.05, 0, 0.05, 0.07, 0.1, 0.13, 0.15]
# ecr_treat_list = [-0.1, -0.07, -0.05, 0, 0.05, 0.07, 0.1]
cr_treat_list = [-0.1, -0.07, -0.05, 0, 0.05, 0.07, 0.1]
booking_percent = 0

base_path = "/home/xiaoju/chenchunrong/project/BR/"
data_dir = base_path+'/'+'data/'
mdl_out_dir = base_path+'/'+'model_out/'
cr_mdl_out_dir = base_path+'/'+'model_out/'
stg_out_dir = base_path+'/'+'stg_out/'
analysis_out_dir = base_path+'/'+'analysis_out/'

dir_path = '{}_{}_{}_DATA'.format(country_code, file_start, file_end)
data_dir = base_path+'/data/'+dir_path+'/'#+"_online"


data_path_prefix = data_dir
if feature_version == 'v9':
    data_path_suffix = "{}#{}#{}_{}_{}_simu_v9.txt".format(country_code,product_id, city_id, file_start, file_end)
elif feature_version == 'v8':
    data_path_suffix = "{}#{}#{}_{}_{}_simu.txt".format(country_code,product_id, city_id, file_start, file_end)

data_file_path = data_path_prefix + data_path_suffix

from astro_model import CheckException,getLoaderSingleInstInitConfig, initConfig, ModelType, RemoteSyncLoader, RemoteLoader, registerLoadModelFunc

# 模型加载


def error_handler(e: CheckException):
    """
    模型加载
    """
    # 打日志,上报metric等行为
    logging.warning("model get error %s", str(e))
    # True代表发生异常的时候抛出异常中止程序，False代表不抛出异常

    return False
model_path =  mdl_out_dir
# 方法2 获取单例方式
model_loader = getLoaderSingleInstInitConfig(
    project_code= "pricing",  # 项目code和上传模型时候填写的一致 必填
    onerror=error_handler,  # 发生报错的回调,默认为在测试环境打日志
    refresh_interval=2*60,  # 异步刷新间隔, 单位是s, 默认是2分钟, 仅在 RemoteLoader 中使用
    modelPath=model_path,  # 模型下载后存放的路径, 默认为当前文件夹 必填
    retries=1,  # 请求接口或下载模型时出现错误重试次数, 默认为1
    retry_interval=1,  # 请求接口失败或下载模型时出现错误重试间隔,单位是s，默认为1
    env="us",  # 环境表示，目前支持:sim, us. 默认为 "us" 必填
    model_types=[ModelType.CF,ModelType.XGB,ModelType.TF],  # 注册支持的模型列表 必填
    loadFeature=True # loadFeature 为True表示加载特征, 默认为false
)

def get_model_from_lego(name,mode):

    # cf模型
    if mode == "cf":
        # TODO: 加载cf模型
        model = UpliftRandomForestClassifier(control_name="0.0")
        model_loader.initModel(name, model_type=ModelType.CF)
        model, feature = model_loader.loadModelAndFeatures(name)
    # xgb模型
    else:
        # 初始化需要的模型 验证模型的类型和加载方式没有问题
        model_loader.initModel(name, model_type=ModelType.XGB)
        model, feature = model_loader.loadModelAndFeatures(name)
#     model_loader.run()

    return model,feature


cr_model_name = "odt_br_16_55000233_cr_20260501"

cr_cf_model_name = "odt_br_16_55000233_cr_cf_20260501"

tf_model_name = "odt_br_16_55000233_ecr_tf_20260501"

# 根据模型名字加载特征和模型


# 加载ecr模型
model_loader.initModel(tf_model_name, model_type=ModelType.TF)
tf_model,tf_features = model_loader.loadModelAndFeatures(tf_model_name)
print(tf_model_name)
#加载cr模型
cr_bst,cr_feature_cols = get_model_from_lego(cr_model_name,'cr')
cf_cr_mdl_file,cf_cr_feature_cols = get_model_from_lego(cr_cf_model_name,'cf')
print(cf_cr_mdl_file)

cf_cr_model = UpliftRandomForestClassifier(control_name="0.0", train_method="Sub")

ecr_fea_list = ecr_feature_cols = tf_features
cr_fea_list = cr_feature_cols

# hit_sp_explore==1 只取探索数据 发现没有
def get_large_df(data_path, bj_start,bj_end,keep_cols):
    df = None
    data_iter = 0
    batch_size = int(1e6)
#         print('batch size is {}:'.format(batch_size))
    raw_df = pd.read_csv(data_path, sep='\t', chunksize=batch_size, low_memory=True,
                         error_bad_lines=False, compression='gzip')
    for chunk_df in raw_df:
#         print('=====  Iteration: {}  ====='.format(data_iter))
#         print('RAW input_df shape is: {}'.format(chunk_df.shape))
        print(chunk_df.head())
        chunk_df= chunk_df[keep_cols]
        print(chunk_df.groupby('hit_sp_explore').size().sort_index())
        chunk_df = chunk_df[
            chunk_df["bj_date"].between(bj_start, bj_end)
        ]
#         chunk_df = chunk_df.sample(n=100000,random_state=10)
        if df is None:
            df = chunk_df
        else:
            df = pd.concat([df, chunk_df])
        data_iter = data_iter + 1

    return df

var_ecr_list = ['pred_ecr' + '_' + str(treat) for treat in ecr_treat_list]
var_cr_list = ['pred_cr' + '_' + str(treat) for treat in cr_treat_list]

compression_cols_tmp = ["is_call", "is_finish", "gmv", "online_driver_divide",
                        "driverpretotalfee", "driver_dynamic_times", "hit_sp_explore", "hit_driver_sp_explore",
                        "pre_total_fee", "dynamic_times", "bj_date", "road_dist_km", "pre_total_fee_dp",
                        "driverpretotalfee_dp", "sptimesforpassenger", "sptimesfordriver", "estimate_id",
                        "PriceTimes", "DriverPriceTimes", "city_id", "weekday", "hour", "min_price_hit",
                        "mininum_price", "dri_minimum_price",
                        "driver_min_price_hit", "eta", "fp_version"]
compression_cols = list(set(ecr_feature_cols).union(cr_feature_cols).union(tf_features).union(cf_cr_feature_cols).union(compression_cols_tmp))

# 加载数据

df = get_large_df(data_file_path, bj_start, bj_end, compression_cols)


df['road_dist_km_int'] = np.floor(df['road_dist_km'])



# if df.shape[0] > 8000000:
#     df = df.sample(n=8000000,random_state=10)



pax_fee_cut = np.nanpercentile(df[df["is_finish"] == 1]['pre_total_fee'], 99.5)
#     print('pax max cut is: {}'.format(pax_fee_cut))



# 数据处理

pax_fee_cut = np.nanpercentile(df[df["is_finish"] == 1]['pre_total_fee'], 99.5)
#     print('pax max cut is: {}'.format(pax_fee_cut))

df['is_sp_explore'] = ((df['hit_sp_explore'] == 'True') | (df['hit_sp_explore'] == True)).astype('int')
df["dri_fee"] = df["driverpretotalfee"] * df["driver_dynamic_times"]

df = df.loc[df['pre_total_fee'] < pax_fee_cut, :]  # 预估价过滤

# df[ecr_feature_cols] = df[ecr_feature_cols].fillna(0.0)
# df[cr_feature_cols] = df[cr_feature_cols].fillna(0.0)


df["PriceTimes_ini"] = df["PriceTimes"]
df["DriverPriceTimes_ini"] = df["DriverPriceTimes"]


df["DriverPriceTimes"] = 0
df["cr0"] = cr_bst.predict(xgb.DMatrix(df[cr_fea_list]))



df.hit_sp_explore.value_counts()

df['dp'] = df['dynamic_times']
df['dp_drv'] = df['driver_dynamic_times']
df['dist'] = df['road_dist_km']

print(df.columns.tolist())
df['bj_date'].min(), df['bj_date'].max(), df['bj_date'].nunique()
df.groupby('bj_date').size().sort_index()

bin_cut_points = {}
bin_cut_points["weekday"] = [i + 0.5 for i in range(-1, 7)]
bin_cut_points["hour"] = [i + 0.5 for i in range(-1, 24)]
df = get_bin_df(df, ["weekday", "dist",'hour'], [10, 10,24], bin_cut_points=bin_cut_points, keep_digits=6)

# 先乘客

## 各调价点预测

var_ecr_list = []
var_cr_list = []
var_grab_list = []

import tensorflow as tf
input_tensor = tf.convert_to_tensor(df[tf_features].values, dtype=tf.float32)
pred_res = tf_model(input=input_tensor)
final_result = pred_res['p_mu'].numpy()
for i, treat in enumerate([0, 0.05,-0.05,0.07,-0.07,0.1,-0.1,0.13,-0.13,0.15,-0.15]):
    var_ecr = 'pred_ecr' + '_' + str(treat)
#         print('st art estimation {}: '.format(var_ecr))
    df[var_ecr] = final_result[:, i]
    var_ecr_list.append(var_ecr)

cr_pred = cf_cr_model.predict_json(cf_cr_mdl_file, df[cf_cr_feature_cols])
for treat in cr_treat_list:
    var_cr = 'pred_cr' + '_' + str(treat) + "_stp2"
#         print('start estimation {}: '.format(var_cr))
    if treat == 0:
        df[var_cr] = df['cr0'].values
    else:
        df[var_cr] = cr_pred['delta_' + str(treat)] + df['cr0']
    var_cr_list.append(var_cr)

#     print('SIMU input_df shape is: {}'.format(df.shape))
df['is_sp_explore'] = ((df['hit_sp_explore'] == 'True') | (df['hit_sp_explore'] == True)).astype('int')

#     print('======  Finish Estimation  =====')


var_ecr_list = ['pred_ecr' + '_' + str(treat) for treat in ecr_treat_list]
var_cr_list = ['pred_cr' + '_' + str(treat) + "_stp2" for treat in cr_treat_list]

CR0 = np.sum(df['pred_ecr_0'] * df['cr0']) / np.sum(df['pred_ecr_0'])
print(CR0)

dp_list = list(np.nanpercentile(df["dynamic_times"],np.linspace(0,100,21)))
DP_TH = dp_list[-4]
dp_drv_list = list(np.nanpercentile(df["driver_dynamic_times"],np.linspace(0,100,21)))
DP_DRV_TH = dp_drv_list[-4]
DP_TH, DP_DRV_TH

df["dri_fee"] = df["driverpretotalfee"] * df["driver_dynamic_times"]

pax_base_price = 'pre_total_fee_dp'  # 没考虑税：map表
dri_base_price = 'dri_fee'

x0 = df[pax_base_price]

dri_income = df[dri_base_price]

var_ecr_list = []
var_call_list = []
# var_call_gmv_list = []
var_trip_list_stp1 = []
var_gmv_list_stp1 = []
var_cost_list_stp1 = []
var_gp_list_stp1 = []
var_tch_list_stp1 = []
var_cr_ecr_list_stp1 = []
var_real_cost_list_stp1 = []
var_real_gmv_list_stp1 = []
# metrics of all treatment
y_pred_cr = df["cr0"]
for treat in ecr_treat_list:
    x_delata = treat
    var_ecr = 'pred_ecr' + '_' + str(treat)
    var_call = 'call' + '_' + str(treat)
    var_trip = 'trip' + '_' + str(treat) + "_stp1"
    var_gmv = 'gmv' + '_' + str(treat) + "_stp1"
    var_cost = 'cost' + '_' + str(treat) + "_stp1"
    var_gp = 'gp' + '_' + str(treat) + "_stp1"
    var_tch = 'tch_' + str(treat) + "_stp1"
    var_cr_ecr = 'cr_ecr_'+str(treat)+ "_stp1"
    var_real_cost = 'real_cost_' + str(treat) + "_stp1"
    var_real_gmv = 'real_gmv_' + str(treat) + "_stp1"

    y_pred = df[var_ecr]
    df[var_call] = y_pred
    df[var_trip] = y_pred * y_pred_cr
    df[var_gmv] = x0 * (1 + x_delata) * y_pred * y_pred_cr * (1 + booking_percent)
    df[var_cost] = df[var_trip] * dri_income
    df[var_gp] = df[var_gmv] - df[var_cost]
    df[var_tch] = df["eta"] * y_pred * y_pred_cr
    df[var_cr_ecr] = df[var_ecr] * (df["cr0"] - CR0)
    df[var_real_cost] = df["dri_fee"] * (1 + treat)
    df[var_real_gmv] = df["pre_total_fee_dp"] * (1 + treat)


    var_ecr_list.append(var_ecr)
    var_call_list.append(var_call)
    var_trip_list_stp1.append(var_trip)
    var_gmv_list_stp1.append(var_gmv)
    var_cost_list_stp1.append(var_cost)
    var_gp_list_stp1.append(var_gp)
    var_tch_list_stp1.append(var_tch)
    var_cr_ecr_list_stp1.append(var_cr_ecr)
    var_real_cost_list_stp1.append(var_real_cost)
    var_real_gmv_list_stp1.append(var_real_gmv)

function = 'gmv'
constraint = ['trip','tr','cr']

# 剔除高里程对策略的影响
#     print("剔除40km以上的订单 {} 条".format(df[df.dist.astype(float) > 40].shape[0]))
df = df[df.dist <= 40]

# 2  读取线上lambda
# 2.1  获取冒泡数据并初步处理
import os
os.system('hdfs dfs -get hdfs://DClusterUS1/user/root_international_algo_dev/international_rider_strategy/hive/international_rider_strategy/samples_city_local/BR//suchengcheng/BR#55000233_2026-05-01_2026-06-07_simu_v9.txt')

import glob
csv_file_list = glob.glob(f"MX_driver_odt_{s_data}_{e_data}_55000233_bub_check_detail.csv/part-*.csv")
first_csv_file = csv_file_list[0] if csv_file_list else ""
path = first_csv_file

def get_large_df(data_path):
    df = None
    data_iter = 0
    batch_size = int(1e6)
#         print('batch size is {}:'.format(batch_size))
    raw_df = pd.read_csv(data_path, sep=',', chunksize=batch_size, low_memory=True,
                         error_bad_lines=False
#                          , compression='gzip'
                        )
    for chunk_df in raw_df:
#             print('=====  Iteration: {}  ====='.format(data_iter))
#             print('RAW input_df shape is: {}'.format(chunk_df.shape))
        chunk_df = chunk_df[['running_group', 'estimate_id', 'lambdaGMV',
       'lambdaGP', 'lambdaTrip', 'lambdaCr', 'lambdaCost', 'lambdaCostDri',
       'lambdaGMVDri', 'lambdaGPDri', 'lambdaGPWkdDri', 'lambdaTripDri',
       'lambdaCrDri', 'lambdaCall', 'structral_times','lambdaGPWkd',
       'driver_structral_times','sp_apollo_dpth','dri_dp_th','tr_cap_th','ecrList','crList','ODT_HIT','road_dist','current_stat_date','bj_times']]

        if df is None:
            df = chunk_df
        else:
            df = pd.concat([df, chunk_df])
        data_iter = data_iter + 1

    return df
group_df_= get_large_df(path)

group_df_ = group_df_.drop(columns=["ecrList", "crList"], errors="ignore")
group_df_.to_csv("labmda_bubtime.csv", index=False, encoding="utf-8-sig")

# 1. 自动筛选所有 lambda 开头的列（无需手动写列名）
lambda_cols = [col for col in group_df_.columns if col.startswith('lambda')]
print(f"检测到的lambda列：{lambda_cols}")

# 2. 按日期分组，统计【每天每个lambda列的唯一值数量】
# 唯一值 = 1 → 当天所有请求的lambda值完全相同
# 唯一值 > 1 → 当天有不同的lambda值
daily_lambda_unique = group_df_.groupby('current_stat_date')[lambda_cols].nunique()

# 3. 展示每日唯一值统计结果（核心结果）
print("\n===== 每日lambda列唯一值数量统计（1=全天相同，>1=有不同值）=====")
print(daily_lambda_unique)

# 4. 快速总结：哪些lambda列【所有日期都保持唯一值（全天相同）】
print("\n===== 全日期稳定性总结 =====")
for col in lambda_cols:
    # 判断该列所有天的唯一值都是1
    all_same = (daily_lambda_unique[col] == 1).all()
    if all_same:
        print(f"✅ {col}：**每天的值都完全相同**（全量数据无变化）")
    else:
        # 统计有多少天是不同的
        diff_days = (daily_lambda_unique[col] > 1).sum()
        print(f"⚠️ {col}：有 {diff_days} 天存在不同值，其余天相同")

# 你重点关注的3个lambda列
focus_cols = ['lambdaTrip', 'lambdaGP', 'lambdaCr']
# 日期列
date_col = 'current_stat_date'

# 循环统计每个列的每日取值+数量
for col in focus_cols:
    print(f"\n\n{'=' * 50}")
    print(f"📊 【{col}】每日取值与对应数量统计")
    print(f"{'=' * 50}")

    # 按日期分组，统计每个值的出现次数（高效内置方法）
    daily_value_count = group_df_.groupby(date_col)[col].value_counts().unstack(fill_value=0)

    # 展示结果（行=日期，列=lambda取值，值=对应出现的行数）
    print(daily_value_count)
'''

==================================================
📊 【lambdaTrip】每日取值与对应数量统计
==================================================
lambdaTrip         0.00010  0.01973  0.02137  0.02293  0.02425  0.36874  \
current_stat_date                                                         
2026-05-01               0        0        0        0        0        0   
2026-05-02               0        0        0        0        0        0   
2026-05-03               0        0        0        0        0        0   
2026-05-04               0        0        0        0        0        0   
2026-05-05               0        0        0        0        0        0   
2026-05-06               0        0        0        0        0        0   
2026-05-07               0        0        0        0        0        0   
2026-05-08               0        0        0        0        0     1248   
2026-05-09               0        0        0        0        0        0   
2026-05-10               0        0        0        0        0        0   
2026-05-11               0        0        0        0        0        0   
2026-05-12               0        0        0        0        0        0   
2026-05-13           72703        0        0        0        0        0   
2026-05-14           88785        0        0        0        0        0   
2026-05-15           94629        0        0        0        0        0   
2026-05-16           64486        0        0        0        0        0   
2026-05-17           46592        0        0        0        0        0   
2026-05-18           71072        0        0        0        0        0   
2026-05-19           55786        0        0        0        0        0   
2026-05-20           57289        0        0        0        0        0   
2026-05-21           57614        0        0        0        0        0   
2026-05-22           64060        0        0        0        0        0   
2026-05-23           56435        0        0        0        0        0   
2026-05-24           46731        0        0        0        0        0   
2026-05-25           56106        0        0        0        0        0   
2026-05-26           68812        0        0        0        0        0   
2026-05-27           56711        0        0        0        0        0   
2026-05-28           17433        0    19252    19557        0        0   
2026-05-29               0        0    33259    34158        0        0   
2026-05-30               0        0    32066    32929        0        0   
2026-05-31               0    18529     4904     5113    18738        0   
2026-06-01           55261     1605        0        0     1410        0   
2026-06-02           56244        0        0        0        0        0   
2026-06-03           64622        0        0        0        0        0   
2026-06-04           54528        0        0        0        0        0   
2026-06-05           69517        0        0        0        0        0   
2026-06-06           76352        0        0        0        0        0   
2026-06-07           22528        0        0        0        0        0   

lambdaTrip         0.42622  0.43074  0.46991  0.47989  ...  1.14920  1.26915  \
current_stat_date                                      ...                     
2026-05-01               0        0        0        0  ...        0        0   
2026-05-02               0        0        0        0  ...        0        0   
2026-05-03               0        0        0        0  ...        0        0   
2026-05-04             163        0        0        0  ...        0    29347   
2026-05-05           31192        0        0        0  ...        0        0   
2026-05-06             369       62    32006        0  ...        0        0   
2026-05-07               0    36791        0        0  ...        0        0   
2026-05-08               0      351        0   142468  ...        0        0   
2026-05-09               0        0        0   152097  ...        0        0   
2026-05-10               0        0        0    10674  ...        0        0   
2026-05-11               0        0        0        0  ...        0        0   
2026-05-12               0        0        0        0  ...    91787        0   
2026-05-13               0        0        0        0  ...    19255        0   
2026-05-14               0        0        0        0  ...        0        0   
2026-05-15               0        0        0        0  ...        0        0   
2026-05-16               0        0        0        0  ...        0        0   
2026-05-17               0        0        0        0  ...        0        0   
2026-05-18               0        0        0        0  ...        0        0   
2026-05-19               0        0        0        0  ...        0        0   
2026-05-20               0        0        0        0  ...        0        0   
2026-05-21               0        0        0        0  ...        0        0   
2026-05-22               0        0        0        0  ...        0        0   
2026-05-23               0        0        0        0  ...        0        0   
2026-05-24               0        0        0        0  ...        0        0   
2026-05-25               0        0        0        0  ...        0        0   
2026-05-26               0        0        0        0  ...        0        0   
2026-05-27               0        0        0        0  ...        0        0   
2026-05-28               0        0        0        0  ...        0        0   
2026-05-29               0        0        0        0  ...        0        0   
2026-05-30               0        0        0        0  ...        0        0   
2026-05-31               0        0        0        0  ...        0        0   
2026-06-01               0        0        0        0  ...        0        0   
2026-06-02               0        0        0        0  ...        0        0   
2026-06-03               0        0        0        0  ...        0        0   
2026-06-04               0        0        0        0  ...        0        0   
2026-06-05               0        0        0        0  ...        0        0   
2026-06-06               0        0        0        0  ...        0        0   
2026-06-07               0        0        0        0  ...        0        0   

lambdaTrip         1.33905  1.50414  1.62340  1.62703  1.76548  1.93087  \
current_stat_date                                                         
2026-05-01               0    24411        0        0        0        0   
2026-05-02               0      889        0    25236      672    25036   
2026-05-03               0        0        0        0    19601        0   
2026-05-04           29845        0        0        0      158        0   
2026-05-05               0        0        0        0        0        0   
2026-05-06               0        0        0        0        0        0   
2026-05-07               0        0        0        0        0        0   
2026-05-08               0        0        0        0        0        0   
2026-05-09               0        0        0        0        0        0   
2026-05-10               0        0    90221        0        0        0   
2026-05-11               0        0   102592        0        0        0   
2026-05-12               0        0     2039        0        0        0   
2026-05-13               0        0        0        0        0        0   
2026-05-14               0        0        0        0        0        0   
2026-05-15               0        0        0        0        0        0   
2026-05-16               0        0        0        0        0        0   
2026-05-17               0        0        0        0        0        0   
2026-05-18               0        0        0        0        0        0   
2026-05-19               0        0        0        0        0        0   
2026-05-20               0        0        0        0        0        0   
2026-05-21               0        0        0        0        0        0   
2026-05-22               0        0        0        0        0        0   
2026-05-23               0        0        0        0        0        0   
2026-05-24               0        0        0        0        0        0   
2026-05-25               0        0        0        0        0        0   
2026-05-26               0        0        0        0        0        0   
2026-05-27               0        0        0        0        0        0   
2026-05-28               0        0        0        0        0        0   
2026-05-29               0        0        0        0        0        0   
2026-05-30               0        0        0        0        0        0   
2026-05-31               0        0        0        0        0        0   
2026-06-01               0        0        0        0        0        0   
2026-06-02               0        0        0        0        0        0   
2026-06-03               0        0        0        0        0        0   
2026-06-04               0        0        0        0        0        0   
2026-06-05               0        0        0        0        0        0   
2026-06-06               0        0        0        0        0        0   
2026-06-07               0        0        0        0        0        0   

lambdaTrip         1.93765  2.08937  
current_stat_date                    
2026-05-01               0    24436  
2026-05-02             713      892  
2026-05-03           19561        0  
2026-05-04             208        0  
2026-05-05               0        0  
2026-05-06               0        0  
2026-05-07               0        0  
2026-05-08               0        0  
2026-05-09               0        0  
2026-05-10               0        0  
2026-05-11               0        0  
2026-05-12               0        0  
2026-05-13               0        0  
2026-05-14               0        0  
2026-05-15               0        0  
2026-05-16               0        0  
2026-05-17               0        0  
2026-05-18               0        0  
2026-05-19               0        0  
2026-05-20               0        0  
2026-05-21               0        0  
2026-05-22               0        0  
2026-05-23               0        0  
2026-05-24               0        0  
2026-05-25               0        0  
2026-05-26               0        0  
2026-05-27               0        0  
2026-05-28               0        0  
2026-05-29               0        0  
2026-05-30               0        0  
2026-05-31               0        0  
2026-06-01               0        0  
2026-06-02               0        0  
2026-06-03               0        0  
2026-06-04               0        0  
2026-06-05               0        0  
2026-06-06               0        0  
2026-06-07               0        0  

[38 rows x 23 columns]


==================================================
📊 【lambdaGP】每日取值与对应数量统计
==================================================
lambdaGP           0.0001
current_stat_date        
2026-05-01          48847
2026-05-02          53438
2026-05-03          39162
2026-05-04          59848
2026-05-05          61625
2026-05-06          63852
2026-05-07          74726
2026-05-08         144462
2026-05-09         152097
2026-05-10         100895
2026-05-11         102592
2026-05-12          93826
2026-05-13          91958
2026-05-14          88785
2026-05-15          94629
2026-05-16          64486
2026-05-17          46592
2026-05-18          71072
2026-05-19          55786
2026-05-20          57289
2026-05-21          57614
2026-05-22          64060
2026-05-23          56435
2026-05-24          46731
2026-05-25          56106
2026-05-26          68812
2026-05-27          56711
2026-05-28          56242
2026-05-29          67417
2026-05-30          64995
2026-05-31          47284
2026-06-01          58276
2026-06-02          56244
2026-06-03          64622
2026-06-04          54528
2026-06-05          69517
2026-06-06          76352
2026-06-07          22528


==================================================
📊 【lambdaCr】每日取值与对应数量统计
==================================================
lambdaCr           0.00010  0.07794  1.09888  1.33887  1.37716  1.43413  \
current_stat_date                                                         
2026-05-01           24411    24436        0        0        0        0   
2026-05-02           26797      892    25036        0        0        0   
2026-05-03           19601        0        0        0        0        0   
2026-05-04           29668        0        0      127        0        0   
2026-05-05           31192        0        0    30433        0        0   
2026-05-06           32437        0        0      283    31082       50   
2026-05-07           36791        0        0        0        0    37935   
2026-05-08            1599        0        0        0        0      395   
2026-05-09               0        0        0        0        0        0   
2026-05-10               0        0        0        0        0        0   
2026-05-11               0        0        0        0        0        0   
2026-05-12               0        0        0        0        0        0   
2026-05-13               0        0        0        0        0        0   
2026-05-14               0        0        0        0        0        0   
2026-05-15               0        0        0        0        0        0   
2026-05-16               0        0        0        0        0        0   
2026-05-17               0        0        0        0        0        0   
2026-05-18               0        0        0        0        0        0   
2026-05-19               0        0        0        0        0        0   
2026-05-20               0        0        0        0        0        0   
2026-05-21               0        0        0        0        0        0   
2026-05-22               0        0        0        0        0        0   
2026-05-23               0        0        0        0        0        0   
2026-05-24               0        0        0        0        0        0   
2026-05-25               0        0        0        0        0        0   
2026-05-26               0        0        0        0        0        0   
2026-05-27               0        0        0        0        0        0   
2026-05-28               0        0        0        0        0        0   
2026-05-29               0        0        0        0        0        0   
2026-05-30               0        0        0        0        0        0   
2026-05-31               0        0        0        0        0        0   
2026-06-01               0        0        0        0        0        0   
2026-06-02               0        0        0        0        0        0   
2026-06-03               0        0        0        0        0        0   
2026-06-04               0        0        0        0        0        0   
2026-06-05               0        0        0        0        0        0   
2026-06-06               0        0        0        0        0        0   
2026-06-07               0        0        0        0        0        0   

lambdaCr           1.44661  1.50952  1.53708  1.90395  ...  3.79971  3.98327  \
current_stat_date                                      ...                     
2026-05-01               0        0        0        0  ...        0        0   
2026-05-02             713        0        0        0  ...        0        0   
2026-05-03           19561        0        0        0  ...        0        0   
2026-05-04             208    29845        0        0  ...        0        0   
2026-05-05               0        0        0        0  ...        0        0   
2026-05-06               0        0        0        0  ...        0        0   
2026-05-07               0        0        0        0  ...        0        0   
2026-05-08               0        0   142468        0  ...        0        0   
2026-05-09               0        0   152097        0  ...        0        0   
2026-05-10               0        0    10674    90221  ...        0        0   
2026-05-11               0        0        0   102592  ...        0        0   
2026-05-12               0        0        0     2039  ...        0    91787   
2026-05-13               0        0        0        0  ...        0    19255   
2026-05-14               0        0        0        0  ...        0        0   
2026-05-15               0        0        0        0  ...        0        0   
2026-05-16               0        0        0        0  ...        0        0   
2026-05-17               0        0        0        0  ...        0        0   
2026-05-18               0        0        0        0  ...        0        0   
2026-05-19               0        0        0        0  ...        0        0   
2026-05-20               0        0        0        0  ...    21497        0   
2026-05-21               0        0        0        0  ...     6146        0   
2026-05-22               0        0        0        0  ...        0        0   
2026-05-23               0        0        0        0  ...        0        0   
2026-05-24               0        0        0        0  ...        0        0   
2026-05-25               0        0        0        0  ...        0        0   
2026-05-26               0        0        0        0  ...        0        0   
2026-05-27               0        0        0        0  ...        0        0   
2026-05-28               0        0        0        0  ...        0        0   
2026-05-29               0        0        0        0  ...        0        0   
2026-05-30               0        0        0        0  ...        0        0   
2026-05-31               0        0        0        0  ...        0        0   
2026-06-01               0        0        0        0  ...        0        0   
2026-06-02               0        0        0        0  ...        0        0   
2026-06-03               0        0        0        0  ...        0        0   
2026-06-04               0        0        0        0  ...        0        0   
2026-06-05               0        0        0        0  ...        0        0   
2026-06-06               0        0        0        0  ...        0        0   
2026-06-07               0        0        0        0  ...        0        0   

lambdaCr           4.60422  4.61868  4.83553  5.86862  5.88418  6.11492  \
current_stat_date                                                         
2026-05-01               0        0        0        0        0        0   
2026-05-02               0        0        0        0        0        0   
2026-05-03               0        0        0        0        0        0   
2026-05-04               0        0        0        0        0        0   
2026-05-05               0        0        0        0        0        0   
2026-05-06               0        0        0        0        0        0   
2026-05-07               0        0        0        0        0        0   
2026-05-08               0        0        0        0        0        0   
2026-05-09               0        0        0        0        0        0   
2026-05-10               0        0        0        0        0        0   
2026-05-11               0        0        0        0        0        0   
2026-05-12               0        0        0        0        0        0   
2026-05-13               0        0    72703        0        0        0   
2026-05-14               0        0    88785        0        0        0   
2026-05-15               0        0     5252        0        0    43578   
2026-05-16               0        0        0    30699    30703     1535   
2026-05-17               0        0        0    23585    23007        0   
2026-05-18               0        0        0    35415    35657        0   
2026-05-19           20348    20723        0     7203     7512        0   
2026-05-20            6646     7216        0        0        0        0   
2026-05-21               0        0        0        0        0        0   
2026-05-22               0        0        0        0        0        0   
2026-05-23               0        0        0        0        0        0   
2026-05-24               0        0        0        0        0        0   
2026-05-25               0        0        0        0        0        0   
2026-05-26               0        0        0        0        0        0   
2026-05-27               0        0        0        0        0        0   
2026-05-28               0        0        0        0        0        0   
2026-05-29               0        0        0        0        0        0   
2026-05-30               0        0        0        0        0        0   
2026-05-31               0        0        0        0        0        0   
2026-06-01               0        0        0        0        0        0   
2026-06-02               0        0        0        0        0        0   
2026-06-03               0        0        0        0        0        0   
2026-06-04               0        0        0        0        0        0   
2026-06-05               0        0        0        0        0        0   
2026-06-06               0        0        0        0        0        0   
2026-06-07               0        0        0        0        0        0   

lambdaCr           6.11548  6.11926  
current_stat_date                    
2026-05-01               0        0  
2026-05-02               0        0  
2026-05-03               0        0  
2026-05-04               0        0  
2026-05-05               0        0  
2026-05-06               0        0  
2026-05-07               0        0  
2026-05-08               0        0  
2026-05-09               0        0  
2026-05-10               0        0  
2026-05-11               0        0  
2026-05-12               0        0  
2026-05-13               0        0  
2026-05-14               0        0  
2026-05-15           44053     1746  
2026-05-16            1549        0  
2026-05-17               0        0  
2026-05-18               0        0  
2026-05-19               0        0  
2026-05-20               0        0  
2026-05-21               0        0  
2026-05-22               0        0  
2026-05-23               0        0  
2026-05-24               0        0  
2026-05-25               0        0  
2026-05-26               0        0  
2026-05-27               0        0  
2026-05-28               0        0  
2026-05-29               0        0  
2026-05-30               0        0  
2026-05-31               0        0  
2026-06-01               0        0  
2026-06-02               0        0  
2026-06-03               0        0  
2026-06-04               0        0  
2026-06-05               0        0  
2026-06-06               0        0  
2026-06-07               0        0  

[38 rows x 41 columns]
'''

lambda_value = {"lambda_gmv": "lambdaGMV",
                "lambda_trip": "lambdaTrip",
                "lambda_gp": "lambdaGP",
                "lambda_cost": "lambdaCost",
                "lambda_tr": "lambdaGP",
                "lambda_cr": "lambdaCr",
                "lambda_gmv_dri": "lambdaGMVDri",
                "lambda_trip_dri": "lambdaTripDri",
                "lambda_gp_dri": "lambdaGPDri",
                "lambda_tr_dri": "lambdaGPDri",
                "lambda_cost_dri": "lambdaCostDri",
                "lambda_cr_dri": "lambdaCrDri",
                "lambda_gp_weekend": "lambdaGPWkd",
                "lambda_tr_weekend": "lambdaGPWkd",
                "lambda_gp_weekend_dri": "lambdaGPWkdDri",
                "lambda_tr_weekend_dri": "lambdaGPWkdDri",
                "lambda_call": "lambdaCall"}


price_times = [0,0.05,-0.05,0.07,-0.07,0.1,-0.1,0.13,-0.13,0.15,-0.15]
driver_price_times = [0,0.05,-0.05,0.07,-0.07,0.1,-0.1]

# 2.2  和sp特征数据合并
df_right = group_df_[group_df_['estimate_id'].isin(df['estimate_id'])]
df_final = df.merge(df_right, left_on = 'estimate_id',right_on = 'estimate_id')
del df
del df_right

df_final.to_csv("bub_sample.csv", index=False, encoding="utf-8-sig")
df_final = pd.read_csv("bub_sample.csv", encoding="utf-8-sig")
df_final.groupby('bj_date').size().sort_index()
df_final.head(1).to_csv("bub_sample_1_exmple.csv", index=False, encoding="utf-8-sig")

# 1. 自动筛选所有 lambda 开头的列（无需手动写列名）
lambda_cols = [col for col in df_final.columns if col.startswith('lambda')]
print(f"检测到的lambda列：{lambda_cols}")

# 2. 按日期分组，统计【每天每个lambda列的唯一值数量】
# 唯一值 = 1 → 当天所有请求的lambda值完全相同
# 唯一值 > 1 → 当天有不同的lambda值
daily_lambda_unique = df_final.groupby('bj_date')[lambda_cols].nunique()

# 3. 展示每日唯一值统计结果（核心结果）
print("\n===== 每日lambda列唯一值数量统计（1=全天相同，>1=有不同值）=====")
print(daily_lambda_unique)

# 4. 快速总结：哪些lambda列【所有日期都保持唯一值（全天相同）】
print("\n===== 全日期稳定性总结 =====")
for col in lambda_cols:
    # 判断该列所有天的唯一值都是1
    all_same = (daily_lambda_unique[col] == 1).all()
    if all_same:
        print(f"✅ {col}：**每天的值都完全相同**（全量数据无变化）")
    else:
        # 统计有多少天是不同的
        diff_days = (daily_lambda_unique[col] > 1).sum()
        print(f"⚠️ {col}：有 {diff_days} 天存在不同值，其余天相同")


'''
检测到的lambda列：['lambdaGMV', 'lambdaGP', 'lambdaTrip', 'lambdaCr', 'lambdaCost', 'lambdaCostDri', 'lambdaGMVDri', 'lambdaGPDri', 'lambdaGPWkdDri', 'lambdaTripDri', 'lambdaCrDri', 'lambdaCall', 'lambdaGPWkd']

===== 每日lambda列唯一值数量统计（1=全天相同，>1=有不同值）=====
            lambdaGMV  lambdaGP  lambdaTrip  lambdaCr  lambdaCost  \
bj_date                                                             
2026-05-31          2         2           1         2           1   
2026-06-01          2         2           1         2           1   
2026-06-02          2         2           1         2           1   
2026-06-03          2         2           1         2           1   
2026-06-04          2         1           1         1           1   
2026-06-05          2         1           1         1           1   
2026-06-06          2         1           1         1           1   

            lambdaCostDri  lambdaGMVDri  lambdaGPDri  lambdaGPWkdDri  \
bj_date                                                                
2026-05-31              1             1            2               1   
2026-06-01              1             1            2               2   
2026-06-02              1             1            2               1   
2026-06-03              1             1            2               1   
2026-06-04              1             1            1               1   
2026-06-05              1             1            1               1   
2026-06-06              1             1            1               2   

            lambdaTripDri  lambdaCrDri  lambdaCall  lambdaGPWkd  
bj_date                                                          
2026-05-31              1            1           0            0  
2026-06-01              1            1           0            0  
2026-06-02              1            1           0            0  
2026-06-03              1            1           0            0  
2026-06-04              1            1           0            0  
2026-06-05              1            1           0            0  
2026-06-06              1            1           0            0  

===== 全日期稳定性总结 =====
⚠️ lambdaGMV：有 7 天存在不同值，其余天相同
⚠️ lambdaGP：有 4 天存在不同值，其余天相同
✅ lambdaTrip：**每天的值都完全相同**（全量数据无变化）
⚠️ lambdaCr：有 4 天存在不同值，其余天相同
✅ lambdaCost：**每天的值都完全相同**（全量数据无变化）
✅ lambdaCostDri：**每天的值都完全相同**（全量数据无变化）
✅ lambdaGMVDri：**每天的值都完全相同**（全量数据无变化）
⚠️ lambdaGPDri：有 4 天存在不同值，其余天相同
⚠️ lambdaGPWkdDri：有 2 天存在不同值，其余天相同
✅ lambdaTripDri：**每天的值都完全相同**（全量数据无变化）
✅ lambdaCrDri：**每天的值都完全相同**（全量数据无变化）
⚠️ lambdaCall：有 0 天存在不同值，其余天相同
⚠️ lambdaGPWkd：有 0 天存在不同值，其余天相同
'''

# 你重点关注的3个lambda列
focus_cols = ['lambdaTrip', 'lambdaGP', 'lambdaCr']
# 日期列
date_col = 'bj_date'

# 循环统计每个列的每日取值+数量
for col in focus_cols:
    print(f"\n\n{'=' * 50}")
    print(f"📊 【{col}】每日取值与对应数量统计")
    print(f"{'=' * 50}")

    # 按日期分组，统计每个值的出现次数（高效内置方法）
    daily_value_count = df_final.groupby(date_col)[col].value_counts().unstack(fill_value=0)

    # 展示结果（行=日期，列=lambda取值，值=对应出现的行数）
    print(daily_value_count)

# 1. 自动筛选所有 lambda 开头的列（无需手动写列名）
lambda_cols = [col for col in df_final.columns if col.startswith('lambda')]
print(f"检测到的lambda列：{lambda_cols}")

# 2. 按日期分组，统计【每天每个lambda列的唯一值数量】
# 唯一值 = 1 → 当天所有请求的lambda值完全相同
# 唯一值 > 1 → 当天有不同的lambda值
daily_lambda_unique = df_final.groupby('current_stat_date')[lambda_cols].nunique()

# 3. 展示每日唯一值统计结果（核心结果）
print("\n===== 每日lambda列唯一值数量统计（1=全天相同，>1=有不同值）=====")
print(daily_lambda_unique)

# 4. 快速总结：哪些lambda列【所有日期都保持唯一值（全天相同）】
print("\n===== 全日期稳定性总结 =====")
for col in lambda_cols:
    # 判断该列所有天的唯一值都是1
    all_same = (daily_lambda_unique[col] == 1).all()
    if all_same:
        print(f"✅ {col}：**每天的值都完全相同**（全量数据无变化）")
    else:
        # 统计有多少天是不同的
        diff_days = (daily_lambda_unique[col] > 1).sum()
        print(f"⚠️ {col}：有 {diff_days} 天存在不同值，其余天相同")

# 你重点关注的3个lambda列
focus_cols = ['lambdaTrip', 'lambdaGP', 'lambdaCr']
# 日期列
date_col = 'current_stat_date'

# 循环统计每个列的每日取值+数量
for col in focus_cols:
    print(f"\n\n{'=' * 50}")
    print(f"📊 【{col}】每日取值与对应数量统计")
    print(f"{'=' * 50}")

    # 按日期分组，统计每个值的出现次数（高效内置方法）
    daily_value_count = df_final.groupby(date_col)[col].value_counts().unstack(fill_value=0)

    # 展示结果（行=日期，列=lambda取值，值=对应出现的行数）
    print(daily_value_count)

'''
==================================================
📊 【lambdaTrip】每日取值与对应数量统计
==================================================
lambdaTrip         0.0001
current_stat_date        
2026-05-31          24280
2026-06-01          25119
2026-06-02          23528
2026-06-03          29952
2026-06-04          26278
2026-06-05          30872
2026-06-06          12594


==================================================
📊 【lambdaGP】每日取值与对应数量统计
==================================================
lambdaGP           0.28732  0.29049  0.34722  0.35018  0.40501
current_stat_date                                             
2026-05-31           24210       26        0       44        0
2026-06-01               0    25119        0        0        0
2026-06-02               0       17    23237        0      274
2026-06-03               0        0        0        0    29952
2026-06-04               0        0        0        0    26278
2026-06-05               0        0        0        0    30872
2026-06-06               0        0        0        0    12594


==================================================
📊 【lambdaCr】每日取值与对应数量统计
==================================================
lambdaCr           21.03564  21.12739  22.32187  22.49955  23.05344
current_stat_date                                                  
2026-05-31            24210        26        44         0         0
2026-06-01                0     25119         0         0         0
2026-06-02                0        17         0     23237       274
2026-06-03                0         0         0         0     29952
2026-06-04                0         0         0         0     26278
2026-06-05                0         0         0         0     30872
2026-06-06                0         0         0         0     12594
'''

# 3  构建模型
# 3.1  每天固定lambda
df_ = df_final.query('hit_sp_explore == False and ODT_HIT == 1 and road_dist < 40')
date_list = df_.bj_date.drop_duplicates().sort_values().tolist()
# print('======  Start Metrics Cal  =====')

df_["dri_fee"] = df_["driverpretotalfee"] * df_["driver_dynamic_times"]

pax_base_price = 'pre_total_fee_dp'  # 没考虑税：map表
dri_base_price = 'dri_fee'

x0 = df_[pax_base_price]

dri_income = df_[dri_base_price]

var_ecr_list = []
var_call_list = []
# var_call_gmv_list = []
var_trip_list_stp1 = []
var_gmv_list_stp1 = []
var_cost_list_stp1 = []
var_gp_list_stp1 = []
var_tch_list_stp1 = []
var_cr_ecr_list_stp1 = []
var_real_cost_list_stp1 = []
var_real_gmv_list_stp1 = []
# metrics of all treatment
y_pred_cr = df_["cr0"]
for treat in ecr_treat_list:
    x_delata = treat
    var_ecr = 'pred_ecr' + '_' + str(treat)
    var_call = 'call' + '_' + str(treat)
    var_trip = 'trip' + '_' + str(treat) + "_stp1"
    var_gmv = 'gmv' + '_' + str(treat) + "_stp1"
    var_cost = 'cost' + '_' + str(treat) + "_stp1"
    var_gp = 'gp' + '_' + str(treat) + "_stp1"
    var_tch = 'tch_' + str(treat) + "_stp1"
    var_cr_ecr = 'cr_ecr_' + str(treat) + "_stp1"
    var_real_cost = 'real_cost_' + str(treat) + "_stp1"
    var_real_gmv = 'real_gmv_' + str(treat) + "_stp1"

    y_pred = df_[var_ecr]
    df_[var_call] = y_pred
    df_[var_trip] = y_pred * y_pred_cr
    df_[var_gmv] = x0 * (1 + x_delata) * y_pred * y_pred_cr * (1 + booking_percent)
    df_[var_cost] = df_[var_trip] * dri_income
    df_[var_gp] = df_[var_gmv] - df_[var_cost]
    df_[var_tch] = df_["eta"] * y_pred * y_pred_cr
    df_[var_cr_ecr] = df_[var_ecr] * (df_["cr0"] - CR0)
    df_[var_real_cost] = df_["dri_fee"] * (1 + treat)
    df_[var_real_gmv] = df_["pre_total_fee_dp"] * (1 + treat)

    var_ecr_list.append(var_ecr)
    var_call_list.append(var_call)
    var_trip_list_stp1.append(var_trip)
    var_gmv_list_stp1.append(var_gmv)
    var_cost_list_stp1.append(var_cost)
    var_gp_list_stp1.append(var_gp)
    var_tch_list_stp1.append(var_tch)
    var_cr_ecr_list_stp1.append(var_cr_ecr)
    var_real_cost_list_stp1.append(var_real_cost)
    var_real_gmv_list_stp1.append(var_real_gmv)

# date_list = []
# date_list.append(e_data)

total_call_opt = 0
total_trip_opt = 0
total_gmv_opt = 0
total_gp_opt = 0
total_cost_opt = 0
total_gp_opt_weekends = 0

total_call_base = 0
total_trip_base = 0
total_gmv_base = 0
total_gp_base = 0
total_cost_base = 0
total_gmv_base_weekends = 0
total_gp_base_weekends = 0

total_sp = 0
total_dsp = 0
total_num = 0
df_cur_solution = pd.DataFrame()

for cur_date in date_list:
    print("------------------ {} -------------------".format(cur_date))

    #     df_cur = df_[df_["current_stat_date"] == cur_date]
    #     df_cur=df_[(df_["bj_date"]==cur_date)&(df_["running_group"]=="group_1_tf_sep")]
    df_cur = df_[(df_["current_stat_date"] == cur_date)]
    #     df_cur=df_[(df_["bj_date"]==cur_date)&(df_["running_group"]=="group_2_tf")]
    df_cur.reset_index(drop=True, inplace=True)

    function = 'gmv'
    constraint = ['trip', 'tr', 'cr']
    Trip_Mat_val_stp1 = df_cur[var_trip_list_stp1].values
    Call_Mat_val_stp1 = df_cur[var_call_list].values
    GMV_Mat_val_stp1 = df_cur[var_gmv_list_stp1].values
    Cost_Mat_val_stp1 = df_cur[var_cost_list_stp1].values
    GP_Mat_val_stp1 = GMV_Mat_val_stp1 - Cost_Mat_val_stp1
    Tch_Mat_val_stp1 = df_cur[var_tch_list_stp1].values
    CR_ECR_Mat_val_stp1 = df_cur[var_cr_ecr_list_stp1].values
    Real_GMV_Mat_val = df_cur[var_real_gmv_list_stp1].values

    Trip_base_stp1 = np.sum(df_cur['trip_0_stp1'])
    Call_base_stp1 = np.sum(df_cur["call_0"])
    GMV_base_stp1 = np.sum(df_cur['gmv_0_stp1'])
    Cost_base_stp1 = np.sum(df_cur['cost_0_stp1'])
    GP_base_stp1 = GMV_base_stp1 - Cost_base_stp1
    Tch_base_stp1 = np.sum(df_cur['tch_0_stp1'])
    CR_base_stp1 = Trip_base_stp1 / Call_base_stp1

    GMV_Mat_val_reg_stp1 = df_cur[var_gmv_list_stp1].values
    Cost_Mat_val_reg_stp1 = df_cur[var_cost_list_stp1].values
    Trip_Mat_val_reg_stp1 = df_cur[var_trip_list_stp1].values
    GP_Mat_val_reg_stp1 = GMV_Mat_val_stp1 - Cost_Mat_val_stp1

    # 核心约束1：禁止高价订单使用降价策略 要是高价订单选降价策略会导致gmv大幅下降
    GMV_Mat_val_reg_stp1[(df_cur['dp'] > DP_TH + 1e-6), 0:sum(i < 0 for i in ecr_treat_list)] = -1 * GMV_base_stp1
    #     print('乘客侧dpth 筛选调价系数 {} 条'.format(sum(df_cur['dp'] > DP_TH + 1e-6)))

    if minprice_sp_type in ["0", "", None, "None"]:
        # 原始逻辑 根据PreTotalFee<MiniNumPrice及sp<0; 满足则将dsp置为0；dsp_new = 0
        condition = (df_cur['min_price_hit'].astype(str) == '1')
        rows = np.where(condition)
        #         print('方案0 乘客侧最低消费 筛选调价系数 {} 条'.format(sum(condition)))
        cols = list(range(0, sum(i < 0 for i in ecr_treat_list)))  # 小于零0的那列
        specified_col_index = sum(i < 0 for i in ecr_treat_list)  # 0的那一列
        for col in cols:
            # 避免argmax 在同等条件下不选0，选强制选0，避免影响sp系数
            GMV_Mat_val_reg_stp1[rows, col] = -1 * GMV_base_stp1
            GP_Mat_val_reg_stp1[rows, col] = -1 * GP_base_stp1
            Trip_Mat_val_reg_stp1[rows, col] = -1 * Trip_base_stp1
            Cost_Mat_val_reg_stp1[rows, col] = -1 * Cost_base_stp1

            GMV_Mat_val_stp1[rows, col] = GMV_Mat_val_stp1[rows, specified_col_index]
            Cost_Mat_val_stp1[rows, col] = Cost_Mat_val_stp1[rows, specified_col_index]
            Trip_Mat_val_stp1[rows, col] = Trip_Mat_val_stp1[rows, specified_col_index]
            GP_Mat_val_stp1[rows, col] = GP_Mat_val_stp1[rows, specified_col_index]
            CR_ECR_Mat_val_stp1[rows, col] = CR_ECR_Mat_val_stp1[rows, specified_col_index]
            GP_Mat_val_stp1 = GMV_Mat_val_stp1 - Cost_Mat_val_stp1

    else:

        # 判断PreTotalFee*dp*(1+sp)< MiniNumPrice及sp<0; 满足则将sp置为0；sp_new = 0
        miniprice_flag = Real_GMV_Mat_val < np.repeat(df_cur['mininum_price'].values.reshape(1, -1),
                                                      len(ecr_treat_list), axis=0).T

        replacement = [True] * (len(ecr_treat_list) // 2) + [False] + [True] * (len(ecr_treat_list) // 2)
        rows_to_replace = np.all(miniprice_flag, axis=1)  # 不满足采用sp0
        miniprice_flag[rows_to_replace] = replacement
        column_sum = np.sum(miniprice_flag, axis=0)
        #         print('方案1 乘客侧最低消费 筛选调价系数{} {} 条'.format(ecr_treat_list, column_sum))
        specified_col_index = sum(i < 0 for i in ecr_treat_list)  # 0的那一列

        GMV_Mat_val_reg_stp1 = np.where(miniprice_flag, np.concatenate((np.repeat(
            np.array([-2000 for i in GMV_Mat_val_reg_stp1]).reshape(1, -1), len(ecr_treat_list) // 2, axis=0).T,
                                                                        np.repeat(np.array(
                                                                            [i[specified_col_index] for i in
                                                                             GMV_Mat_val_reg_stp1]).reshape(1, -1),
                                                                                  len(ecr_treat_list) // 2 + 1,
                                                                                  axis=0).T), axis=1),
                                        GMV_Mat_val_reg_stp1)
        Trip_Mat_val_reg_stp1 = np.where(miniprice_flag, np.concatenate((np.repeat(
            np.array([-1000 for i in Trip_Mat_val_reg_stp1]).reshape(1, -1), len(ecr_treat_list) // 2, axis=0).T,
                                                                         np.repeat(np.array(
                                                                             [i[specified_col_index] for i in
                                                                              Trip_Mat_val_reg_stp1]).reshape(1, -1),
                                                                                   len(ecr_treat_list) // 2 + 1,
                                                                                   axis=0).T), axis=1),
                                         Trip_Mat_val_reg_stp1)
        Cost_Mat_val_reg_stp1 = np.where(miniprice_flag, np.concatenate((np.repeat(
            np.array([-1000 for i in Cost_Mat_val_reg_stp1]).reshape(1, -1), len(ecr_treat_list) // 2, axis=0).T,
                                                                         np.repeat(np.array(
                                                                             [i[specified_col_index] for i in
                                                                              Cost_Mat_val_reg_stp1]).reshape(1, -1),
                                                                                   len(ecr_treat_list) // 2 + 1,
                                                                                   axis=0).T), axis=1),
                                         Cost_Mat_val_reg_stp1)

        GMV_Mat_val_stp1 = np.where(miniprice_flag, np.repeat(
            np.array([i[specified_col_index] for i in GMV_Mat_val_stp1]).reshape(1, -1), len(ecr_treat_list), axis=0).T,
                                    GMV_Mat_val_stp1)
        Cost_Mat_val_stp1 = np.where(miniprice_flag, np.repeat(
            np.array([i[specified_col_index] for i in Cost_Mat_val_reg_stp1]).reshape(1, -1), len(ecr_treat_list),
            axis=0).T, Cost_Mat_val_stp1)
        Trip_Mat_val_stp1 = np.where(miniprice_flag, np.repeat(
            np.array([i[specified_col_index] for i in Trip_Mat_val_reg_stp1]).reshape(1, -1), len(ecr_treat_list),
            axis=0).T, Trip_Mat_val_stp1)
        Call_Mat_val_stp1 = np.where(miniprice_flag, np.repeat(
            np.array([i[specified_col_index] for i in Call_Mat_val_stp1]).reshape(1, -1), len(ecr_treat_list),
            axis=0).T, Call_Mat_val_stp1)
        CR_ECR_Mat_val_stp1 = np.where(miniprice_flag, np.repeat(
            np.array([i[specified_col_index] for i in CR_ECR_Mat_val_stp1]).reshape(1, -1), len(ecr_treat_list),
            axis=0).T, CR_ECR_Mat_val_stp1)
        Real_GMV_Mat_val = np.where(miniprice_flag, np.repeat(
            np.array([i[specified_col_index] for i in Real_GMV_Mat_val]).reshape(1, -1), len(ecr_treat_list), axis=0).T,
                                    Real_GMV_Mat_val)

        gp_mat_val_reg = GMV_Mat_val_reg_stp1 - Cost_Mat_val_reg_stp1
        gp_mat_val = GMV_Mat_val_stp1 - Cost_Mat_val_stp1
    df_cur_week = df_cur[df_cur.weekday.isin([5, 6])]
    GMV_weekends_base_stp1_ = df_cur_week['gmv_0_stp1'].sum()
    Cost_weekends_base_stp1_ = df_cur_week['cost_0_stp1'].sum()
    GP_weekends_base_stp1_ = GMV_weekends_base_stp1_ - Cost_weekends_base_stp1_
    del df_cur_week
    if function.lower() == 'gmv':
        f = GMV_base_stp1
    Mat_val_reg = GMV_Mat_val_reg_stp1

    c = []
    TH = []
    th_real = []
    Mat_val = []
    for cons_tag in constraint:
        if cons_tag == 'gp':
            Mat_val.append(GP_Mat_val_stp1)
        if cons_tag == 'trip':
            c.append(Trip_base_stp1)
            Mat_val.append(Trip_Mat_val_stp1)
        if cons_tag == 'gmv':
            c.append(GMV_base_stp1)
            Mat_val.append(GMV_Mat_val_stp1)
        if cons_tag == 'cost':
            c.append(Cost_base_stp1)
            Mat_val.append(Cost_Mat_val_stp1)
        if cons_tag == 'tr':
            c.append(GMV_base_stp1)
            Mat_val.append(GP_Mat_val_stp1)
        if cons_tag == 'cr':
            c.append(CR_base_stp1)
            Mat_val.append(CR_ECR_Mat_val_stp1)

    coef = [df_cur[lambda_value["lambda_{}".format(c)]].values for c in constraint]
    #     print(coef)
    bub_solution = np.argmax(Mat_val_reg + sum(coef[i].reshape(-1, 1) * Mat_val[i] for i in range(len(c))), axis=1)
    sum_Call_opt = np.sum(np.choose(bub_solution, Call_Mat_val_stp1.T))

    treat_vec = np.array(ecr_treat_list).reshape(1, -1)
    treat_mat = np.repeat(treat_vec, df_cur.shape[0], axis=0)
    sp = np.choose(bub_solution, treat_mat.T)
    df_cur['spforpassenger'] = sp
    df_cur.loc[:, "call_opt"] = np.choose(bub_solution, Call_Mat_val_stp1.T)

    df_cur["pax_opt"] = df_cur['spforpassenger']
    #     print(df_cur.columns[df_cur.columns.duplicated(keep=False)].tolist())

    #     df_cur["ecr_opt"] = df_cur.apply(lambda x: get_ecr_opt(x), axis=1)
    ECR_Mat = df_cur[var_ecr_list].values  # shape (n_rows, 11)
    df_cur["ecr_opt"] = np.choose(bub_solution, ECR_Mat.T)

    pred_opt_ecr = df_cur["ecr_opt"]
    pax_treat = df_cur["pax_opt"]
    var_trip_list = []
    var_gmv_list = []
    var_cost_list = []
    # var_call_list = []
    var_tch_list = []
    var_cr_ecr_list = []
    var_real_cost_list = []
    var_real_gmv_list = []
    for treat in cr_treat_list:
        dri_treat = treat
        var_cr = "pred_cr_" + str(dri_treat) + "_stp2"
        var_trip = "trip_" + str(dri_treat) + "_stp2"
        var_gmv = "gmv_" + str(dri_treat) + "_stp2"
        var_cost = "cost_" + str(dri_treat) + "_stp2"
        var_tch = "tch_" + str(dri_treat) + "_stp2"
        var_cr_ecr = 'cr_ecr_ ' + str(dri_treat) + "_stp2"
        var_real_cost = 'real_cost_' + str(dri_treat) + "_stp2"
        var_real_gmv = 'real_gmv_' + str(dri_treat) + "_stp2"
        #
        pred_treat_cr = df_cur[var_cr]

        df_cur[var_trip] = pred_opt_ecr * pred_treat_cr
        df_cur[var_gmv] = df_cur[pax_base_price] * (1 + pax_treat) * pred_opt_ecr * pred_treat_cr * (
                    1 + booking_percent)
        df_cur[var_cost] = df_cur[dri_base_price] * (1 + dri_treat) * pred_opt_ecr * pred_treat_cr
        df_cur[var_tch] = df_cur["eta"] * pred_opt_ecr * pred_treat_cr
        df_cur[var_cr_ecr] = (pred_treat_cr - CR0)
        df_cur[var_real_cost] = df_cur["dri_fee"] * (1 + dri_treat)
        df_cur[var_real_gmv] = df_cur["pre_total_fee_dp"] * (1 + dri_treat)

        var_trip_list.append(var_trip)
        var_gmv_list.append(var_gmv)
        var_cost_list.append(var_cost)
        var_tch_list.append(var_tch)
        var_cr_ecr_list.append(var_cr_ecr)
        var_real_cost_list.append(var_real_cost)
        var_real_gmv_list.append(var_real_gmv)

    tr_cap = 0.3
    # TR_cap 处理
    df_cur['gmv_pass_stp1'] = df_cur['pre_total_fee_dp'] * (df_cur['pax_opt'] + 1)

    # 切片处理，内存不够
    data = df_cur['gmv_pass_stp1'].values
    num_rows = len(data)
    num_cols = len(cr_treat_list)
    real_pass_gmv_mat_val = np.empty((num_rows, num_cols))
    chunk_size = 100000

    for start in range(0, num_rows, chunk_size):
        end = min(start + chunk_size, num_rows)
        real_pass_gmv_mat_val[start:end, :] = np.tile(data[start:end, np.newaxis], (1, num_cols))

    Real_Cost_Mat_val = df_cur[var_real_cost_list].values

    tr_cap_flag = ((real_pass_gmv_mat_val - Real_Cost_Mat_val) / real_pass_gmv_mat_val) > tr_cap
    column_sum = np.sum(tr_cap_flag, axis=0)

    rows_to_delete = np.all(tr_cap_flag, axis=1)
    #     print(df_cur.shape[0], real_pass_gmv_mat_val.shape[0])
    df_cur['road_dist_km_int'] = np.floor(df_cur['road_dist_km'])

    import copy

    Trip_Mat_val = df_cur[var_trip_list].values
    GMV_Mat_val = df_cur[var_gmv_list].values
    Cost_Mat_val = df_cur[var_cost_list].values
    GP_Mat_val = GMV_Mat_val - Cost_Mat_val
    # Call_Mat_val = df_cur[var_call_list].values
    Tch_Mat_val = df_cur[var_tch_list].values
    CR_ECR_Mat_val = df_cur[var_cr_ecr_list].values
    Real_Cost_Mat_val = df_cur[var_real_cost_list].values

    Trip_base = np.sum(df_cur['trip_0_stp2'])
    GMV_base = np.sum(df_cur['gmv_0_stp2'])
    Cost_base = np.sum(df_cur['cost_0_stp2'])
    GP_base = GMV_base - Cost_base
    Tch_base = np.sum(df_cur['tch_0_stp2'])
    CR_base = Trip_base / Call_base_stp1

    df_cur_weekday_stp2 = copy.copy(df_cur)
    value_columns = list(set(df_cur.columns.tolist()) - set(['weekday']))
    df_cur_weekday_stp2.loc[~df_cur['weekday'].isin([5, 6]), value_columns] = 0

    GP_weekends_Mat_val = df_cur_weekday_stp2[var_gmv_list].values - df_cur_weekday_stp2[var_cost_list].values

    GMV_Mat_val_reg = df_cur[var_gmv_list].values
    Cost_Mat_val_reg = df_cur[var_cost_list].values
    Trip_Mat_val_reg = df_cur[var_trip_list].values
    GP_Mat_val_reg = GMV_Mat_val - Cost_Mat_val

    GMV_Mat_val_reg[(df_cur['dp_drv'] > DP_DRV_TH + 1e-6), 0:sum(i < 0 for i in cr_treat_list)] = -1 * GMV_base
    #     print('司机侧dpth 筛选调价系数 {}条'.format(sum(df_cur['dp_drv'] > DP_DRV_TH + 1e-6)))

    # 判断PreTotalFee*dp*(1+sp)< MiniNumPrice及sp<0; 满足则将sp置为0；sp_new = 0
    miniprice_flag = Real_Cost_Mat_val < np.repeat(df_cur['dri_minimum_price'].values.reshape(1, -1),
                                                   len(cr_treat_list), axis=0).T
    #     print(Real_Cost_Mat_val, df_cur['dri_minimum_price'])
    #     print(miniprice_flag)

    replacement = [True] * (len(cr_treat_list) // 2) + [False] + [True] * (len(cr_treat_list) // 2)
    rows_to_replace = np.all(miniprice_flag, axis=1)  # 不满足采用sp0
    miniprice_flag[rows_to_replace] = replacement
    column_sum = np.sum(miniprice_flag, axis=0)
    #     print('方案1 司机侧最低消费 筛选调价系数{} {} 条'.format(cr_treat_list, column_sum))
    specified_col_index = sum(i < 0 for i in cr_treat_list)  # 0的那一列

    GMV_Mat_val_reg = np.where(miniprice_flag, np.concatenate((np.repeat(
        np.array([-2000 for i in GMV_Mat_val_reg]).reshape(1, -1), len(cr_treat_list) // 2, axis=0).T, np.repeat(
        np.array([i[specified_col_index] for i in GMV_Mat_val_reg]).reshape(1, -1), len(cr_treat_list) // 2 + 1,
        axis=0).T), axis=1), GMV_Mat_val_reg)
    Trip_Mat_val_reg = np.where(miniprice_flag, np.concatenate((np.repeat(
        np.array([-1000 for i in Trip_Mat_val_reg]).reshape(1, -1), len(cr_treat_list) // 2, axis=0).T, np.repeat(
        np.array([i[specified_col_index] for i in Trip_Mat_val_reg]).reshape(1, -1), len(cr_treat_list) // 2 + 1,
        axis=0).T), axis=1), Trip_Mat_val_reg)
    Cost_Mat_val_reg = np.where(miniprice_flag, np.concatenate((np.repeat(
        np.array([-1000 for i in Cost_Mat_val_reg]).reshape(1, -1), len(cr_treat_list) // 2, axis=0).T, np.repeat(
        np.array([i[specified_col_index] for i in Cost_Mat_val_reg]).reshape(1, -1), len(cr_treat_list) // 2 + 1,
        axis=0).T), axis=1), Cost_Mat_val_reg)

    GMV_Mat_val = np.where(miniprice_flag,
                           np.repeat(np.array([i[specified_col_index] for i in GMV_Mat_val]).reshape(1, -1),
                                     len(cr_treat_list), axis=0).T, GMV_Mat_val)
    Cost_Mat_val = np.where(miniprice_flag,
                            np.repeat(np.array([i[specified_col_index] for i in Cost_Mat_val]).reshape(1, -1),
                                      len(cr_treat_list), axis=0).T, Cost_Mat_val)
    Trip_Mat_val = np.where(miniprice_flag,
                            np.repeat(np.array([i[specified_col_index] for i in Trip_Mat_val]).reshape(1, -1),
                                      len(cr_treat_list), axis=0).T, Trip_Mat_val)
    # Call_Mat_val = np.where(miniprice_flag, np.repeat(np.array([i[specified_col_index] for i in Call_Mat_val]).reshape(1, -1), len(cr_treat_list), axis=0).T, Call_Mat_val)
    CR_ECR_Mat_val = np.where(miniprice_flag,
                              np.repeat(np.array([i[specified_col_index] for i in CR_ECR_Mat_val]).reshape(1, -1),
                                        len(cr_treat_list), axis=0).T, CR_ECR_Mat_val)
    GP_weekends_Mat_val = np.where(miniprice_flag, np.repeat(
        np.array([i[specified_col_index] for i in GP_weekends_Mat_val]).reshape(1, -1), len(cr_treat_list), axis=0).T,
                                   GP_weekends_Mat_val)
    Real_Cost_Mat_val = np.where(miniprice_flag,
                                 np.repeat(np.array([i[specified_col_index] for i in Real_Cost_Mat_val]).reshape(1, -1),
                                           len(cr_treat_list), axis=0).T, Real_Cost_Mat_val)

    GP_Mat_val_reg = GMV_Mat_val_reg - Cost_Mat_val_reg
    GP_Mat_val = GMV_Mat_val - Cost_Mat_val

    GMV_Mat_val_reg[tr_cap_flag] = -1 * GMV_base
    GP_Mat_val_reg[tr_cap_flag] = -1 * GP_base
    Trip_Mat_val_reg[tr_cap_flag] = -1 * Trip_base
    Cost_Mat_val_reg[tr_cap_flag] = -1 * Cost_base

    # 如果
    GMV_Mat_val_reg[rows_to_delete, len(cr_treat_list) // 2] = df_cur[rows_to_delete]['gmv_0_stp1']
    GP_Mat_val_reg[rows_to_delete, len(cr_treat_list) // 2] = df_cur[rows_to_delete]['gp_0_stp1']
    Trip_Mat_val_reg[rows_to_delete, len(cr_treat_list) // 2] = df_cur[rows_to_delete]['trip_0_stp1']
    Cost_Mat_val_reg[rows_to_delete, len(cr_treat_list) // 2] = df_cur[rows_to_delete]['cost_0_stp1']

    GP_Mat_val_reg[rows_to_delete, len(cr_treat_list) // 2] = df_cur[rows_to_delete]['gmv_0_stp1'] - \
                                                              df_cur[rows_to_delete]['cost_0_stp1']

    treat_vec = np.array(cr_treat_list).reshape(1, -1)
    treat_mat = np.repeat(treat_vec, df_cur.shape[0], axis=0)

    df_cur_week = df_cur[df_cur.weekday.isin([5, 6])]
    GMV_weekends_base_stp1_ = df_cur_week['gmv_0_stp1'].sum()
    Cost_weekends_base_stp1_ = df_cur_week['cost_0_stp1'].sum()
    GP_weekends_base_stp1_ = GMV_weekends_base_stp1_ - Cost_weekends_base_stp1_
    del df_cur_week

    Trip_base_stp1 = np.sum(df_cur['trip_0_stp2'])
    Call_base_stp1 = np.sum(df_cur["call_0"])
    GMV_base_stp1 = np.sum(df_cur['gmv_0_stp2'])
    Cost_base_stp1 = np.sum(df_cur['cost_0_stp2'])
    GP_base_stp1 = GMV_base_stp1 - Cost_base_stp1
    Tch_base_stp1 = np.sum(df_cur['tch_0_stp2'])
    CR_base_stp1 = Trip_base_stp1 / Call_base_stp1
    CR0 = CR_base_stp1
    Call_opt_stp1 = df_cur["call_opt"].sum()

    constraint = ['trip', 'tr', 'cr', 'cost', 'tr_weekend']

    if function.lower() == 'gmv':
        f = GMV_base_stp1
        Mat_val_reg = GMV_Mat_val_reg

    c = []
    TH = []
    th_real = []
    Mat_val = []
    for cons_tag in constraint:
        if cons_tag == 'gp':
            c.append(GP_base_stp1)
            Mat_val.append(GP_Mat_val)
        if cons_tag == 'trip':
            c.append(Trip_base_stp1)
            Mat_val.append(Trip_Mat_val)
        if cons_tag == 'gmv':
            c.append(GMV_base_stp1)
            Mat_val.append(GMV_Mat_val)
        if cons_tag == 'cost':
            c.append(Cost_base_stp1)
            Mat_val.append(Cost_Mat_val)
        if cons_tag == 'tr':
            c.append(GMV_base_stp1)
            Mat_val.append(GP_Mat_val)

        if cons_tag == 'tr_weekend':
            c.append(GMV_weekends_base_stp1_)
            Mat_val.append(GP_weekends_Mat_val)
        if cons_tag == 'cr':
            c.append(CR_base_stp1)
            Mat_val.append(CR_ECR_Mat_val)

    coef = [df_cur[lambda_value["lambda_{}_dri".format(c)]].values for c in constraint]
    bub_solution = np.argmax(Mat_val_reg + sum(coef[i].reshape(-1, 1) * Mat_val[i] for i in range(len(c))), axis=1)

    sum_Trip_opt = np.sum(np.choose(bub_solution, Trip_Mat_val.T))
    sum_GMV_opt = np.sum(np.choose(bub_solution, GMV_Mat_val.T))
    sum_GP_opt = np.sum(np.choose(bub_solution, GP_Mat_val.T))
    sum_Cost_opt = np.sum(np.choose(bub_solution, Cost_Mat_val.T))
    sum_CR_ECR_opt = np.sum(np.choose(bub_solution, CR_ECR_Mat_val.T))

    sum_GP_opt_weekday_stp2 = np.sum(np.choose(bub_solution, GP_weekends_Mat_val.T))

    total_call_opt += sum_Call_opt
    total_trip_opt += sum_Trip_opt
    total_gmv_opt += sum_GMV_opt
    total_gp_opt += sum_GP_opt
    total_cost_opt += sum_Cost_opt
    total_gp_opt_weekends += sum_GP_opt_weekday_stp2

    total_call_base += Call_base_stp1
    total_trip_base += Trip_base_stp1
    total_gmv_base += GMV_base_stp1
    total_gp_base += GP_base_stp1
    total_cost_base += Cost_base_stp1
    total_gmv_base_weekends += GMV_weekends_base_stp1_
    total_gp_base_weekends += GP_weekends_base_stp1_

    inc_trip_opt = sum_Trip_opt / Trip_base_stp1 - 1
    inc_gmv_opt = sum_GMV_opt / GMV_base_stp1 - 1
    inc_call_opt = sum_Call_opt / Call_base_stp1 - 1
    inc_gp_opt = sum_GP_opt / GP_base_stp1 - 1
    inc_tr_opt = (sum_GP_opt - GP_base_stp1) / GMV_base_stp1
    inc_cost_opt = sum_Cost_opt / Cost_base_stp1 - 1
    inc_cr_opt = (sum_Trip_opt / sum_Call_opt) - Trip_base_stp1 / Call_base_stp1
    inc_tr_weekend_opt = (sum_GP_opt_weekday_stp2 - GP_weekends_base_stp1_) / GMV_weekends_base_stp1_

    # print('result: Trip: {:.4%}; GMV: {:.3%}; Cost: {:.3%}; GP: {:.3%}; TR: {:.3%}; TR_opt: {:.3%}; sp: {}; CR0: {}; CR: {}; delta_CR: {}'.format(inc_trip_opt, inc_gmv_opt, inc_cost_opt, inc_gp_opt, inc_tr_opt, TR_opt, sp, CR0, sum_Trip_opt / sum_Call_opt, inc_cr_opt))
    print(
        '''result: %Call: {:.4%}; %Trip: {:.4%}; %GMV: {:.3%}; %Cost: {:.3%}; %GP: {:.3%};%TR: {:.3%}; CR: {};delta_CR: {:.3%}, TR_week: {:.3%}'''.format(
            inc_call_opt, inc_trip_opt, inc_gmv_opt, inc_cost_opt, inc_gp_opt,
            inc_tr_opt, round(sum_Trip_opt / sum_Call_opt, 6),
            inc_cr_opt, inc_tr_weekend_opt))

total_inc_trip_opt = total_trip_opt / total_trip_base - 1
total_inc_gmv_opt = total_gmv_opt / total_gmv_base - 1
total_inc_call_opt = total_call_opt / total_call_base - 1
total_inc_gp_opt = total_gp_opt / total_gp_base - 1
total_inc_tr_opt = (total_gp_opt - total_gp_base) / total_gmv_base
total_inc_cost_opt = total_cost_opt / total_cost_base - 1
total_inc_cr_opt = (total_trip_opt / total_call_opt) - total_trip_base / total_call_base
total_inc_tr_weekend_opt = (total_gp_opt_weekends - total_gp_base_weekends) / total_gmv_base_weekends
print(
    '''ALL result: %Call: {:.4%}; %Trip: {:.4%}; %GMV: {:.3%}; %Cost: {:.3%}; %GP: {:.3%};%TR: {:.3%}; delta_CR: {:.3%}, TR_week: {:.3%}'''.format(
        total_inc_call_opt, total_inc_trip_opt, total_inc_gmv_opt, total_inc_cost_opt, total_inc_gp_opt,
        total_inc_tr_opt, round(total_trip_opt / total_call_opt, 6),
        total_inc_cr_opt, total_inc_tr_weekend_opt))

# end

# 3.2  每30分钟调整lambda（v1）
# =========================
# 0. 构造 30min window
# =========================

def build_30min_window(df):
    df = df.copy()

    # 优先使用 minute 字段；没有 minute 时，尝试从时间戳字段解析
    if 'minute' in df.columns:
        minute = df['minute'].astype(int)
    elif 'minute_of_hour' in df.columns:
        minute = df['minute_of_hour'].astype(int)
    elif 'stat_time' in df.columns:
        ts = pd.to_datetime(df['stat_time'])
        minute = ts.dt.minute
        df['hour'] = ts.dt.hour
    elif 'create_time' in df.columns:
        ts = pd.to_datetime(df['create_time'])
        minute = ts.dt.minute
        df['hour'] = ts.dt.hour
    else:
        # 没有分钟字段时，只能退化成小时级；建议后续补充请求分钟字段
        minute = pd.Series(np.zeros(df.shape[0], dtype=int), index=df.index)

    df['minute_bucket'] = (minute // 30) * 30
    df['window_order'] = df['hour'].astype(int) * 60 + df['minute_bucket'].astype(int)

    df['window_id'] = (
        df['current_stat_date'].astype(str)
        + '_'
        + df['hour'].astype(int).astype(str).str.zfill(2)
        + '_'
        + df['minute_bucket'].astype(int).astype(str).str.zfill(2)
    )

    return df


df_rl = build_30min_window(df_)
# =========================
# 1. RL 配置
# =========================

PAX_CONSTRAINT = ['trip', 'tr', 'cr']

PAX_LAMBDA_COL = {
    'trip': 'lambdaTrip',
    'tr': 'lambdaGP',     # 你当前代码里 tr 对应 GP 约束
    'cr': 'lambdaCr'
}

DRI_CONSTRAINT = ['trip', 'tr', 'cr', 'cost', 'tr_weekend']

TRIP_TH = -0.01   # Trip 最多比基准低 1%
TR_TH = 0.0       # GP/TR 不低于基准
CR_TH = 0.0       # CR 不低于基准

REWARD_WEIGHTS = {
    'gmv': 1.0,
    'trip': 5.0,
    'tr': 5.0,
    'cr': 5.0,
    'smooth': 0.1
}

ACTION_UNIT_RATIO = 0.05   # 每次 lambda 调整 5%
LAMBDA_CLIP_RATIO = 0.20   # lambda 相对初始值最多上下浮动 20%

from itertools import product

def safe_div(a, b, eps=1e-9):
    return a / max(abs(b), eps)


def get_day_initial_pax_lambda(df_day):
    """
    线上/离线 lambda 是一天一变，因此用 median 取当天初始 lambda。
    """
    return {
        'lambdaTrip': df_day['lambdaTrip'].median(),
        'lambdaGP': df_day['lambdaGP'].median(),
        'lambdaCr': df_day['lambdaCr'].median()
    }


def get_driver_lambda_coef(df_cur, constraint):
    """
    司机侧 lambda 仍然沿用原来的离线/线上 lambda。
    """
    return [
        df_cur[lambda_value["lambda_{}_dri".format(c)]].values
        for c in constraint
    ]


def clip_lambda(cur_lambda, init_lambda, clip_ratio=LAMBDA_CLIP_RATIO):
    out = {}
    for k, v in cur_lambda.items():
        base = init_lambda[k]
        lo = base * (1 - clip_ratio)
        hi = base * (1 + clip_ratio)

        # lambda 可能为 0 时，给一个绝对浮动空间
        if abs(base) < 1e-9:
            lo = -clip_ratio
            hi = clip_ratio

        out[k] = min(max(v, lo), hi)
    return out


def apply_lambda_action(cur_lambda, init_lambda, action_tuple):
    """
    action_tuple 例如：
    (-1, 0, +1) 分别对应 lambdaTrip / lambdaGP / lambdaCr 的调整方向。
    """
    keys = ['lambdaTrip', 'lambdaGP', 'lambdaCr']
    new_lambda = cur_lambda.copy()

    for i, k in enumerate(keys):
        base = max(abs(init_lambda[k]), 1e-9)
        new_lambda[k] = new_lambda[k] + action_tuple[i] * ACTION_UNIT_RATIO * base

    return clip_lambda(new_lambda, init_lambda)


#乘客侧
def run_passenger_stage_30min(df_cur, cur_pax_lambda):
    """
    输入：
        df_cur: 当前 30min 窗口订单
        cur_pax_lambda: RL 调整后的乘客侧 lambda

    输出：
        df_out: 带 pax_opt / ecr_opt / call_opt 的窗口数据
        pax_metrics: 乘客侧 stp1 指标
    """
    df_cur = df_cur.copy().reset_index(drop=True)

    # ===== stp1 Mat 构造 =====
    Trip_Mat_val = df_cur[var_trip_list_stp1].values.copy()
    Call_Mat_val = df_cur[var_call_list].values.copy()
    GMV_Mat_val = df_cur[var_gmv_list_stp1].values.copy()
    Cost_Mat_val = df_cur[var_cost_list_stp1].values.copy()
    GP_Mat_val = GMV_Mat_val - Cost_Mat_val
    CR_ECR_Mat_val = df_cur[var_cr_ecr_list_stp1].values.copy()
    Real_GMV_Mat_val = df_cur[var_real_gmv_list_stp1].values.copy()

    # ===== base: passenger treat=0 =====
    Trip_base = np.sum(df_cur['trip_0_stp1'])
    Call_base = np.sum(df_cur['call_0'])
    GMV_base = np.sum(df_cur['gmv_0_stp1'])
    Cost_base = np.sum(df_cur['cost_0_stp1'])
    GP_base = GMV_base - Cost_base
    CR_base = Trip_base / max(Call_base, 1e-9)

    # ===== reg 矩阵：用于 argmax，带硬规则过滤 =====
    GMV_Mat_val_reg = GMV_Mat_val.copy()
    Cost_Mat_val_reg = Cost_Mat_val.copy()
    Trip_Mat_val_reg = Trip_Mat_val.copy()
    GP_Mat_val_reg = GP_Mat_val.copy()

    neg_cols = [i for i, t in enumerate(ecr_treat_list) if t < 0]
    zero_idx = list(ecr_treat_list).index(0)

    # 规则 1：高动调 / 高价订单禁止降价
    high_dp_rows = np.where(df_cur['dp'].values > DP_TH + 1e-6)[0]
    if len(high_dp_rows) > 0 and len(neg_cols) > 0:
        GMV_Mat_val_reg[np.ix_(high_dp_rows, neg_cols)] = -1 * GMV_base
        GP_Mat_val_reg[np.ix_(high_dp_rows, neg_cols)] = -1 * GP_base
        Trip_Mat_val_reg[np.ix_(high_dp_rows, neg_cols)] = -1 * Trip_base
        Cost_Mat_val_reg[np.ix_(high_dp_rows, neg_cols)] = -1 * Cost_base

    # 规则 2：最低消费命中禁止降价
    if minprice_sp_type in ["0", "", None, "None"]:
        min_price_rows = np.where(df_cur['min_price_hit'].astype(str).values == '1')[0]

        if len(min_price_rows) > 0 and len(neg_cols) > 0:
            GMV_Mat_val_reg[np.ix_(min_price_rows, neg_cols)] = -1 * GMV_base
            GP_Mat_val_reg[np.ix_(min_price_rows, neg_cols)] = -1 * GP_base
            Trip_Mat_val_reg[np.ix_(min_price_rows, neg_cols)] = -1 * Trip_base
            Cost_Mat_val_reg[np.ix_(min_price_rows, neg_cols)] = -1 * Cost_base

            # 真实指标矩阵中，非法降价候选替换为 0 档，避免后面 choose 取到非法值
            for mat in [GMV_Mat_val, Cost_Mat_val, Trip_Mat_val, GP_Mat_val, Call_Mat_val, CR_ECR_Mat_val]:
                mat[np.ix_(min_price_rows, neg_cols)] = mat[min_price_rows, zero_idx].reshape(-1, 1)

    else:
        miniprice_flag = Real_GMV_Mat_val < np.repeat(
            df_cur['mininum_price'].values.reshape(-1, 1),
            len(ecr_treat_list),
            axis=1
        )

        # 只处理降价档位
        treat_arr = np.array(ecr_treat_list)
        invalid_flag = miniprice_flag & (treat_arr.reshape(1, -1) < 0)

        if invalid_flag.any():
            GMV_Mat_val_reg[invalid_flag] = -1 * GMV_base
            GP_Mat_val_reg[invalid_flag] = -1 * GP_base
            Trip_Mat_val_reg[invalid_flag] = -1 * Trip_base
            Cost_Mat_val_reg[invalid_flag] = -1 * Cost_base

            rows, cols = np.where(invalid_flag)
            for mat in [GMV_Mat_val, Cost_Mat_val, Trip_Mat_val, GP_Mat_val, Call_Mat_val, CR_ECR_Mat_val]:
                mat[rows, cols] = mat[rows, zero_idx]

    # ===== Lagrangian score =====
    Mat_val_reg = GMV_Mat_val_reg

    score = (
        Mat_val_reg
        + cur_pax_lambda['lambdaTrip'] * Trip_Mat_val
        + cur_pax_lambda['lambdaGP'] * GP_Mat_val
        + cur_pax_lambda['lambdaCr'] * CR_ECR_Mat_val
    )

    bub_solution = np.argmax(score, axis=1)

    treat_vec = np.array(ecr_treat_list)
    sp = treat_vec[bub_solution]

    df_out = df_cur.copy()
    df_out['spforpassenger'] = sp
    df_out['pax_opt'] = sp
    df_out['call_opt'] = np.choose(bub_solution, Call_Mat_val.T)

    # ecr_opt 不用 apply，直接从 pred_ecr 矩阵取，速度更快
    ECR_Mat = df_out[var_ecr_list].values
    df_out['ecr_opt'] = np.choose(bub_solution, ECR_Mat.T)

    # ===== 乘客侧窗口指标 =====
    sum_Call_opt = np.sum(np.choose(bub_solution, Call_Mat_val.T))
    sum_Trip_opt = np.sum(np.choose(bub_solution, Trip_Mat_val.T))
    sum_GMV_opt = np.sum(np.choose(bub_solution, GMV_Mat_val.T))
    sum_Cost_opt = np.sum(np.choose(bub_solution, Cost_Mat_val.T))
    sum_GP_opt = np.sum(np.choose(bub_solution, GP_Mat_val.T))
    sum_CR_ECR_opt = np.sum(np.choose(bub_solution, CR_ECR_Mat_val.T))

    pax_metrics = {
        'call_base_stp1': Call_base,
        'trip_base_stp1': Trip_base,
        'gmv_base_stp1': GMV_base,
        'cost_base_stp1': Cost_base,
        'gp_base_stp1': GP_base,
        'cr_base_stp1': CR_base,

        'call_opt_stp1': sum_Call_opt,
        'trip_opt_stp1': sum_Trip_opt,
        'gmv_opt_stp1': sum_GMV_opt,
        'cost_opt_stp1': sum_Cost_opt,
        'gp_opt_stp1': sum_GP_opt,
        'cr_ecr_opt_stp1': sum_CR_ECR_opt,

        'inc_call_stp1': sum_Call_opt / max(Call_base, 1e-9) - 1,
        'inc_trip_stp1': sum_Trip_opt / max(Trip_base, 1e-9) - 1,
        'inc_gmv_stp1': sum_GMV_opt / max(GMV_base, 1e-9) - 1,
        'inc_gp_stp1': sum_GP_opt / max(GP_base, 1e-9) - 1,
        'inc_tr_stp1': (sum_GP_opt - GP_base) / max(GMV_base, 1e-9),
        'inc_cr_stp1': (sum_Trip_opt / max(sum_Call_opt, 1e-9)) - CR_base
    }

    return df_out, pax_metrics

#司机侧
def run_driver_stage_fixed_lambda(df_cur, pax_metrics):
    """
    司机侧保持原逻辑：
    - 乘客侧 pax_opt / ecr_opt 已由 RL 当前 lambda 决定
    - 司机侧 lambda 仍然使用 df_cur 中回带的离线/线上 lambda
    """
    df_cur = df_cur.copy().reset_index(drop=True)

    pred_opt_ecr = df_cur["ecr_opt"]
    pax_treat = df_cur["pax_opt"]

    var_trip_list = []
    var_gmv_list = []
    var_cost_list = []
    var_tch_list = []
    var_cr_ecr_list = []
    var_real_cost_list = []
    var_real_gmv_list = []

    # 这里 CR0 口径沿用当前窗口 stp1 base CR
    CR0_driver = pax_metrics['cr_base_stp1']

    for treat in cr_treat_list:
        dri_treat = treat

        var_cr = "pred_cr_" + str(dri_treat) + "_stp2"
        var_trip = "trip_" + str(dri_treat) + "_stp2"
        var_gmv = "gmv_" + str(dri_treat) + "_stp2"
        var_cost = "cost_" + str(dri_treat) + "_stp2"
        var_tch = "tch_" + str(dri_treat) + "_stp2"
        var_cr_ecr = "cr_ecr_" + str(dri_treat) + "_stp2"
        var_real_cost = "real_cost_" + str(dri_treat) + "_stp2"
        var_real_gmv = "real_gmv_" + str(dri_treat) + "_stp2"

        pred_treat_cr = df_cur[var_cr]

        df_cur[var_trip] = pred_opt_ecr * pred_treat_cr
        df_cur[var_gmv] = df_cur[pax_base_price] * (1 + pax_treat) * pred_opt_ecr * pred_treat_cr * (1 + booking_percent)
        df_cur[var_cost] = df_cur[dri_base_price] * (1 + dri_treat) * pred_opt_ecr * pred_treat_cr
        df_cur[var_tch] = df_cur["eta"] * pred_opt_ecr * pred_treat_cr

        # 原代码的司机侧 CR 约束线性项口径
        df_cur[var_cr_ecr] = pred_treat_cr - CR0_driver

        df_cur[var_real_cost] = df_cur["dri_fee"] * (1 + dri_treat)
        df_cur[var_real_gmv] = df_cur["pre_total_fee_dp"] * (1 + dri_treat)

        var_trip_list.append(var_trip)
        var_gmv_list.append(var_gmv)
        var_cost_list.append(var_cost)
        var_tch_list.append(var_tch)
        var_cr_ecr_list.append(var_cr_ecr)
        var_real_cost_list.append(var_real_cost)
        var_real_gmv_list.append(var_real_gmv)

    # ===== 司机侧矩阵 =====
    Trip_Mat_val = df_cur[var_trip_list].values.copy()
    GMV_Mat_val = df_cur[var_gmv_list].values.copy()
    Cost_Mat_val = df_cur[var_cost_list].values.copy()
    GP_Mat_val = GMV_Mat_val - Cost_Mat_val
    Tch_Mat_val = df_cur[var_tch_list].values.copy()
    CR_ECR_Mat_val = df_cur[var_cr_ecr_list].values.copy()
    Real_Cost_Mat_val = df_cur[var_real_cost_list].values.copy()

    Trip_base = np.sum(df_cur['trip_0_stp2'])
    GMV_base = np.sum(df_cur['gmv_0_stp2'])
    Cost_base = np.sum(df_cur['cost_0_stp2'])
    GP_base = GMV_base - Cost_base
    Tch_base = np.sum(df_cur['tch_0_stp2'])

    Call_base_stp1 = pax_metrics['call_base_stp1']
    CR_base = Trip_base / max(Call_base_stp1, 1e-9)

    # 周末 base
    df_cur_week = df_cur[df_cur.weekday.isin([5, 6])]
    GMV_weekends_base = df_cur_week['gmv_0_stp2'].sum()
    Cost_weekends_base = df_cur_week['cost_0_stp2'].sum()
    GP_weekends_base = GMV_weekends_base - Cost_weekends_base

    # 周末矩阵
    df_cur_weekend_tmp = df_cur.copy()
    value_columns = list(set(df_cur_weekend_tmp.columns.tolist()) - set(['weekday']))
    df_cur_weekend_tmp.loc[~df_cur_weekend_tmp['weekday'].isin([5, 6]), value_columns] = 0
    GP_weekends_Mat_val = df_cur_weekend_tmp[var_gmv_list].values - df_cur_weekend_tmp[var_cost_list].values

    # ===== reg 矩阵：用于 argmax，带司机侧硬规则 =====
    GMV_Mat_val_reg = GMV_Mat_val.copy()
    Cost_Mat_val_reg = Cost_Mat_val.copy()
    Trip_Mat_val_reg = Trip_Mat_val.copy()
    GP_Mat_val_reg = GP_Mat_val.copy()

    neg_cols = [i for i, t in enumerate(cr_treat_list) if t < 0]
    zero_idx = list(cr_treat_list).index(0)

    # 规则 1：司机侧高动调禁止降价
    high_dp_rows = np.where(df_cur['dp_drv'].values > DP_DRV_TH + 1e-6)[0]
    if len(high_dp_rows) > 0 and len(neg_cols) > 0:
        GMV_Mat_val_reg[np.ix_(high_dp_rows, neg_cols)] = -1 * GMV_base
        GP_Mat_val_reg[np.ix_(high_dp_rows, neg_cols)] = -1 * GP_base
        Trip_Mat_val_reg[np.ix_(high_dp_rows, neg_cols)] = -1 * Trip_base
        Cost_Mat_val_reg[np.ix_(high_dp_rows, neg_cols)] = -1 * Cost_base

    # 规则 2：司机侧最低消费
    miniprice_flag = Real_Cost_Mat_val < np.repeat(
        df_cur['dri_minimum_price'].values.reshape(-1, 1),
        len(cr_treat_list),
        axis=1
    )

    treat_arr = np.array(cr_treat_list)
    invalid_flag = miniprice_flag & (treat_arr.reshape(1, -1) < 0)

    if invalid_flag.any():
        GMV_Mat_val_reg[invalid_flag] = -1 * GMV_base
        GP_Mat_val_reg[invalid_flag] = -1 * GP_base
        Trip_Mat_val_reg[invalid_flag] = -1 * Trip_base
        Cost_Mat_val_reg[invalid_flag] = -1 * Cost_base

        rows, cols = np.where(invalid_flag)
        for mat in [GMV_Mat_val, Cost_Mat_val, Trip_Mat_val, GP_Mat_val, CR_ECR_Mat_val, GP_weekends_Mat_val, Real_Cost_Mat_val]:
            mat[rows, cols] = mat[rows, zero_idx]

    # 规则 3：tr_cap
    tr_cap = 0.3
    df_cur['gmv_pass_stp1'] = df_cur['pre_total_fee_dp'] * (df_cur['pax_opt'] + 1)

    real_pass_gmv_mat_val = np.tile(
        df_cur['gmv_pass_stp1'].values.reshape(-1, 1),
        (1, len(cr_treat_list))
    )

    tr_cap_flag = ((real_pass_gmv_mat_val - Real_Cost_Mat_val) / np.maximum(real_pass_gmv_mat_val, 1e-9)) > tr_cap
    rows_to_delete = np.all(tr_cap_flag, axis=1)

    GMV_Mat_val_reg[tr_cap_flag] = -1 * GMV_base
    GP_Mat_val_reg[tr_cap_flag] = -1 * GP_base
    Trip_Mat_val_reg[tr_cap_flag] = -1 * Trip_base
    Cost_Mat_val_reg[tr_cap_flag] = -1 * Cost_base

    # 所有司机候选都被 tr_cap 过滤时，强制回到司机 0 档
    if rows_to_delete.any():
        GMV_Mat_val_reg[rows_to_delete, zero_idx] = GMV_Mat_val[rows_to_delete, zero_idx]
        GP_Mat_val_reg[rows_to_delete, zero_idx] = GP_Mat_val[rows_to_delete, zero_idx]
        Trip_Mat_val_reg[rows_to_delete, zero_idx] = Trip_Mat_val[rows_to_delete, zero_idx]
        Cost_Mat_val_reg[rows_to_delete, zero_idx] = Cost_Mat_val[rows_to_delete, zero_idx]

    # ===== driver argmax =====
    function = 'gmv'
    constraint = DRI_CONSTRAINT

    if function.lower() == 'gmv':
        Mat_val_reg = GMV_Mat_val_reg

    Mat_val = []
    for cons_tag in constraint:
        if cons_tag == 'trip':
            Mat_val.append(Trip_Mat_val)
        elif cons_tag == 'tr':
            Mat_val.append(GP_Mat_val)
        elif cons_tag == 'cr':
            Mat_val.append(CR_ECR_Mat_val)
        elif cons_tag == 'cost':
            Mat_val.append(Cost_Mat_val)
        elif cons_tag == 'tr_weekend':
            Mat_val.append(GP_weekends_Mat_val)
        elif cons_tag == 'gp':
            Mat_val.append(GP_Mat_val)
        elif cons_tag == 'gmv':
            Mat_val.append(GMV_Mat_val)
        else:
            raise ValueError(f'unknown driver constraint: {cons_tag}')

    coef = get_driver_lambda_coef(df_cur, constraint)

    score = Mat_val_reg + sum(
        coef[i].reshape(-1, 1) * Mat_val[i]
        for i in range(len(constraint))
    )

    bub_solution = np.argmax(score, axis=1)

    driver_treat_vec = np.array(cr_treat_list)
    dsp = driver_treat_vec[bub_solution]
    df_cur['dspfordriver'] = dsp
    df_cur['driver_opt'] = dsp

    # ===== 聚合最终指标 =====
    sum_Trip_opt = np.sum(np.choose(bub_solution, Trip_Mat_val.T))
    sum_GMV_opt = np.sum(np.choose(bub_solution, GMV_Mat_val.T))
    sum_GP_opt = np.sum(np.choose(bub_solution, GP_Mat_val.T))
    sum_Cost_opt = np.sum(np.choose(bub_solution, Cost_Mat_val.T))
    sum_CR_ECR_opt = np.sum(np.choose(bub_solution, CR_ECR_Mat_val.T))
    sum_GP_weekend_opt = np.sum(np.choose(bub_solution, GP_weekends_Mat_val.T))

    sum_Call_opt = pax_metrics['call_opt_stp1']
    Call_base = pax_metrics['call_base_stp1']

    metrics = {
        'call_base': Call_base,
        'trip_base': Trip_base,
        'gmv_base': GMV_base,
        'cost_base': Cost_base,
        'gp_base': GP_base,
        'gp_weekend_base': GP_weekends_base,
        'gmv_weekend_base': GMV_weekends_base,

        'call_opt': sum_Call_opt,
        'trip_opt': sum_Trip_opt,
        'gmv_opt': sum_GMV_opt,
        'cost_opt': sum_Cost_opt,
        'gp_opt': sum_GP_opt,
        'cr_ecr_opt': sum_CR_ECR_opt,
        'gp_weekend_opt': sum_GP_weekend_opt,

        'inc_call': sum_Call_opt / max(Call_base, 1e-9) - 1,
        'inc_trip': sum_Trip_opt / max(Trip_base, 1e-9) - 1,
        'inc_gmv': sum_GMV_opt / max(GMV_base, 1e-9) - 1,
        'inc_cost': sum_Cost_opt / max(Cost_base, 1e-9) - 1,
        'inc_gp': sum_GP_opt / max(GP_base, 1e-9) - 1,
        'inc_tr': (sum_GP_opt - GP_base) / max(GMV_base, 1e-9),
        'inc_cr': (sum_Trip_opt / max(sum_Call_opt, 1e-9)) - (Trip_base / max(Call_base, 1e-9)),
        'inc_tr_weekend': (sum_GP_weekend_opt - GP_weekends_base) / max(GMV_weekends_base, 1e-9),

        'cr_opt': sum_Trip_opt / max(sum_Call_opt, 1e-9),
        'cr_base': Trip_base / max(Call_base, 1e-9),

        'avg_pax_treat': df_cur['pax_opt'].mean(),
        'avg_driver_treat': df_cur['driver_opt'].mean(),
        'pax_discount_ratio': (df_cur['pax_opt'] < 0).mean(),
        'pax_increase_ratio': (df_cur['pax_opt'] > 0).mean(),
        'driver_discount_ratio': (df_cur['driver_opt'] < 0).mean(),
        'driver_increase_ratio': (df_cur['driver_opt'] > 0).mean(),
        'n_order': df_cur.shape[0]
    }

    return df_cur, metrics

def simulate_30min_window(df_win, cur_pax_lambda):
    """
    一个 30min step:
    1. 用当前 passenger lambda 选乘客侧 treatment
    2. 固定乘客侧结果
    3. 用原司机侧 lambda 选司机侧 treatment
    4. 返回最终大盘指标
    """
    df_after_pax, pax_metrics = run_passenger_stage_30min(
        df_cur=df_win,
        cur_pax_lambda=cur_pax_lambda
    )

    df_solution, final_metrics = run_driver_stage_fixed_lambda(
        df_cur=df_after_pax,
        pax_metrics=pax_metrics
    )

    # 同时保留 stp1 信息，方便 state / reward 观察
    final_metrics.update({
        'inc_call_stp1': pax_metrics['inc_call_stp1'],
        'inc_trip_stp1': pax_metrics['inc_trip_stp1'],
        'inc_gmv_stp1': pax_metrics['inc_gmv_stp1'],
        'inc_gp_stp1': pax_metrics['inc_gp_stp1'],
        'inc_tr_stp1': pax_metrics['inc_tr_stp1'],
        'inc_cr_stp1': pax_metrics['inc_cr_stp1']
    })

    return df_solution, final_metrics

def compute_rl_reward(metrics, fixed_lambda_metrics, old_lambda, new_lambda):
    """
    reward:
    - GMV 相对 fixed daily lambda baseline 的提升
    - Trip / TR / CR 约束惩罚
    - lambda 平滑惩罚
    """

    # 1. GMV uplift：和固定原始 lambda 的同窗口结果比较
    gmv_lift_vs_fixed = (
        metrics['gmv_opt'] - fixed_lambda_metrics['gmv_opt']
    ) / max(abs(fixed_lambda_metrics['gmv_opt']), 1e-9)

    # 2. 约束惩罚：沿用你原来的 base-relative 口径
    # inc_trip >= -0.01
    # inc_tr >= 0
    # inc_cr >= 0
    v_trip = max(0.0, TRIP_TH - metrics['inc_trip'])
    v_tr = max(0.0, TR_TH - metrics['inc_tr'])
    v_cr = max(0.0, CR_TH - metrics['inc_cr'])

    # 3. 平滑惩罚
    smooth_loss = 0.0
    for k in new_lambda:
        smooth_loss += (new_lambda[k] - old_lambda[k]) ** 2

    reward = (
        REWARD_WEIGHTS['gmv'] * gmv_lift_vs_fixed
        - REWARD_WEIGHTS['trip'] * v_trip
        - REWARD_WEIGHTS['tr'] * v_tr
        - REWARD_WEIGHTS['cr'] * v_cr
        - REWARD_WEIGHTS['smooth'] * smooth_loss
    )

    reward_info = {
        'reward': reward,
        'gmv_lift_vs_fixed': gmv_lift_vs_fixed,
        'v_trip': v_trip,
        'v_tr': v_tr,
        'v_cr': v_cr,
        'smooth_loss': smooth_loss
    }

    return reward, reward_info

class PassengerLambda30MinEnv:
    def __init__(self, df_day):
        self.df_day = df_day.copy().reset_index(drop=True)

        self.window_list = (
            self.df_day[['window_id', 'window_order']]
            .drop_duplicates()
            .sort_values('window_order')['window_id']
            .tolist()
        )

        self.init_lambda = get_day_initial_pax_lambda(self.df_day)

        # action: lambdaTrip/lambdaGP/lambdaCr 分别取 -1/0/+1
        self.action_list = list(product([-1, 0, 1], repeat=3))

        self.fixed_baseline_cache = {}
        self._precompute_fixed_lambda_baseline()

        self.reset()

    def _precompute_fixed_lambda_baseline(self):
        """
        固定全天原始 passenger lambda 的 30min baseline。
        reward 里的 gmv uplift 用它做对照。
        """
        for window_id in self.window_list:
            df_win = self.df_day[self.df_day['window_id'] == window_id].copy()

            if df_win.shape[0] == 0:
                continue

            _, metrics = simulate_30min_window(
                df_win=df_win,
                cur_pax_lambda=self.init_lambda
            )

            self.fixed_baseline_cache[window_id] = metrics

    def reset(self):
        self.t = 0
        self.cur_lambda = self.init_lambda.copy()
        self.prev_metrics = None
        self.reward_history = []
        self.solution_list = []
        self.log_list = []

        return self._build_state()

    def _build_state(self):
        total_steps = max(len(self.window_list), 1)
        remain_steps = total_steps - self.t

        state = []

        # 1. 时间位置
        state.append(self.t / total_steps)
        state.append(remain_steps / total_steps)

        # 2. 当前 lambda 相对初始 lambda 的偏移
        for k in ['lambdaTrip', 'lambdaGP', 'lambdaCr']:
            base = max(abs(self.init_lambda[k]), 1e-9)
            state.append((self.cur_lambda[k] - self.init_lambda[k]) / base)

        # 3. 上一窗口大盘状态
        if self.prev_metrics is None:
            state.extend([0.0] * 8)
        else:
            state.extend([
                self.prev_metrics['inc_call'],
                self.prev_metrics['inc_trip'],
                self.prev_metrics['inc_gmv'],
                self.prev_metrics['inc_cost'],
                self.prev_metrics['inc_gp'],
                self.prev_metrics['inc_tr'],
                self.prev_metrics['inc_cr'],
                self.prev_metrics['inc_tr_weekend']
            ])

        # 4. 最近奖励趋势
        if len(self.reward_history) == 0:
            state.append(0.0)
        else:
            state.append(np.mean(self.reward_history[-3:]))

        return np.array(state, dtype=np.float32)

    def step(self, action_idx):
        action_tuple = self.action_list[action_idx]

        if self.t >= len(self.window_list):
            return self._build_state(), 0.0, True, {}

        window_id = self.window_list[self.t]
        df_win = self.df_day[self.df_day['window_id'] == window_id].copy()

        old_lambda = self.cur_lambda.copy()
        new_lambda = apply_lambda_action(
            cur_lambda=self.cur_lambda,
            init_lambda=self.init_lambda,
            action_tuple=action_tuple
        )

        df_solution, metrics = simulate_30min_window(
            df_win=df_win,
            cur_pax_lambda=new_lambda
        )

        fixed_metrics = self.fixed_baseline_cache[window_id]

        reward, reward_info = compute_rl_reward(
            metrics=metrics,
            fixed_lambda_metrics=fixed_metrics,
            old_lambda=old_lambda,
            new_lambda=new_lambda
        )

        # 更新状态
        self.cur_lambda = new_lambda
        self.prev_metrics = metrics
        self.reward_history.append(reward)

        # 保存订单级 solution
        df_solution['rl_window_id'] = window_id
        df_solution['rl_step'] = self.t
        df_solution['rl_lambdaTrip'] = new_lambda['lambdaTrip']
        df_solution['rl_lambdaGP'] = new_lambda['lambdaGP']
        df_solution['rl_lambdaCr'] = new_lambda['lambdaCr']
        self.solution_list.append(df_solution)

        log = {
            'window_id': window_id,
            'step': self.t,
            'action_idx': action_idx,
            'action_tuple': action_tuple,

            'lambdaTrip': new_lambda['lambdaTrip'],
            'lambdaGP': new_lambda['lambdaGP'],
            'lambdaCr': new_lambda['lambdaCr'],

            **metrics,
            **reward_info
        }

        self.log_list.append(log)

        self.t += 1
        done = self.t >= len(self.window_list)

        next_state = self._build_state()

        info = {
            'window_id': window_id,
            'metrics': metrics,
            'reward_info': reward_info,
            'lambda': new_lambda,
            'action_tuple': action_tuple
        }

        return next_state, reward, done, info

    def get_logs(self):
        return pd.DataFrame(self.log_list)

    def get_solution_df(self):
        if len(self.solution_list) == 0:
            return pd.DataFrame()
        return pd.concat(self.solution_list, axis=0, ignore_index=True)

def fixed_policy(env, state):
    """
    不调整 lambda，对应 action = (0,0,0)
    """
    return env.action_list.index((0, 0, 0))


def simple_rule_policy(env, state):
    """
    简单规则：
    - 上一窗口 Trip 不达标，增加 lambdaTrip
    - 上一窗口 TR 不达标，增加 lambdaGP
    - 上一窗口 CR 不达标，增加 lambdaCr
    """
    if env.prev_metrics is None:
        return env.action_list.index((0, 0, 0))

    a_trip = 1 if env.prev_metrics['inc_trip'] < TRIP_TH else 0
    a_tr = 1 if env.prev_metrics['inc_tr'] < TR_TH else 0
    a_cr = 1 if env.prev_metrics['inc_cr'] < CR_TH else 0

    action_tuple = (a_trip, a_tr, a_cr)
    return env.action_list.index(action_tuple)


def run_policy_one_day(env, policy_fn):
    state = env.reset()
    done = False

    while not done:
        action_idx = policy_fn(env, state)
        next_state, reward, done, info = env.step(action_idx)
        state = next_state

    log_df = env.get_logs()
    sol_df = env.get_solution_df()

    return log_df, sol_df

# =========================
# 运行某一天
# =========================

cur_date = s_data

df_day = df_rl[
    (df_rl["current_stat_date"] == cur_date)
#     & (df_rl["running_group"] == "group_1_tf_sep")
].copy()

env = PassengerLambda30MinEnv(df_day)

# 1. 固定 lambda baseline
fixed_log, fixed_solution = run_policy_one_day(env, fixed_policy)

# 2. 规则调节策略
rule_log, rule_solution = run_policy_one_day(env, simple_rule_policy)

print("fixed total reward:", fixed_log['reward'].sum())
print("rule total reward:", rule_log['reward'].sum())

compare = pd.DataFrame({
    'policy': ['fixed_lambda', 'rule_adjust'],
    'total_reward': [
        fixed_log['reward'].sum(),
        rule_log['reward'].sum()
    ],
    'total_gmv': [
        fixed_log['gmv_opt'].sum(),
        rule_log['gmv_opt'].sum()
    ],
    'total_trip': [
        fixed_log['trip_opt'].sum(),
        rule_log['trip_opt'].sum()
    ],
    'total_call': [
        fixed_log['call_opt'].sum(),
        rule_log['call_opt'].sum()
    ],
    'total_gp': [
        fixed_log['gp_opt'].sum(),
        rule_log['gp_opt'].sum()
    ],
    'avg_inc_trip': [
        fixed_log['inc_trip'].mean(),
        rule_log['inc_trip'].mean()
    ],
    'avg_inc_tr': [
        fixed_log['inc_tr'].mean(),
        rule_log['inc_tr'].mean()
    ],
    'avg_inc_cr': [
        fixed_log['inc_cr'].mean(),
        rule_log['inc_cr'].mean()
    ]
})

print(compare)

state = env.reset()
next_state, reward, done, info = env.step(action_idx)
state_dim = env.reset().shape[0]
action_dim = len(env.action_list)   # 3^3 = 27

all_dates = sorted(df_rl['current_stat_date'].unique())

for episode in range(num_episodes):
    cur_date = np.random.choice(all_dates)

    df_day = df_rl[
        (df_rl["current_stat_date"] == cur_date)
        & (df_rl["running_group"] == "group_1_tf_sep")
    ].copy()

    env = PassengerLambda30MinEnv(df_day)
    state = env.reset()
    done = False

    while not done:
        # DQN: epsilon-greedy 选择 action_idx
        action_idx = agent.select_action(state)

        next_state, reward, done, info = env.step(action_idx)

        agent.replay_buffer.add(
            state=state,
            action=action_idx,
            reward=reward,
            next_state=next_state,
            done=done
        )

        agent.train_step()

        state = next_state

# 3.3  30min Passenger-Lambda RL （v2）
"""
30min Passenger-Lambda RL 回带仿真代码

使用方式：
    1. 先保证 df_final 已经在 notebook / python 环境中存在。
       df_final 需要已经包含：
       - bj_times
       - stp1 乘客侧各 treatment 指标列：call_*, trip_*_stp1, gmv_*_stp1, cost_*_stp1, gp_*_stp1, cr_ecr_*_stp1
       - stp2 司机侧 pred_cr_*_stp2
       - 线上 lambda：lambdaTrip, lambdaGP, lambdaCr, lambdaTripDri, lambdaGPDri, lambdaCrDri, lambdaCostDri, lambdaGPWkdDri
       - 基础价格/规则字段：pre_total_fee_dp, dri_fee 或 driverpretotalfee/driver_dynamic_times, dp, dp_drv, min_price_hit, mininum_price, dri_minimum_price
    2. 运行：
        df_rl = prepare_rl_dataframe(df_final)
        train_result = train_dqn_on_df(
            df_rl,
            train_dates=None,
            num_episodes=100,
            running_group="group_1_tf_sep"  # 没有这个字段或不需要过滤时传 None
        )
    3. 评估：
        agent = train_result["agent"]
        eval_result = evaluate_agent_on_dates(
            df_rl,
            agent,
            eval_dates=train_result["train_dates"],
            running_group="group_1_tf_sep"
        )
"""
import random
from collections import deque
from itertools import product



# 1. 全局配置
# ============================================================

# 乘客侧候选 treatment，顺序必须与 stp1 指标矩阵列顺序一致
ECR_TREAT_LIST = [-0.15, -0.13, -0.1, -0.07, -0.05, 0, 0.05, 0.07, 0.1, 0.13, 0.15]

# 司机侧候选 treatment，顺序必须与 stp2 指标矩阵列顺序一致
CR_TREAT_LIST = [-0.1, -0.07, -0.05, 0, 0.05, 0.07, 0.1]

# 第一版 RL 只调乘客侧 lambda
PAX_LAMBDA_KEYS = ["lambdaTrip", "lambdaGP", "lambdaCr"]

# 乘客侧约束：trip / tr / cr
PAX_CONSTRAINT = ["trip", "tr", "cr"]

# 司机侧约束仍沿用原离线/线上 lambda
DRI_CONSTRAINT = ["trip", "tr", "cr", "cost", "tr_weekend"]

# 约束阈值：沿用原来的 base-relative 口径
TRIP_TH = -0.01     # Trip 最多比 base 策略低 1%
TR_TH = 0.0         # TR/GP 不低于 base
CR_TH = 0.0         # CR 不低于 base


# RL 决策窗口，建议先从 15 分钟试起
WINDOW_MINUTES = 15

# 动作档位：
# 第一版建议仍然用 [-1, 0, 1]，动作数 = 3^3 = 27
# 如果后续确认 15min 下 reward 更稳定，再改成 [-2, -1, 0, 1, 2]，动作数 = 5^3 = 125
ACTION_LEVELS = [-1, 0, 1]
# ACTION_LEVELS = [-2, -1, 0, 1, 2]

# 单步调整幅度：
ACTION_UNIT_RATIO = 0.05

# 全天 lambda 相对初始值上下最多浮动 50%
LAMBDA_CLIP_RATIO = 0.5

# reward 可选截断，防止个别窗口极端 reward 影响 DQN
# 不想截断就设为 None
REWARD_CLIP_VALUE = 5.0


# reward 权重
REWARD_WEIGHTS = {
    "gmv": 1.0,
    "trip": 5.0,
    "tr": 5.0,
    "cr": 5.0,
    "smooth": 0.1,
}

# 其他配置
BOOKING_PERCENT = 0.0
MINPRICE_SP_TYPE = "0"  # 与原代码一致：方案 0 下最低消费命中直接禁止乘客降价
DEFAULT_TR_CAP = 0.3

# 2. 字段映射与工具函数

def safe_div(a, b, eps=1e-9):
    return a / max(abs(b), eps)


def _fmt_treat(treat):
    """
    保持和当前 df_final 列名一致，例如：
    -0.1 -> "-0.1"
    0 -> "0"
    0.05 -> "0.05"
    """
    return str(treat)


def get_stp1_col(treat, metric):
    t = _fmt_treat(treat)
    if metric == "ecr":
        return f"pred_ecr_{t}"
    if metric == "call":
        return f"call_{t}"
    if metric == "trip":
        return f"trip_{t}_stp1"
    if metric == "gmv":
        return f"gmv_{t}_stp1"
    if metric == "cost":
        return f"cost_{t}_stp1"
    if metric == "gp":
        return f"gp_{t}_stp1"
    if metric == "tch":
        return f"tch_{t}_stp1"
    if metric == "cr_ecr":
        return f"cr_ecr_{t}_stp1"
    if metric == "real_cost":
        return f"real_cost_{t}_stp1"
    if metric == "real_gmv":
        return f"real_gmv_{t}_stp1"
    raise ValueError(f"unknown stp1 metric: {metric}")


def get_stp2_col(treat, metric):
    t = _fmt_treat(treat)
    if metric == "pred_cr":
        return f"pred_cr_{t}_stp2"
    if metric == "trip":
        return f"trip_{t}_stp2"
    if metric == "gmv":
        return f"gmv_{t}_stp2"
    if metric == "cost":
        return f"cost_{t}_stp2"
    if metric == "tch":
        return f"tch_{t}_stp2"
    if metric == "cr_ecr":
        return f"cr_ecr_{t}_stp2"
    if metric == "real_cost":
        return f"real_cost_{t}_stp2"
    if metric == "real_gmv":
        return f"real_gmv_{t}_stp2"
    raise ValueError(f"unknown stp2 metric: {metric}")


def build_cols(treat_list, metric, stage):
    if stage == 1:
        return [get_stp1_col(t, metric) for t in treat_list]
    if stage == 2:
        return [get_stp2_col(t, metric) for t in treat_list]
    raise ValueError(stage)


VAR_ECR_LIST = build_cols(ECR_TREAT_LIST, "ecr", 1)
VAR_CALL_LIST = build_cols(ECR_TREAT_LIST, "call", 1)
VAR_TRIP_LIST_STP1 = build_cols(ECR_TREAT_LIST, "trip", 1)
VAR_GMV_LIST_STP1 = build_cols(ECR_TREAT_LIST, "gmv", 1)
VAR_COST_LIST_STP1 = build_cols(ECR_TREAT_LIST, "cost", 1)
VAR_GP_LIST_STP1 = build_cols(ECR_TREAT_LIST, "gp", 1)
VAR_TCH_LIST_STP1 = build_cols(ECR_TREAT_LIST, "tch", 1)
VAR_CR_ECR_LIST_STP1 = build_cols(ECR_TREAT_LIST, "cr_ecr", 1)
VAR_REAL_COST_LIST_STP1 = build_cols(ECR_TREAT_LIST, "real_cost", 1)
VAR_REAL_GMV_LIST_STP1 = build_cols(ECR_TREAT_LIST, "real_gmv", 1)


def require_columns(df, cols, name="df"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少必要字段 {len(missing)} 个，前 30 个: {missing[:30]}")


def get_window_scalar(df_win, col, default_value):
    if col in df_win.columns:
        v = pd.to_numeric(df_win[col], errors="coerce").dropna()
        if len(v) > 0:
            return float(v.iloc[0])
    return float(default_value)


def get_day_initial_pax_lambda(df_day):
    """
    bj_times 每天第一条数据的 lambda 默认为当天线上采用的 lambda。（这里可以换，因为在前面数据分析的发现，离线数据上每天采用的lambda不一定是相同的）
    因此这里按 bj_times 排序后取第一条，而不是 median。
    """
    df_day = df_day.sort_values("bj_times").reset_index(drop=True)
    first = df_day.iloc[0]

    return {
        "lambdaTrip": float(first["lambdaTrip"]),
        "lambdaGP": float(first["lambdaGP"]),
        "lambdaCr": float(first["lambdaCr"]),
    }


def get_driver_lambda_coef(df_cur, constraint):
    """
    司机侧 lambda 仍然采用原 df_final 中回带的线上/离线 lambda。
    """
    coef = []
    for c in constraint:
        col = lambda_value[f"lambda_{c}_dri"]
        coef.append(df_cur[col].astype(float).values)
    return coef

#波动上下界
def clip_lambda(cur_lambda, init_lambda, clip_ratio=LAMBDA_CLIP_RATIO):
    out = {}
    for k, v in cur_lambda.items():
        base = float(init_lambda[k])

        if abs(base) < 1e-9:
            # 若当天初始 lambda 近似 0，用绝对 clip，避免完全无法动
            lo = -clip_ratio
            hi = clip_ratio
        else:
            lo = base * (1 - clip_ratio)
            hi = base * (1 + clip_ratio)
            if lo > hi:
                lo, hi = hi, lo

        out[k] = float(min(max(v, lo), hi))

    return out


def apply_lambda_action(cur_lambda, init_lambda, action_tuple):
    """
    action_tuple:
        (a_trip, a_tr, a_cr)，每个元素属于 {-1, 0, 1}
    """
    new_lambda = cur_lambda.copy()

    for i, k in enumerate(PAX_LAMBDA_KEYS):
        base = max(abs(float(init_lambda[k])), 1e-9)
        delta = action_tuple[i] * ACTION_UNIT_RATIO * base
        new_lambda[k] = float(new_lambda[k] + delta)

    return clip_lambda(new_lambda, init_lambda)


# 3. df_final 预处理：按 bj_times 划分 30min 窗口
# ============================================================

def prepare_rl_dataframe(df_final):
    """
    输入 df_final
    输出增加：
        - bj_times: datetime
        - rl_date: 按 bj_times 的北京日期
        - rl_window_start: bj_times floor 到 30min
        - window_id
        - window_order
    """
    df = df_final.copy()

    require_columns(
        df,
        [
            "bj_times",
            "lambdaTrip", "lambdaGP", "lambdaCr",
            "lambdaTripDri", "lambdaGPDri", "lambdaCrDri", "lambdaCostDri", "lambdaGPWkdDri",
            "pre_total_fee_dp", "driverpretotalfee", "driver_dynamic_times",
            "dp", "dp_drv", "min_price_hit", "mininum_price", "dri_minimum_price",
            "weekday",
        ],
        name="df_final"
    )

    require_columns(
        df,
        VAR_ECR_LIST + VAR_CALL_LIST + VAR_TRIP_LIST_STP1 + VAR_GMV_LIST_STP1
        + VAR_COST_LIST_STP1 + VAR_CR_ECR_LIST_STP1
        + [get_stp2_col(t, "pred_cr") for t in CR_TREAT_LIST],
        name="df_final"
    )

    df["bj_times"] = pd.to_datetime(df["bj_times"])
    df = df.sort_values("bj_times").reset_index(drop=True)

    # 统一基础字段
    if "dri_fee" not in df.columns:
        df["dri_fee"] = df["driverpretotalfee"] * df["driver_dynamic_times"]

    if "dist" not in df.columns and "road_dist_km" in df.columns:
        df["dist"] = df["road_dist_km"]

    # 按 bj_times 的北京日期作为 episode day
    df["rl_date"] = df["bj_times"].dt.strftime("%Y-%m-%d")
    # 按配置的 WINDOW_MINUTES 切窗口
    window_rule = f"{int(WINDOW_MINUTES)}min"
    df["rl_window_start"] = df["bj_times"].dt.floor(window_rule)

    df["window_id"] = df["rl_window_start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["window_order"] = (
            df["rl_window_start"].dt.hour * 60
            + df["rl_window_start"].dt.minute
    ).astype(int)

    return df


# 4. stp1：乘客侧根据 RL lambda 选择 pax treatment
# ============================================================

def run_passenger_stage_30min(df_win, cur_pax_lambda):
    """
    当前 30min 窗口内：
        用 RL 调整后的 passenger lambda 重新选择乘客侧 treatment。
    """
    df_cur = df_win.copy().reset_index(drop=True)

    # 窗口内动态阈值：优先使用 df_final 里的线上阈值
    dp_th_default = df_cur["dp"].quantile(0.85) if "dp" in df_cur.columns and len(df_cur) > 0 else 999.0
    dp_th = get_window_scalar(df_cur, "sp_apollo_dpth", dp_th_default)

    Trip_Mat_val = df_cur[VAR_TRIP_LIST_STP1].astype(float).values.copy()
    Call_Mat_val = df_cur[VAR_CALL_LIST].astype(float).values.copy()
    GMV_Mat_val = df_cur[VAR_GMV_LIST_STP1].astype(float).values.copy()
    Cost_Mat_val = df_cur[VAR_COST_LIST_STP1].astype(float).values.copy()
    GP_Mat_val = GMV_Mat_val - Cost_Mat_val
    CR_ECR_Mat_val = df_cur[VAR_CR_ECR_LIST_STP1].astype(float).values.copy()
    Real_GMV_Mat_val = df_cur[VAR_REAL_GMV_LIST_STP1].astype(float).values.copy()

    # base：乘客侧 treat=0
    zero_idx = ECR_TREAT_LIST.index(0)
    neg_cols = [i for i, t in enumerate(ECR_TREAT_LIST) if t < 0]

    Trip_base = float(df_cur[get_stp1_col(0, "trip")].sum())
    Call_base = float(df_cur[get_stp1_col(0, "call")].sum())
    GMV_base = float(df_cur[get_stp1_col(0, "gmv")].sum())
    Cost_base = float(df_cur[get_stp1_col(0, "cost")].sum())
    GP_base = GMV_base - Cost_base
    CR_base = Trip_base / max(Call_base, 1e-9)

    # reg 矩阵：用于 argmax，承载硬规则过滤
    GMV_Mat_val_reg = GMV_Mat_val.copy()
    Trip_Mat_val_reg = Trip_Mat_val.copy()
    Cost_Mat_val_reg = Cost_Mat_val.copy()
    GP_Mat_val_reg = GP_Mat_val.copy()

    # 规则 1：高动调/高价订单禁止降价
    high_dp_rows = np.where(df_cur["dp"].astype(float).values > dp_th + 1e-6)[0]
    if len(high_dp_rows) > 0 and len(neg_cols) > 0:
        GMV_Mat_val_reg[np.ix_(high_dp_rows, neg_cols)] = -1.0 * max(abs(GMV_base), 1.0)
        GP_Mat_val_reg[np.ix_(high_dp_rows, neg_cols)] = -1.0 * max(abs(GP_base), 1.0)
        Trip_Mat_val_reg[np.ix_(high_dp_rows, neg_cols)] = -1.0 * max(abs(Trip_base), 1.0)
        Cost_Mat_val_reg[np.ix_(high_dp_rows, neg_cols)] = -1.0 * max(abs(Cost_base), 1.0)

    # 规则 2：最低消费命中禁止降价
    if MINPRICE_SP_TYPE in ["0", "", None, "None"]:
        min_price_rows = np.where(df_cur["min_price_hit"].astype(str).values == "1")[0]

        if len(min_price_rows) > 0 and len(neg_cols) > 0:
            GMV_Mat_val_reg[np.ix_(min_price_rows, neg_cols)] = -1.0 * max(abs(GMV_base), 1.0)
            GP_Mat_val_reg[np.ix_(min_price_rows, neg_cols)] = -1.0 * max(abs(GP_base), 1.0)
            Trip_Mat_val_reg[np.ix_(min_price_rows, neg_cols)] = -1.0 * max(abs(Trip_base), 1.0)
            Cost_Mat_val_reg[np.ix_(min_price_rows, neg_cols)] = -1.0 * max(abs(Cost_base), 1.0)

            # 指标矩阵中也替换成 0 档，防止后续 choose 到非法降价候选
            for mat in [GMV_Mat_val, Cost_Mat_val, Trip_Mat_val, GP_Mat_val, Call_Mat_val, CR_ECR_Mat_val]:
                mat[np.ix_(min_price_rows, neg_cols)] = mat[min_price_rows, zero_idx].reshape(-1, 1)
    else:
        miniprice_flag = Real_GMV_Mat_val < np.repeat(
            df_cur["mininum_price"].astype(float).values.reshape(-1, 1),
            len(ECR_TREAT_LIST),
            axis=1
        )

        treat_arr = np.array(ECR_TREAT_LIST)
        invalid_flag = miniprice_flag & (treat_arr.reshape(1, -1) < 0)

        if invalid_flag.any():
            GMV_Mat_val_reg[invalid_flag] = -1.0 * max(abs(GMV_base), 1.0)
            GP_Mat_val_reg[invalid_flag] = -1.0 * max(abs(GP_base), 1.0)
            Trip_Mat_val_reg[invalid_flag] = -1.0 * max(abs(Trip_base), 1.0)
            Cost_Mat_val_reg[invalid_flag] = -1.0 * max(abs(Cost_base), 1.0)

            rows, cols = np.where(invalid_flag)
            for mat in [GMV_Mat_val, Cost_Mat_val, Trip_Mat_val, GP_Mat_val, Call_Mat_val, CR_ECR_Mat_val]:
                mat[rows, cols] = mat[rows, zero_idx]

    # passenger Lagrangian score
    score = (
        GMV_Mat_val_reg
        + cur_pax_lambda["lambdaTrip"] * Trip_Mat_val
        + cur_pax_lambda["lambdaGP"] * GP_Mat_val
        + cur_pax_lambda["lambdaCr"] * CR_ECR_Mat_val
    )

    bub_solution = np.argmax(score, axis=1)
    pax_treat_vec = np.array(ECR_TREAT_LIST)
    pax_opt = pax_treat_vec[bub_solution]

    df_out = df_cur.copy()
    df_out["spforpassenger"] = pax_opt
    df_out["pax_opt"] = pax_opt

    df_out["call_opt"] = np.choose(bub_solution, Call_Mat_val.T)

    ECR_Mat_val = df_out[VAR_ECR_LIST].astype(float).values
    df_out["ecr_opt"] = np.choose(bub_solution, ECR_Mat_val.T)

    sum_Call_opt = float(np.sum(np.choose(bub_solution, Call_Mat_val.T)))
    sum_Trip_opt = float(np.sum(np.choose(bub_solution, Trip_Mat_val.T)))
    sum_GMV_opt = float(np.sum(np.choose(bub_solution, GMV_Mat_val.T)))
    sum_Cost_opt = float(np.sum(np.choose(bub_solution, Cost_Mat_val.T)))
    sum_GP_opt = float(np.sum(np.choose(bub_solution, GP_Mat_val.T)))
    sum_CR_ECR_opt = float(np.sum(np.choose(bub_solution, CR_ECR_Mat_val.T)))

    pax_metrics = {
        "call_base_stp1": Call_base,
        "trip_base_stp1": Trip_base,
        "gmv_base_stp1": GMV_base,
        "cost_base_stp1": Cost_base,
        "gp_base_stp1": GP_base,
        "cr_base_stp1": CR_base,

        "call_opt_stp1": sum_Call_opt,
        "trip_opt_stp1": sum_Trip_opt,
        "gmv_opt_stp1": sum_GMV_opt,
        "cost_opt_stp1": sum_Cost_opt,
        "gp_opt_stp1": sum_GP_opt,
        "cr_ecr_opt_stp1": sum_CR_ECR_opt,

        "inc_call_stp1": safe_div(sum_Call_opt, Call_base) - 1,
        "inc_trip_stp1": safe_div(sum_Trip_opt, Trip_base) - 1,
        "inc_gmv_stp1": safe_div(sum_GMV_opt, GMV_base) - 1,
        "inc_cost_stp1": safe_div(sum_Cost_opt, Cost_base) - 1,
        "inc_gp_stp1": safe_div(sum_GP_opt, GP_base) - 1,
        "inc_tr_stp1": safe_div(sum_GP_opt - GP_base, GMV_base),
        "inc_cr_stp1": safe_div(sum_Trip_opt, sum_Call_opt) - CR_base,

        "dp_th_used": dp_th,
    }

    return df_out, pax_metrics

# 5. stp2：司机侧固定原 lambda 选择 driver treatment
# ============================================================

def run_driver_stage_fixed_lambda(df_after_pax, pax_metrics):
    """
    司机侧沿用原来的离线/线上 lambda。
    输入 df_after_pax 必须已有：
        - pax_opt
        - ecr_opt
        - call_opt
    """
    df_cur = df_after_pax.copy().reset_index(drop=True)

    pred_opt_ecr = df_cur["ecr_opt"].astype(float)
    pax_treat = df_cur["pax_opt"].astype(float)

    if "dri_fee" not in df_cur.columns:
        df_cur["dri_fee"] = df_cur["driverpretotalfee"] * df_cur["driver_dynamic_times"]

    pax_base_price = "pre_total_fee_dp"
    dri_base_price = "dri_fee"

    # stp2 中 CR0 沿用当前窗口 stp1 base CR
    CR0_driver = pax_metrics["cr_base_stp1"]

    var_trip_list = []
    var_gmv_list = []
    var_cost_list = []
    var_tch_list = []
    var_cr_ecr_list = []
    var_real_cost_list = []
    var_real_gmv_list = []

    for treat in CR_TREAT_LIST:
        var_cr = get_stp2_col(treat, "pred_cr")
        var_trip = get_stp2_col(treat, "trip")
        var_gmv = get_stp2_col(treat, "gmv")
        var_cost = get_stp2_col(treat, "cost")
        var_tch = get_stp2_col(treat, "tch")
        var_cr_ecr = get_stp2_col(treat, "cr_ecr")
        var_real_cost = get_stp2_col(treat, "real_cost")
        var_real_gmv = get_stp2_col(treat, "real_gmv")

        pred_treat_cr = df_cur[var_cr].astype(float)

        df_cur[var_trip] = pred_opt_ecr * pred_treat_cr
        df_cur[var_gmv] = df_cur[pax_base_price].astype(float) * (1 + pax_treat) * pred_opt_ecr * pred_treat_cr * (1 + BOOKING_PERCENT)
        df_cur[var_cost] = df_cur[dri_base_price].astype(float) * (1 + treat) * pred_opt_ecr * pred_treat_cr
        df_cur[var_tch] = df_cur["eta"].astype(float) * pred_opt_ecr * pred_treat_cr if "eta" in df_cur.columns else 0.0

        # 沿用原代码司机侧口径：pred_treat_cr - CR0
        df_cur[var_cr_ecr] = pred_treat_cr - CR0_driver

        df_cur[var_real_cost] = df_cur["dri_fee"].astype(float) * (1 + treat)
        df_cur[var_real_gmv] = df_cur["pre_total_fee_dp"].astype(float) * (1 + treat)

        var_trip_list.append(var_trip)
        var_gmv_list.append(var_gmv)
        var_cost_list.append(var_cost)
        var_tch_list.append(var_tch)
        var_cr_ecr_list.append(var_cr_ecr)
        var_real_cost_list.append(var_real_cost)
        var_real_gmv_list.append(var_real_gmv)

    Trip_Mat_val = df_cur[var_trip_list].astype(float).values.copy()
    GMV_Mat_val = df_cur[var_gmv_list].astype(float).values.copy()
    Cost_Mat_val = df_cur[var_cost_list].astype(float).values.copy()
    GP_Mat_val = GMV_Mat_val - Cost_Mat_val
    CR_ECR_Mat_val = df_cur[var_cr_ecr_list].astype(float).values.copy()
    Real_Cost_Mat_val = df_cur[var_real_cost_list].astype(float).values.copy()

    Trip_base = float(df_cur[get_stp2_col(0, "trip")].sum())
    GMV_base = float(df_cur[get_stp2_col(0, "gmv")].sum())
    Cost_base = float(df_cur[get_stp2_col(0, "cost")].sum())
    GP_base = GMV_base - Cost_base

    Call_base = float(pax_metrics["call_base_stp1"])
    CR_base = Trip_base / max(Call_base, 1e-9)

    # 周末 base 与周末矩阵
    is_weekend = df_cur["weekday"].isin([5, 6]).values

    GMV_weekends_base = float(df_cur.loc[is_weekend, get_stp2_col(0, "gmv")].sum())
    Cost_weekends_base = float(df_cur.loc[is_weekend, get_stp2_col(0, "cost")].sum())
    GP_weekends_base = GMV_weekends_base - Cost_weekends_base

    GP_weekends_Mat_val = GP_Mat_val.copy()
    GP_weekends_Mat_val[~is_weekend, :] = 0.0

    # reg 矩阵：用于 argmax
    GMV_Mat_val_reg = GMV_Mat_val.copy()
    Trip_Mat_val_reg = Trip_Mat_val.copy()
    Cost_Mat_val_reg = Cost_Mat_val.copy()
    GP_Mat_val_reg = GP_Mat_val.copy()

    zero_idx = CR_TREAT_LIST.index(0)
    neg_cols = [i for i, t in enumerate(CR_TREAT_LIST) if t < 0]

    # 窗口内司机侧动态阈值
    dp_drv_default = df_cur["dp_drv"].quantile(0.85) if "dp_drv" in df_cur.columns and len(df_cur) > 0 else 999.0
    dp_drv_th = get_window_scalar(df_cur, "dri_dp_th", dp_drv_default)

    # 规则 1：司机侧高动调禁止降价
    high_dp_rows = np.where(df_cur["dp_drv"].astype(float).values > dp_drv_th + 1e-6)[0]
    if len(high_dp_rows) > 0 and len(neg_cols) > 0:
        GMV_Mat_val_reg[np.ix_(high_dp_rows, neg_cols)] = -1.0 * max(abs(GMV_base), 1.0)
        GP_Mat_val_reg[np.ix_(high_dp_rows, neg_cols)] = -1.0 * max(abs(GP_base), 1.0)
        Trip_Mat_val_reg[np.ix_(high_dp_rows, neg_cols)] = -1.0 * max(abs(Trip_base), 1.0)
        Cost_Mat_val_reg[np.ix_(high_dp_rows, neg_cols)] = -1.0 * max(abs(Cost_base), 1.0)

    # 规则 2：司机最低消费
    miniprice_flag = Real_Cost_Mat_val < np.repeat(
        df_cur["dri_minimum_price"].astype(float).values.reshape(-1, 1),
        len(CR_TREAT_LIST),
        axis=1
    )

    treat_arr = np.array(CR_TREAT_LIST)
    invalid_flag = miniprice_flag & (treat_arr.reshape(1, -1) < 0)

    if invalid_flag.any():
        GMV_Mat_val_reg[invalid_flag] = -1.0 * max(abs(GMV_base), 1.0)
        GP_Mat_val_reg[invalid_flag] = -1.0 * max(abs(GP_base), 1.0)
        Trip_Mat_val_reg[invalid_flag] = -1.0 * max(abs(Trip_base), 1.0)
        Cost_Mat_val_reg[invalid_flag] = -1.0 * max(abs(Cost_base), 1.0)

        rows, cols = np.where(invalid_flag)
        for mat in [GMV_Mat_val, Cost_Mat_val, Trip_Mat_val, GP_Mat_val, CR_ECR_Mat_val, GP_weekends_Mat_val, Real_Cost_Mat_val]:
            mat[rows, cols] = mat[rows, zero_idx]

    # 规则 3：tr_cap
    tr_cap = get_window_scalar(df_cur, "tr_cap_th", DEFAULT_TR_CAP)

    real_pass_gmv_mat_val = np.tile(
        (df_cur["pre_total_fee_dp"].astype(float).values * (1 + df_cur["pax_opt"].astype(float).values)).reshape(-1, 1),
        (1, len(CR_TREAT_LIST))
    )

    tr_cap_flag = ((real_pass_gmv_mat_val - Real_Cost_Mat_val) / np.maximum(real_pass_gmv_mat_val, 1e-9)) > tr_cap
    rows_all_invalid = np.all(tr_cap_flag, axis=1)

    GMV_Mat_val_reg[tr_cap_flag] = -1.0 * max(abs(GMV_base), 1.0)
    GP_Mat_val_reg[tr_cap_flag] = -1.0 * max(abs(GP_base), 1.0)
    Trip_Mat_val_reg[tr_cap_flag] = -1.0 * max(abs(Trip_base), 1.0)
    Cost_Mat_val_reg[tr_cap_flag] = -1.0 * max(abs(Cost_base), 1.0)

    # 所有候选都被 tr_cap 过滤时，强制保留 0 档
    if rows_all_invalid.any():
        GMV_Mat_val_reg[rows_all_invalid, zero_idx] = GMV_Mat_val[rows_all_invalid, zero_idx]
        GP_Mat_val_reg[rows_all_invalid, zero_idx] = GP_Mat_val[rows_all_invalid, zero_idx]
        Trip_Mat_val_reg[rows_all_invalid, zero_idx] = Trip_Mat_val[rows_all_invalid, zero_idx]
        Cost_Mat_val_reg[rows_all_invalid, zero_idx] = Cost_Mat_val[rows_all_invalid, zero_idx]

    # driver Lagrangian score
    Mat_val_reg = GMV_Mat_val_reg

    Mat_val = []
    for cons_tag in DRI_CONSTRAINT:
        if cons_tag == "trip":
            Mat_val.append(Trip_Mat_val)
        elif cons_tag == "tr":
            Mat_val.append(GP_Mat_val)
        elif cons_tag == "cr":
            Mat_val.append(CR_ECR_Mat_val)
        elif cons_tag == "cost":
            Mat_val.append(Cost_Mat_val)
        elif cons_tag == "tr_weekend":
            Mat_val.append(GP_weekends_Mat_val)
        elif cons_tag == "gp":
            Mat_val.append(GP_Mat_val)
        elif cons_tag == "gmv":
            Mat_val.append(GMV_Mat_val)
        else:
            raise ValueError(f"unknown driver constraint: {cons_tag}")

    coef = get_driver_lambda_coef(df_cur, DRI_CONSTRAINT)

    score = Mat_val_reg + sum(
        coef[i].reshape(-1, 1) * Mat_val[i]
        for i in range(len(DRI_CONSTRAINT))
    )

    bub_solution = np.argmax(score, axis=1)

    driver_treat_vec = np.array(CR_TREAT_LIST)
    driver_opt = driver_treat_vec[bub_solution]

    df_solution = df_cur.copy()
    df_solution["dspfordriver"] = driver_opt
    df_solution["driver_opt"] = driver_opt

    sum_Call_opt = float(pax_metrics["call_opt_stp1"])
    sum_Trip_opt = float(np.sum(np.choose(bub_solution, Trip_Mat_val.T)))
    sum_GMV_opt = float(np.sum(np.choose(bub_solution, GMV_Mat_val.T)))
    sum_Cost_opt = float(np.sum(np.choose(bub_solution, Cost_Mat_val.T)))
    sum_GP_opt = float(np.sum(np.choose(bub_solution, GP_Mat_val.T)))
    sum_CR_ECR_opt = float(np.sum(np.choose(bub_solution, CR_ECR_Mat_val.T)))
    sum_GP_weekend_opt = float(np.sum(np.choose(bub_solution, GP_weekends_Mat_val.T)))

    metrics = {
        "call_base": Call_base,
        "trip_base": Trip_base,
        "gmv_base": GMV_base,
        "cost_base": Cost_base,
        "gp_base": GP_base,
        "gmv_weekend_base": GMV_weekends_base,
        "gp_weekend_base": GP_weekends_base,

        "call_opt": sum_Call_opt,
        "trip_opt": sum_Trip_opt,
        "gmv_opt": sum_GMV_opt,
        "cost_opt": sum_Cost_opt,
        "gp_opt": sum_GP_opt,
        "cr_ecr_opt": sum_CR_ECR_opt,
        "gp_weekend_opt": sum_GP_weekend_opt,

        "inc_call": safe_div(sum_Call_opt, Call_base) - 1,
        "inc_trip": safe_div(sum_Trip_opt, Trip_base) - 1,
        "inc_gmv": safe_div(sum_GMV_opt, GMV_base) - 1,
        "inc_cost": safe_div(sum_Cost_opt, Cost_base) - 1,
        "inc_gp": safe_div(sum_GP_opt, GP_base) - 1,
        "inc_tr": safe_div(sum_GP_opt - GP_base, GMV_base),
        "inc_cr": safe_div(sum_Trip_opt, sum_Call_opt) - safe_div(Trip_base, Call_base),
        "inc_tr_weekend": safe_div(sum_GP_weekend_opt - GP_weekends_base, GMV_weekends_base),

        "cr_opt": safe_div(sum_Trip_opt, sum_Call_opt),
        "cr_base": safe_div(Trip_base, Call_base),

        "avg_pax_treat": float(df_solution["pax_opt"].mean()),
        "avg_driver_treat": float(df_solution["driver_opt"].mean()),
        "pax_discount_ratio": float((df_solution["pax_opt"] < 0).mean()),
        "pax_increase_ratio": float((df_solution["pax_opt"] > 0).mean()),
        "driver_discount_ratio": float((df_solution["driver_opt"] < 0).mean()),
        "driver_increase_ratio": float((df_solution["driver_opt"] > 0).mean()),

        "n_order": int(df_solution.shape[0]),
        "dp_drv_th_used": dp_drv_th,
        "tr_cap_used": tr_cap,
    }

    # 加上 stp1 指标，方便 debug / state
    metrics.update({
        "inc_call_stp1": pax_metrics["inc_call_stp1"],
        "inc_trip_stp1": pax_metrics["inc_trip_stp1"],
        "inc_gmv_stp1": pax_metrics["inc_gmv_stp1"],
        "inc_cost_stp1": pax_metrics["inc_cost_stp1"],
        "inc_gp_stp1": pax_metrics["inc_gp_stp1"],
        "inc_tr_stp1": pax_metrics["inc_tr_stp1"],
        "inc_cr_stp1": pax_metrics["inc_cr_stp1"],
    })

    return df_solution, metrics

# 6. 单个 30min step 仿真 + reward
# ============================================================

def simulate_30min_window(df_win, cur_pax_lambda):
    # 清掉预测列含 NaN 的行，防止 NaN 污染 metrics / state / reward
    pred_cols = (
        VAR_ECR_LIST + VAR_CALL_LIST + VAR_TRIP_LIST_STP1
        + VAR_GMV_LIST_STP1 + VAR_COST_LIST_STP1 + VAR_CR_ECR_LIST_STP1
        + [get_stp2_col(t, "pred_cr") for t in CR_TREAT_LIST]
    )
    drop_cols = [c for c in pred_cols if c in df_win.columns]
    df_win = df_win.dropna(subset=drop_cols).reset_index(drop=True)

    if df_win.shape[0] == 0:
        return None, None

    df_after_pax, pax_metrics = run_passenger_stage_30min(
        df_win=df_win,
        cur_pax_lambda=cur_pax_lambda
    )

    df_solution, final_metrics = run_driver_stage_fixed_lambda(
        df_after_pax=df_after_pax,
        pax_metrics=pax_metrics
    )

    return df_solution, final_metrics


def compute_rl_reward(metrics, fixed_lambda_metrics, old_lambda, new_lambda):
    """
    reward 设计：
        GMV 相对固定日级 lambda baseline 的提升
        - trip/tr/cr 约束违约
        - lambda 平滑惩罚
    """
    gmv_lift_vs_fixed = safe_div(
        metrics["gmv_opt"] - fixed_lambda_metrics["gmv_opt"],
        fixed_lambda_metrics["gmv_opt"]
    )

    # base-relative 约束
    v_trip = max(0.0, TRIP_TH - metrics["inc_trip"])
    v_tr = max(0.0, TR_TH - metrics["inc_tr"])
    v_cr = max(0.0, CR_TH - metrics["inc_cr"])

    smooth_loss = 0.0
    for k in PAX_LAMBDA_KEYS:
        denom = max(abs(old_lambda[k]), 1e-9)
        smooth_loss += ((new_lambda[k] - old_lambda[k]) / denom) ** 2

    raw_reward = (
        REWARD_WEIGHTS["gmv"] * gmv_lift_vs_fixed
        - REWARD_WEIGHTS["trip"] * v_trip
        - REWARD_WEIGHTS["tr"] * v_tr
        - REWARD_WEIGHTS["cr"] * v_cr
        - REWARD_WEIGHTS["smooth"] * smooth_loss
    )

    if REWARD_CLIP_VALUE is not None:
        reward = float(np.clip(raw_reward, -REWARD_CLIP_VALUE, REWARD_CLIP_VALUE))
    else:
        reward = float(raw_reward)

    reward_info = {
        "reward": float(reward),
        "raw_reward": float(raw_reward),
        "gmv_lift_vs_fixed": float(gmv_lift_vs_fixed),
        "v_trip": float(v_trip),
        "v_tr": float(v_tr),
        "v_cr": float(v_cr),
        "smooth_loss": float(smooth_loss),
    }

    return float(reward), reward_info


# 7. RL 环境：每 30min 调一次 passenger lambda
# ============================================================

class PassengerLambda30MinEnv:
    def __init__(self, df_day, keep_solution=True):
        """
        df_day: 单个 rl_date 下的数据，已经经过 prepare_rl_dataframe。
        """
        self.df_day = df_day.copy().sort_values("bj_times").reset_index(drop=True)
        self.keep_solution = keep_solution

        self.window_list = (
            self.df_day[["window_id", "window_order", "rl_window_start"]]
            .drop_duplicates()
            .sort_values(["rl_window_start"])["window_id"]
            .tolist()
        )

        if len(self.window_list) == 0:
            raise ValueError("df_day 没有任何 window")

        self.init_lambda = get_day_initial_pax_lambda(self.df_day)

        # 每个 lambda 的动作属于 {-1,0,+1}，三维动作空间共 27 个动作
        self.action_list = list(product(ACTION_LEVELS, repeat=3))

        # 预计算固定日级 lambda baseline
        self.fixed_baseline_cache = {}
        self._precompute_fixed_lambda_baseline()

        self.reset()

    def _precompute_fixed_lambda_baseline(self):
        for window_id in self.window_list:
            df_win = self.df_day[self.df_day["window_id"] == window_id].copy()
            if df_win.shape[0] == 0:
                continue

            _, metrics = simulate_30min_window(
                df_win=df_win,
                cur_pax_lambda=self.init_lambda
            )

            if metrics is not None:
                self.fixed_baseline_cache[window_id] = metrics

    def reset(self):
        self.t = 0
        self.cur_lambda = self.init_lambda.copy()
        self.prev_metrics = None
        self.reward_history = []
        self.solution_list = []
        self.log_list = []
        return self._build_state()

    def _get_current_window_time_features(self):
        """
        当前待决策窗口的时间特征。
        这些信息线上可获得
        """
        if self.t >= len(self.window_list):
            # episode 结束后返回默认值
            return [0.0, 1.0, 0.0, 0.0]

        window_id = self.window_list[self.t]
        df_win = self.df_day[self.df_day["window_id"] == window_id]

        if df_win.shape[0] == 0:
            return [0.0, 1.0, 0.0, 0.0]

        ts = pd.to_datetime(df_win["rl_window_start"].iloc[0])
        minute_of_day = ts.hour * 60 + ts.minute

        angle = 2.0 * np.pi * minute_of_day / 1440.0

        hour_sin = np.sin(angle)
        hour_cos = np.cos(angle)

        # 优先使用数据里的 weekday，保持和你原业务口径一致
        if "weekday" in df_win.columns:
            weekday_val = pd.to_numeric(df_win["weekday"], errors="coerce").dropna()
            if len(weekday_val) > 0:
                weekday = int(weekday_val.iloc[0])
            else:
                weekday = int(ts.weekday())
        else:
            weekday = int(ts.weekday())

        weekday_norm = weekday / 6.0 if weekday <= 6 else 0.0
        is_weekend = 1.0 if weekday in [5, 6] else 0.0

        return [
            float(hour_sin),
            float(hour_cos),
            float(weekday_norm),
            float(is_weekend),
        ]

    def _build_state(self):
        total_steps = max(len(self.window_list), 1)
        remain_steps = total_steps - self.t

        state = []

        # 1. 时间进度
        state.append(self.t / total_steps)
        state.append(remain_steps / total_steps)

        # 2. 当前待决策窗口的时间特征
        # hour_sin, hour_cos, weekday_norm, is_weekend
        state.extend(self._get_current_window_time_features())

        # 3. lambda 相对当天初始 lambda 的偏移
        for k in PAX_LAMBDA_KEYS:
            base = max(abs(self.init_lambda[k]), 1e-9)
            state.append((self.cur_lambda[k] - self.init_lambda[k]) / base)

        # 4. 上一个窗口最终大盘状态
        if self.prev_metrics is None:
            state.extend([0.0] * 10)
        else:
            state.extend([
                self.prev_metrics["inc_call"],
                self.prev_metrics["inc_trip"],
                self.prev_metrics["inc_gmv"],
                self.prev_metrics["inc_cost"],
                self.prev_metrics["inc_gp"],
                self.prev_metrics["inc_tr"],
                self.prev_metrics["inc_cr"],
                self.prev_metrics["inc_tr_weekend"],
                self.prev_metrics["avg_pax_treat"],
                self.prev_metrics["avg_driver_treat"],
            ])

        # 5. rolling reward
        if len(self.reward_history) == 0:
            state.append(0.0)
        else:
            state.append(float(np.mean(self.reward_history[-3:])))

        return np.array(state, dtype=np.float32)

    @property
    def state_dim(self):
        return self._build_state().shape[0]

    @property
    def action_dim(self):
        return len(self.action_list)

    def step(self, action_idx):
        if self.t >= len(self.window_list):
            return self._build_state(), 0.0, True, {}

        action_tuple = self.action_list[int(action_idx)]
        window_id = self.window_list[self.t]

        df_win = self.df_day[self.df_day["window_id"] == window_id].copy()

        old_lambda = self.cur_lambda.copy()
        new_lambda = apply_lambda_action(
            cur_lambda=self.cur_lambda,
            init_lambda=self.init_lambda,
            action_tuple=action_tuple
        )

        df_solution, metrics = simulate_30min_window(
            df_win=df_win,
            cur_pax_lambda=new_lambda
        )

        # simulate_30min_window 返回 None 表示该窗口全为 NaN 行，跳过本步
        if metrics is None:
            self.cur_lambda = new_lambda
            self.t += 1
            done = self.t >= len(self.window_list)
            return self._build_state(), float('nan'), done, {"window_id": window_id, "skipped": True}

        fixed_metrics = self.fixed_baseline_cache[window_id]

        reward, reward_info = compute_rl_reward(
            metrics=metrics,
            fixed_lambda_metrics=fixed_metrics,
            old_lambda=old_lambda,
            new_lambda=new_lambda
        )

        self.cur_lambda = new_lambda
        self.prev_metrics = metrics
        self.reward_history.append(reward)

        if self.keep_solution:
            df_solution = df_solution.copy()
            df_solution["rl_window_id"] = window_id
            df_solution["rl_step"] = self.t
            df_solution["rl_action_idx"] = int(action_idx)
            df_solution["rl_action_tuple"] = str(action_tuple)
            df_solution["rl_lambdaTrip"] = new_lambda["lambdaTrip"]
            df_solution["rl_lambdaGP"] = new_lambda["lambdaGP"]
            df_solution["rl_lambdaCr"] = new_lambda["lambdaCr"]
            self.solution_list.append(df_solution)

        log = {
            "window_id": window_id,
            "step": self.t,
            "action_idx": int(action_idx),
            "action_tuple": str(action_tuple),

            "lambdaTrip": new_lambda["lambdaTrip"],
            "lambdaGP": new_lambda["lambdaGP"],
            "lambdaCr": new_lambda["lambdaCr"],

            "fixed_gmv_opt": fixed_metrics["gmv_opt"],
            "fixed_trip_opt": fixed_metrics["trip_opt"],
            "fixed_call_opt": fixed_metrics["call_opt"],

            **metrics,
            **reward_info,
        }

        self.log_list.append(log)

        self.t += 1
        done = self.t >= len(self.window_list)
        next_state = self._build_state()

        info = {
            "window_id": window_id,
            "metrics": metrics,
            "reward_info": reward_info,
            "lambda": new_lambda,
            "action_tuple": action_tuple,
        }

        return next_state, reward, done, info

    def get_logs(self):
        return pd.DataFrame(self.log_list)

    def get_solution_df(self):
        if len(self.solution_list) == 0:
            return pd.DataFrame()
        return pd.concat(self.solution_list, axis=0, ignore_index=True)


# 8. Baseline / 规则策略
# ============================================================

def fixed_policy(env, state):
    zero_action = tuple([0] * len(PAX_LAMBDA_KEYS))
    return env.action_list.index(zero_action)


def simple_rule_policy(env, state):
    """
    简单可解释策略，用来 sanity check：
        上一窗口哪个约束不达标，就把对应 lambda 往 + 方向调。

    这里默认 lambda 方向与原 Lagrangian score 一致。
    如果发现 + 方向使约束更差，需要反向。
    """
    zero_action = tuple([0] * len(PAX_LAMBDA_KEYS))

    if env.prev_metrics is None:
        return env.action_list.index(zero_action)

    # 单步规则策略仍然只走 +1，不直接走 +2，避免震荡
    a_trip = 1 if env.prev_metrics["inc_trip"] < TRIP_TH else 0
    a_tr = 1 if env.prev_metrics["inc_tr"] < TR_TH else 0
    a_cr = 1 if env.prev_metrics["inc_cr"] < CR_TH else 0

    action_tuple = (a_trip, a_tr, a_cr)

    if action_tuple not in env.action_list:
        action_tuple = zero_action

    return env.action_list.index(action_tuple)

def run_policy_one_day(df_day, policy_fn, keep_solution=True):
    env = PassengerLambda30MinEnv(df_day, keep_solution=keep_solution)
    state = env.reset()
    done = False

    while not done:
        action_idx = policy_fn(env, state)
        next_state, reward, done, info = env.step(action_idx)
        state = next_state

    return env.get_logs(), env.get_solution_df()


def summarize_rl_log(log_df, name="policy"):
    if log_df.empty:
        raise ValueError("log_df is empty")

    total_call_opt = log_df["call_opt"].sum()
    total_trip_opt = log_df["trip_opt"].sum()
    total_gmv_opt = log_df["gmv_opt"].sum()
    total_cost_opt = log_df["cost_opt"].sum()
    total_gp_opt = log_df["gp_opt"].sum()
    total_gp_weekend_opt = log_df["gp_weekend_opt"].sum()

    total_call_base = log_df["call_base"].sum()
    total_trip_base = log_df["trip_base"].sum()
    total_gmv_base = log_df["gmv_base"].sum()
    total_cost_base = log_df["cost_base"].sum()
    total_gp_base = log_df["gp_base"].sum()
    total_gmv_weekend_base = log_df["gmv_weekend_base"].sum()
    total_gp_weekend_base = log_df["gp_weekend_base"].sum()

    result = {
        "policy": name,
        "total_reward": log_df["reward"].sum(),

        "inc_call": safe_div(total_call_opt, total_call_base) - 1,
        "inc_trip": safe_div(total_trip_opt, total_trip_base) - 1,
        "inc_gmv": safe_div(total_gmv_opt, total_gmv_base) - 1,
        "inc_cost": safe_div(total_cost_opt, total_cost_base) - 1,
        "inc_gp": safe_div(total_gp_opt, total_gp_base) - 1,
        "inc_tr": safe_div(total_gp_opt - total_gp_base, total_gmv_base),
        "inc_cr": safe_div(total_trip_opt, total_call_opt) - safe_div(total_trip_base, total_call_base),
        "inc_tr_weekend": safe_div(total_gp_weekend_opt - total_gp_weekend_base, total_gmv_weekend_base),

        "cr_opt": safe_div(total_trip_opt, total_call_opt),
        "cr_base": safe_div(total_trip_base, total_call_base),

        "total_call_opt": total_call_opt,
        "total_trip_opt": total_trip_opt,
        "total_gmv_opt": total_gmv_opt,
        "total_cost_opt": total_cost_opt,
        "total_gp_opt": total_gp_opt,
    }

    print(
        "{} ALL result: %Call: {:.4%}; %Trip: {:.4%}; %GMV: {:.3%}; "
        "%Cost: {:.3%}; %GP: {:.3%}; %TR: {:.3%}; CR: {:.6f}; "
        "delta_CR: {:.3%}; TR_week: {:.3%}; reward: {:.6f}".format(
            name,
            result["inc_call"],
            result["inc_trip"],
            result["inc_gmv"],
            result["inc_cost"],
            result["inc_gp"],
            result["inc_tr"],
            result["cr_opt"],
            result["inc_cr"],
            result["inc_tr_weekend"],
            result["total_reward"],
        )
    )

    return result

# 9. DQN Agent
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, int(action), float(reward), next_state, bool(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        return (
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.int64),
            np.asarray(reward, dtype=np.float32),
            np.asarray(next_state, dtype=np.float32),
            np.asarray(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        return torch, nn, optim
    except Exception as e:
        raise ImportError(
            "DQN 训练需要 PyTorch。请先安装/加载 torch，或者先使用 fixed_policy/simple_rule_policy 验证环境。"
        ) from e


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.95,
        batch_size=64,
        buffer_size=5000,
        target_update_interval=100,
        hidden_dim=128,
        seed=2026,
    ):
        torch, nn, optim = _import_torch()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.torch = torch
        self.nn = nn
        self.optim = optim

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.train_steps = 0

        class QNet(nn.Module):
            def __init__(self, s_dim, a_dim, h_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(s_dim, h_dim),
                    nn.ReLU(),
                    nn.Linear(h_dim, h_dim),
                    nn.ReLU(),
                    nn.Linear(h_dim, a_dim),
                )

            def forward(self, x):
                return self.net(x)

        self.q_net = QNet(state_dim, action_dim, hidden_dim)
        self.target_net = QNet(state_dim, action_dim, hidden_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        torch = self.torch
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        torch = self.torch
        nn = self.nn

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        q_values = self.q_net(states).gather(1, actions)
        # Double DQN:
        # 1. 用 q_net 选择 next action
        # 2. 用 target_net 评估该 action 的 Q 值
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * (1.0 - dones) * next_q_values

        # Huber loss 比 MSE 更抗异常 reward
        # loss 左项必须是 q_net 的输出(梯度才能流回被训练的网络);
        # 若用 target_net 输出则对 q_net 梯度恒0, 训练静默空转
        loss = nn.functional.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.train_steps += 1

        if self.train_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

# 10. DQN 训练与评估入口
# ============================================================

def _filter_df_for_day(df_rl, rl_date, running_group=None):
    df_day = df_rl[df_rl["rl_date"] == rl_date].copy()

    if running_group is not None and "running_group" in df_day.columns:
        df_day = df_day[df_day["running_group"] == running_group].copy()

    df_day = df_day.sort_values("bj_times").reset_index(drop=True)
    return df_day


def train_dqn_on_df(
    df_rl,
    train_dates=None,
    running_group=None,
    num_episodes=100,
    epsilon_start=0.5,
    epsilon_end=0.05,
    epsilon_decay=0.98,
    seed=2026,
):
    """
    在 df_rl 上训练 DQN。
    每个 episode 随机抽一天，每天内部按 bj_times 30min 窗口顺序走完整轨迹。
    """
    if train_dates is None:
        train_dates = sorted(df_rl["rl_date"].unique().tolist())

    if len(train_dates) == 0:
        raise ValueError("train_dates 为空")

    # 用第一天初始化 state/action 维度
    first_day = None
    for d in train_dates:
        tmp = _filter_df_for_day(df_rl, d, running_group=running_group)
        if tmp.shape[0] > 0:
            first_day = tmp
            break

    if first_day is None:
        raise ValueError("所有 train_dates 过滤后均为空，请检查 running_group 或 rl_date")

    env0 = PassengerLambda30MinEnv(first_day, keep_solution=False)
    agent = DQNAgent(
        state_dim=env0.state_dim,
        action_dim=env0.action_dim,
        seed=seed,
    )

    epsilon = epsilon_start
    episode_logs = []

    # 预先计算每天 dropna 后的有效行数，过滤掉全 NaN 日期
    pred_cols_check = (
        VAR_ECR_LIST + VAR_CALL_LIST + VAR_TRIP_LIST_STP1
        + VAR_GMV_LIST_STP1 + VAR_COST_LIST_STP1 + VAR_CR_ECR_LIST_STP1
        + [get_stp2_col(t, "pred_cr") for t in CR_TREAT_LIST]
    )
    valid_train_dates = []
    for d in train_dates:
        tmp = _filter_df_for_day(df_rl, d, running_group=running_group)
        drop_cols = [c for c in pred_cols_check if c in tmp.columns]
        valid_rows = tmp.dropna(subset=drop_cols).shape[0]
        if valid_rows >= 100:   # 至少 100 行有效数据才纳入训练
            valid_train_dates.append(d)
    if len(valid_train_dates) == 0:
        raise ValueError("所有 train_dates dropna 后均不足 100 行，请检查数据")
    print(f"有效训练日期: {len(valid_train_dates)}/{len(train_dates)} 天，"
          f"排除: {sorted(set(train_dates) - set(valid_train_dates))}")

    for ep in range(num_episodes):
        cur_date = random.choice(valid_train_dates)
        df_day = _filter_df_for_day(df_rl, cur_date, running_group=running_group)

        if df_day.shape[0] == 0:
            continue

        env = PassengerLambda30MinEnv(df_day, keep_solution=False)
        state = env.reset()
        done = False
        ep_reward = 0.0
        ep_losses = []

        while not done:
            action_idx = agent.select_action(state, epsilon=epsilon)
            next_state, reward, done, info = env.step(action_idx)

            # 过滤 NaN：NaN reward 或 NaN state 进入 buffer 会导致 loss 永远是 NaN
            if (not np.isnan(reward)
                    and not np.any(np.isnan(state))
                    and not np.any(np.isnan(next_state))):
                agent.replay_buffer.add(state, action_idx, reward, next_state, done)
                loss = agent.train_step()
                if loss is not None and not np.isnan(loss):
                    ep_losses.append(loss)

            state = next_state
            if not np.isnan(reward):
                ep_reward += reward

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        log_df = env.get_logs()
        episode_logs.append({
            "episode": ep,
            "rl_date": cur_date,
            "episode_reward": ep_reward,
            "epsilon": epsilon,
            "avg_loss": float(np.mean(ep_losses)) if len(ep_losses) > 0 else np.nan,
            "n_steps": len(log_df),
            "inc_gmv_total": safe_div(log_df["gmv_opt"].sum(), log_df["gmv_base"].sum()) - 1,
            "inc_trip_total": safe_div(log_df["trip_opt"].sum(), log_df["trip_base"].sum()) - 1,
            "inc_tr_total": safe_div(log_df["gp_opt"].sum() - log_df["gp_base"].sum(), log_df["gmv_base"].sum()),
            "inc_cr_total": safe_div(log_df["trip_opt"].sum(), log_df["call_opt"].sum()) - safe_div(log_df["trip_base"].sum(), log_df["call_base"].sum()),
        })

        if (ep + 1) % 10 == 0:
            print(
                f"[episode {ep + 1}/{num_episodes}] date={cur_date}, "
                f"reward={ep_reward:.6f}, epsilon={epsilon:.4f}, "
                f"avg_loss={episode_logs[-1]['avg_loss']}"
            )

    return {
        "agent": agent,
        "episode_log": pd.DataFrame(episode_logs),
        "train_dates": train_dates,
        "running_group": running_group,
    }


def make_agent_policy(agent, epsilon=0.0):
    def _policy(env, state):
        return agent.select_action(state, epsilon=epsilon)
    return _policy


def evaluate_agent_on_dates(df_rl, agent, eval_dates=None, running_group=None, keep_solution=True):
    if eval_dates is None:
        eval_dates = sorted(df_rl["rl_date"].unique().tolist())

    all_logs = []
    all_solutions = []
    all_results = []

    agent_policy = make_agent_policy(agent, epsilon=0.0)

    for d in eval_dates:
        df_day = _filter_df_for_day(df_rl, d, running_group=running_group)
        if df_day.shape[0] == 0:
            continue

        log_df, sol_df = run_policy_one_day(df_day, agent_policy, keep_solution=keep_solution)
        log_df["rl_date"] = d

        if not sol_df.empty:
            sol_df["rl_date"] = d

        result = summarize_rl_log(log_df, name=f"dqn_{d}")
        result["rl_date"] = d

        all_logs.append(log_df)
        all_solutions.append(sol_df)
        all_results.append(result)

    return {
        "log_df": pd.concat(all_logs, axis=0, ignore_index=True) if len(all_logs) > 0 else pd.DataFrame(),
        "solution_df": pd.concat(all_solutions, axis=0, ignore_index=True) if len(all_solutions) > 0 else pd.DataFrame(),
        "summary_df": pd.DataFrame(all_results),
    }


def run_baselines_on_dates(df_rl, dates=None, running_group=None, keep_solution=True):
    if dates is None:
        dates = sorted(df_rl["rl_date"].unique().tolist())

    fixed_logs = []
    rule_logs = []
    fixed_solutions = []
    rule_solutions = []
    results = []

    for d in dates:
        df_day = _filter_df_for_day(df_rl, d, running_group=running_group)
        if df_day.shape[0] == 0:
            continue

        fixed_log, fixed_sol = run_policy_one_day(df_day, fixed_policy, keep_solution=keep_solution)
        rule_log, rule_sol = run_policy_one_day(df_day, simple_rule_policy, keep_solution=keep_solution)

        fixed_log["rl_date"] = d
        rule_log["rl_date"] = d
        fixed_sol["rl_date"] = d if not fixed_sol.empty else d
        rule_sol["rl_date"] = d if not rule_sol.empty else d

        fixed_result = summarize_rl_log(fixed_log, name=f"fixed_{d}")
        rule_result = summarize_rl_log(rule_log, name=f"rule_{d}")

        fixed_result["rl_date"] = d
        fixed_result["policy_type"] = "fixed"

        rule_result["rl_date"] = d
        rule_result["policy_type"] = "rule"

        results.extend([fixed_result, rule_result])

        fixed_logs.append(fixed_log)
        rule_logs.append(rule_log)
        fixed_solutions.append(fixed_sol)
        rule_solutions.append(rule_sol)

    return {
        "fixed_log_df": pd.concat(fixed_logs, axis=0, ignore_index=True) if len(fixed_logs) > 0 else pd.DataFrame(),
        "rule_log_df": pd.concat(rule_logs, axis=0, ignore_index=True) if len(rule_logs) > 0 else pd.DataFrame(),
        "fixed_solution_df": pd.concat(fixed_solutions, axis=0, ignore_index=True) if len(fixed_solutions) > 0 else pd.DataFrame(),
        "rule_solution_df": pd.concat(rule_solutions, axis=0, ignore_index=True) if len(rule_solutions) > 0 else pd.DataFrame(),
        "summary_df": pd.DataFrame(results),
    }

# 11. 主流程： df_final 自动切 train / valid / test
# ============================================================

def split_dates_by_time(
    all_dates,
    train_end="2026-05-31",
    valid_end="2026-06-04",
    train_ratio=0.70,
    valid_ratio=0.15,
):
    """
    按时间顺序切分日期。

    优先使用 train_end / valid_end：
        train: <= train_end
        valid: train_end < date <= valid_end
        test : > valid_end

    如果 train_end 或 valid_end 为 None，则按比例切分。
    """
    all_dates = sorted([str(d) for d in all_dates])

    if len(all_dates) == 0:
        raise ValueError("all_dates 为空，无法切分 train / valid / test")

    # 优先按指定日期切分
    if train_end is not None and valid_end is not None:
        train_dates = [d for d in all_dates if d <= train_end]
        valid_dates = [d for d in all_dates if train_end < d <= valid_end]
        test_dates = [d for d in all_dates if d > valid_end]

        if len(train_dates) == 0:
            raise ValueError(
                f"train_dates 为空。请检查 train_end={train_end} 是否早于数据最小日期 {all_dates[0]}"
            )
        if len(valid_dates) == 0:
            raise ValueError(
                f"valid_dates 为空。请检查 train_end={train_end}, valid_end={valid_end}"
            )
        if len(test_dates) == 0:
            raise ValueError(
                f"test_dates 为空。请检查 valid_end={valid_end} 是否晚于数据最大日期 {all_dates[-1]}"
            )

        return train_dates, valid_dates, test_dates

    # 如果不传日期边界，则按比例切
    n = len(all_dates)
    train_n = int(n * train_ratio)
    valid_n = int(n * valid_ratio)

    # 至少保证 train / valid / test 都有数据
    train_n = max(train_n, 1)
    valid_n = max(valid_n, 1)

    if train_n + valid_n >= n:
        train_n = max(1, n - 2)
        valid_n = 1

    train_dates = all_dates[:train_n]
    valid_dates = all_dates[train_n: train_n + valid_n]
    test_dates = all_dates[train_n + valid_n:]

    if len(test_dates) == 0:
        raise ValueError(
            f"按比例切分后 test_dates 为空。all_dates 数量={n}，请手动指定 train_dates / valid_dates / test_dates"
        )

    return train_dates, valid_dates, test_dates


def _concat_policy_summary(baseline_summary, dqn_summary, dqn_policy_type):
    """
    合并 fixed / rule / dqn 的 summary，方便比较。
    baseline_summary 来自 run_baselines_on_dates()["summary_df"]
    dqn_summary 来自 evaluate_agent_on_dates()["summary_df"]
    """
    frames = []

    if baseline_summary is not None and not baseline_summary.empty:
        frames.append(baseline_summary.copy())

    if dqn_summary is not None and not dqn_summary.empty:
        dqn_summary = dqn_summary.copy()
        dqn_summary["policy_type"] = dqn_policy_type
        frames.append(dqn_summary)

    if len(frames) == 0:
        return pd.DataFrame()

    compare_df = pd.concat(frames, axis=0, ignore_index=True)

    # 尽量把核心列提前
    preferred_cols = [
        "rl_date",
        "policy_type",
        "policy",
        "inc_call",
        "inc_trip",
        "inc_gmv",
        "inc_cost",
        "inc_gp",
        "inc_tr",
        "inc_cr",
        "inc_tr_weekend",
        "cr_opt",
        "cr_base",
        "total_reward",
        "total_call_opt",
        "total_trip_opt",
        "total_gmv_opt",
        "total_cost_opt",
        "total_gp_opt",
    ]

    ordered_cols = [c for c in preferred_cols if c in compare_df.columns]
    other_cols = [c for c in compare_df.columns if c not in ordered_cols]

    return compare_df[ordered_cols + other_cols]


def main_run_example(
    df_final,
    running_group=None,
    num_episodes=100,
    train_end="2026-03-21",
    valid_end="2026-03-26",
    train_dates=None,
    valid_dates=None,
    test_dates=None,
    keep_solution_test=True,
    keep_solution_valid=False,
    keep_solution_baseline=False,
):
    """
    适用于 df_final 是全量 3/1~3/31 数据的主流程。

    流程：
        1. prepare_rl_dataframe(df_final)
        2. 按 rl_date 切 train / valid / test
        3. 只用 train_dates 训练 DQN
        4. valid_dates 上评估 fixed / rule / dqn
        5. test_dates 上评估 fixed / rule / dqn
        6. 返回所有结果

    默认切分：
        train: 2026-03-01 ~ 2026-03-21
        valid: 2026-03-22 ~ 2026-03-26
        test : 2026-03-27 ~ 2026-03-31

    参数：
        running_group:
            如果不想过滤实验组，传 None。
            如果只想看某个组，比如 group_1_tf_sep，就传 "group_1_tf_sep"。

        keep_solution_test:
            是否保留测试集订单级 solution_df。
            测试集建议 True，方便分析每单 pax_opt / driver_opt。

        keep_solution_valid:
            是否保留验证集订单级 solution_df。
            数据量大时建议 False。

        keep_solution_baseline:
            是否保留 fixed / rule baseline 的订单级 solution_df。
            数据量大时建议 False。
    """

    # 1. 准备 30min RL 数据
    df_rl = prepare_rl_dataframe(df_final)

    all_dates = sorted(df_rl["rl_date"].unique().tolist())

    print("========== Data Info ==========")
    print("rl dates:", all_dates)
    print("n_dates:", len(all_dates))
    print("n_rows:", df_rl.shape[0])

    # 2. 日期切分
    if train_dates is None or valid_dates is None or test_dates is None:
        auto_train_dates, auto_valid_dates, auto_test_dates = split_dates_by_time(
            all_dates,
            train_end=train_end,
            valid_end=valid_end,
        )

        if train_dates is None:
            train_dates = auto_train_dates
        if valid_dates is None:
            valid_dates = auto_valid_dates
        if test_dates is None:
            test_dates = auto_test_dates

    train_dates = sorted([str(d) for d in train_dates])
    valid_dates = sorted([str(d) for d in valid_dates])
    test_dates = sorted([str(d) for d in test_dates])

    print("\n========== Date Split ==========")
    print("train_dates:", train_dates)
    print("valid_dates:", valid_dates)
    print("test_dates :", test_dates)

    if len(train_dates) == 0:
        raise ValueError("train_dates 为空")
    if len(valid_dates) == 0:
        raise ValueError("valid_dates 为空")
    if len(test_dates) == 0:
        raise ValueError("test_dates 为空")

    # 3. 训练 DQN：只用训练集
    print("\n========== Train DQN ==========")
    train_result = train_dqn_on_df(
        df_rl,
        train_dates=train_dates,
        running_group=None,
        num_episodes=300,
        epsilon_start=0.8,
        epsilon_end=0.05,
        epsilon_decay=0.99,
    )

    print("\ntraining episode log tail:")
    print(train_result["episode_log"].tail())

    agent = train_result["agent"]

    # 4. 验证集 baseline
    print("\n========== Valid Baseline: fixed / rule ==========")
    baseline_valid = run_baselines_on_dates(
        df_rl,
        dates=valid_dates,
        running_group=running_group,
        keep_solution=keep_solution_baseline,
    )

    print("\nvalid baseline summary:")
    print(baseline_valid["summary_df"])

    # 5. 验证集 DQN
    print("\n========== Valid DQN ==========")
    valid_result = evaluate_agent_on_dates(
        df_rl,
        agent,
        eval_dates=valid_dates,
        running_group=running_group,
        keep_solution=keep_solution_valid,
    )

    print("\nvalid dqn summary:")
    print(valid_result["summary_df"])

    valid_compare = _concat_policy_summary(
        baseline_valid.get("summary_df", pd.DataFrame()),
        valid_result.get("summary_df", pd.DataFrame()),
        dqn_policy_type="dqn_valid",
    )

    print("\nvalid compare:")
    print(valid_compare)

    # 6. 测试集 baseline
    print("\n========== Test Baseline: fixed / rule ==========")
    baseline_test = run_baselines_on_dates(
        df_rl,
        dates=test_dates,
        running_group=running_group,
        keep_solution=keep_solution_baseline,
    )

    print("\ntest baseline summary:")
    print(baseline_test["summary_df"])

    # 7. 测试集 DQN
    print("\n========== Test DQN ==========")
    test_result = evaluate_agent_on_dates(
        df_rl,
        agent,
        eval_dates=test_dates,
        running_group=running_group,
        keep_solution=keep_solution_test,
    )

    print("\ntest dqn summary:")
    print(test_result["summary_df"])

    test_compare = _concat_policy_summary(
        baseline_test.get("summary_df", pd.DataFrame()),
        test_result.get("summary_df", pd.DataFrame()),
        dqn_policy_type="dqn_test",
    )

    print("\ntest compare:")
    print(test_compare)

    # 8. 返回所有对象，方便 notebook 后续分析
    return {
        "df_rl": df_rl,

        "date_split": {
            "all_dates": all_dates,
            "train_dates": train_dates,
            "valid_dates": valid_dates,
            "test_dates": test_dates,
        },

        "train_result": train_result,

        "baseline_valid": baseline_valid,
        "valid_result": valid_result,
        "valid_compare": valid_compare,

        "baseline_test": baseline_test,
        "test_result": test_result,
        "test_compare": test_compare,
    }

result = main_run_example(
    df_final,
    running_group=None,
    num_episodes=200,
    train_end="2026-05-24",
    valid_end="2026-05-31",
)

episode_log = result["train_result"]["episode_log"].copy()
episode_log["reward_roll20"] = episode_log["episode_reward"].rolling(20).mean()
episode_log[["episode", "episode_reward", "reward_roll20", "epsilon"]].tail(50)




