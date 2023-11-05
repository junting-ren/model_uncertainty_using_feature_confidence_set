import os
import multiprocessing
import numpy as np
import pandas as pd
import time
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import torch
# custom functions
from CS_sim_helper import sim_CS_wrapper, generate_sim_data,check_overfitting, transform_X_poly,transform_X
from models import NueralNet, logistic_regression

today = date.today()
date = today.strftime("%d%m%Y")
file_name = 'linear_regression_'+date+'.csv'
folder_path = './linear_result'
file_name = os.path.join(folder_path, file_name)
if __name__ ==  '__main__': 
    df_total = pd.DataFrame()
    n_sim = 500
    L_v = [0.9, 0.6]
    level_v = [2]
    models_l_v = [[LinearRegression]]
    models_kwargs_l_v = [[{}]]
    N_v = [100, 200, 400, 800]
    N_test_v = [500]
    p_v = [3, 6, 10]
    error_sd_v = [1]
    data_sim_func_v = [generate_sim_data]
    data_kwargs_v = [{'uniform_range_x':[-2,2]}]
    center_G_v = [True]
    center_pred_v = [False]
    use_true_contour_v = [False]
    n_boot_v = [200]
    df_result = []
    param_grid = {'L': L_v, 'level': level_v, 'models_l': models_l_v,'model_kwargs_l':models_kwargs_l_v,
                  'N': N_v, 'N_test': N_test_v, 'p': p_v , 'error_sd': error_sd_v, 
                  'data_sim_func':data_sim_func_v, 'data_kwargs':data_kwargs_v,
                  'center_G':center_G_v, 'center_pred':center_pred_v, 'use_true_contour': use_true_contour_v, 'n_boot':n_boot_v
                 }
    param_grid = ParameterGrid(param_grid)
    start = time.time()
    for param in param_grid:
        para_name_dict = {k:str(v) for k, v in param.items()}
        df = pd.DataFrame(para_name_dict, index = [0])
        ctx = torch.multiprocessing.get_context('spawn')
        pool_obj = ctx.Pool()
        # pool_obj = multiprocessing.Pool()
        cur_para = (n_sim*(param,))
        df_ = pd.concat(pool_obj.map(sim_CS_wrapper, cur_para)).reset_index(drop = True)
        pool_obj.close()
        pool_obj.join()
        para_df = pd.DataFrame(np.repeat(df.values,df_.shape[0], axis=0))
        para_df.columns = df.columns
        df_ = pd.concat([para_df, df_], axis = 1)
        df_total = pd.concat([df_total,df_])
        df_total.to_csv(file_name, index = False)
    end = time.time()
    print('Runtime of the program is ' + str(end -start) + ' seconds')