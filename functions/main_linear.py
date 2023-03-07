import multiprocessing
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import multiprocessing
import time
from datetime import date
from sklearn.model_selection import ParameterGrid
# custom functions
import linear_sim_func

today = date.today()
date = today.strftime("%d%m%Y")
file_name = 'sim_pred_y'+date+'.csv'
if __name__ ==  '__main__': 
    df_total = pd.DataFrame()
    n_sim = 200
    N_v = [100,200]
    N_test_v = [500]
    p_v = [9]
    error_sd_v = [1]
    L_v = [0.9]
    level_v = [0.2]
    use_true_contour_v = [False]
    n_boot_v = [500]
    #ridge_penal_v = [[1,0.5,1e-1, 1e-2, 1e-3, 1e-4]]
    ridge_penal_v = [None]
    MC_v = [False]
    second_stage_v = [False]
    center_G_v = [True]
    center_pred_v = [False]
    residual_boot_v = [False]
    CV_boot_v = [False]
    pred_on_y_v = [True]
    df_result = []
    param_grid = {'N': N_v, 'N_test': N_test_v, 'p': p_v , 'error_sd': error_sd_v, 'L': L_v,
                  'level': level_v,  'use_true_contour': use_true_contour_v, 'n_boot':n_boot_v,
                  'ridge_penal':ridge_penal_v, 'MC':MC_v, 'second_stage':second_stage_v,
                  'center_G':center_G_v, 'center_pred':center_G_v,
                  'residual_boot':residual_boot_v, 'CV_boot':CV_boot_v, 'pred_on_y':pred_on_y_v}
    param_grid = ParameterGrid(param_grid)
    start = time.time()
    for param in param_grid:
        para_name_dict = {k:str(v) for k, v in param.items()}
        df = pd.DataFrame(para_name_dict, index = [0])
        para_df = pd.DataFrame(np.repeat(df.values,n_sim, axis=0))
        para_df.columns = df.columns
        pool_obj = multiprocessing.Pool()
        cur_para = (n_sim*((param['N'], param['N_test'], param['p'], param['error_sd'], \
                          param['L'],param['level'],param['use_true_contour'], param['n_boot'], 
                            param['ridge_penal'], param['MC'], param['second_stage'], 
                            param['center_G'], param['center_pred'], 
                            param['residual_boot'], param['CV_boot'], param['pred_on_y']),))
        df_ = pd.DataFrame(np.array(pool_obj.starmap(linear_sim_func.safe_sim_linear,cur_para)), 
                     columns = ["contain","contain_scb","contain_CS_scb","Lower_bound",
                                "Upper_bound", "L1", "L2", "U1", "U2", "points_in_e1", 
                                "pre_mae", 'pre_mse','inner_points_num', 'outer_points_num' ,'true_set_points_num'])
        pool_obj.close()
        pool_obj.join()
        df_ = pd.concat([para_df, df_], axis = 1)
        df_total = pd.concat([df_total,df_])
        df_total.to_csv(file_name, index = False)
    end = time.time()
    print('Runtime of the program is ' + str(end -start) + ' seconds')