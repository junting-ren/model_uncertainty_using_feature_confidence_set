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
file_name = 'sim_results_ridge'+date+'.csv'
if __name__ ==  '__main__': 
    df_total = pd.DataFrame()
    n_sim = 500
    N_v = [50,150,250]
    N_test_v = [500]
    p_v = [160]
    error_sd_v = [1,4]
    L_v = [0.95]
    level_v = [0.2]
    use_true_contour_v = [False]
    n_boot_v = [2000]
    ridge_v = [True]
    df_result = []
    param_grid = {'N': N_v, 'N_test': N_test_v, 'p': p_v , 'error_sd': error_sd_v, 'L': L_v,
                  'level': level_v,  'use_true_contour': use_true_contour_v, 'n_boot':n_boot_v,'ridge':ridge_v}
    param_grid = ParameterGrid(param_grid)
    start = time.time()
    for param in param_grid:
        df = pd.DataFrame(param, index = [0])
        para_df = pd.DataFrame(np.repeat(df.values,n_sim, axis=0))
        pool_obj = multiprocessing.Pool()
        cur_para = (n_sim*((param['N'], param['N_test'], param['p'], param['error_sd'], \
                          param['L'],param['level'],param['use_true_contour'], param['n_boot'], param['ridge']),))
        df_result.append(np.array(pool_obj.starmap(linear_sim_func.safe_sim_linear,cur_para)))
        pool_obj.close()
        pool_obj.join()
        df_ = pd.DataFrame(np.concatenate(df_result, axis = 0), 
                     columns = ["contain","contain_scb","contain_CS_scb","Lower_bound",
                                "Upper_bound", "L1", "L2", "U1", "U2", "points_in_e1", 
                                "pre_mae", 'inner_points_num', 'outer_points_num' ,'true_set_points_num'])
        df_ = pd.concat([para_df, df_], axis = 1)
        df_total = pd.concat([df_total,df_])
        df_total.to_csv(file_name, index = False)
    end = time.time()
    print('Runtime of the program is ' + str(end -start) + ' seconds')