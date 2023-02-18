import multiprocessing
import numpy as np
import pandas as pd
import time
from datetime import date
from sklearn.model_selection import ParameterGrid
import torch
import NN_sim_func
# custom functions

today = date.today()
date = today.strftime("%d%m%Y")
file_name = 'sim_NN_results_no_penal_mean_predict'+date+'.csv'
if __name__ ==  '__main__': 
    df_total = pd.DataFrame()
    n_sim = 200
    N_v = [100]
    N_test_v = [500]
    p_v = [1]
    error_sd_v = [1]
    transform_func_v = [NN_sim_func.transform_X_poly]
    h_sizes_v = [[5,5]]
    out_size_v = [1]
    n_iter_v = [200]
    lr_v = [0.01]
    device_v = ['cpu']
    patience_v = [20]
    weight_decay_v = [0]
    L_v = [0.9]
    level_v = [None]
    use_true_contour_v = [False]
    n_boot_v = [10]
    MC_v = [False]
    second_stage_v = [True, False]
    center_G_v = [True, False]
    center_pred_v = [True,False]
    mean_num_v = [1]
    uniform_range_v = [[-2,2], None]
    beta_v = [np.array([1,2,1])]
    df_result = []
    param_grid = {'N': N_v, 'N_test': N_test_v, 'p': p_v , 'error_sd': error_sd_v, 
                  'transform_func':transform_func_v, 'h_sizes':h_sizes_v,
                  'out_size':out_size_v, 'n_iter':n_iter_v, 'lr':lr_v, 'device':device_v,
                  'patience':patience_v, 'weight_decay': weight_decay_v,
                  'L': L_v,'level': level_v,  'use_true_contour': use_true_contour_v, 'n_boot':n_boot_v,
                  'MC':MC_v, 'second_stage':second_stage_v, 'center_G':center_G_v, 'center_pred':center_pred_v,
                  'mean_num':mean_num_v, 'uniform_range':uniform_range_v, 'beta':beta_v
                 }
    param_grid = ParameterGrid(param_grid)
    start = time.time()
    for param in param_grid:
        para_name_dict = {k:str(v) for k, v in param.items()}
        df = pd.DataFrame(para_name_dict, index = [0])
        para_df = pd.DataFrame(np.repeat(df.values,n_sim, axis=0))
        para_df.columns = df.columns
        ctx = torch.multiprocessing.get_context('spawn')
        pool_obj = ctx.Pool()
        cur_para = (n_sim*((param['N'], param['N_test'], param['p'], param['error_sd'], \
                            param['transform_func'], param['h_sizes'],param['out_size'],
                            param['n_iter'],param['lr'],param['device'],param['patience'],param['weight_decay'],
                          param['L'],param['level'],param['use_true_contour'], param['n_boot'], 
                           param['MC'],param['second_stage'],param['center_G'],param['center_pred'],
                            param['mean_num'], param['uniform_range'], param['beta']
                           ),))
        df_ = pd.DataFrame(np.array(pool_obj.starmap(NN_sim_func.safe_sim_NN,cur_para)), 
                     columns = ["contain", "contain_scb","contain_CS_scb","Lower_bound",
                                "Upper_bound", "L1", "L2", "U1", "U2", "points_in_e1", 
                                "pre_mae", 'pre_mse','inner_points_num', 'outer_points_num' ,'true_set_points_num'])
        pool_obj.close()
        pool_obj.join()
        df_ = pd.concat([para_df, df_], axis = 1)
        df_total = pd.concat([df_total,df_])
        df_total.to_csv(file_name, index = False)
    end = time.time()
    print('Runtime of the program is ' + str(end -start) + ' seconds')