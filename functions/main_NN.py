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
    n_sim = 1
    N_v = [50,100,200]
    N_test_v = [500]
    p_v = [2,4]
    error_sd_v = [1]
    transform_func_v = [NN_sim_func.transform_X]
    h_sizes_v = [[10,50,10]]
    out_size_v = [1]
    n_iter_v = [100]
    lr_v = [0.01]
    device_v = ['cpu']
    patience_v = [10]
    weight_decay_v = [0]
    L_v = [0.95]
    level_v = [None]
    use_true_contour_v = [False]
    n_boot_v = [500]
    df_result = []
    param_grid = {'N': N_v, 'N_test': N_test_v, 'p': p_v , 'error_sd': error_sd_v, 
                  'transform_func':transform_func_v, 'h_sizes':h_sizes_v,
                  'out_size':out_size_v, 'n_iter':n_iter_v, 'lr':lr_v, 'device':device_v,
                  'patience':patience_v, 'weight_decay': weight_decay_v,
                  'L': L_v,'level': level_v,  'use_true_contour': use_true_contour_v, 'n_boot':n_boot_v}
    param_grid = ParameterGrid(param_grid)
    start = time.time()
    for param in param_grid:
        ctx = torch.multiprocessing.get_context('spawn')
        pool_obj = ctx.Pool()
        cur_para = (n_sim*((param['N'], param['N_test'], param['p'], param['error_sd'], \
                            param['transform_func'], param['h_sizes'],param['out_size'],
                            param['n_iter'],param['lr'],param['device'],param['patience'],param['weight_decay'],
                          param['L'],param['level'],param['use_true_contour'], param['n_boot']),))
        df_result.append(np.array(pool_obj.starmap(NN_sim_func.safe_sim_NN,cur_para)))
        pool_obj.close()
        pool_obj.join()
        net_parameters = 'hidden='+str(param['h_sizes'])+'_n_iter='+str(param['n_iter'])+'_lr='+str(param['lr'])+'_weight_decay='+str(param['weight_decay'])
        df_ = pd.DataFrame(np.concatenate(df_result, axis = 0), 
                     columns = ["contain", "N", "N_test",
                                "p", "error_sd", "level", "use_true_contour",
                                "contain_scb","contain_CS_scb","Lower_bound",
                                "Upper_bound", "L1", "L2", "U1", "U2", "points_in_e1", 
                                "pre_mae", 'inner_points_num', 'outer_points_num' ,'true_set_points_num'])
        df_ = pd.concat([pd.DataFrame([net_parameters]*df_.shape[0]), df_], axis = 1)
        df_total = pd.concat([df_total,df_])
        df_total.to_csv(file_name, index = False)
    end = time.time()
    print('Runtime of the program is ' + str(end -start) + ' seconds')