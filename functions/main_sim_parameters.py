import os
import multiprocessing
import numpy as np
import pandas as pd
import time
from datetime import date
import torch
# custom functions
from CS_sim_helper_simple import sim_CS_wrapper, generate_sim_data
from sklearn.model_selection import ParameterGrid

today = date.today()
date = today.strftime("%d%m%Y")
file_name = 'testing_sim_'+date+'.csv'
folder_path = './'#test_sim_result
file_name = os.path.join(folder_path, file_name)
if __name__ ==  '__main__': 
    df_total = pd.DataFrame()
    n_sim = 500
    L_v = [0.95]
    level_v = [0]
    N_v = [100]
    p_p_null_v = [(100,50)]
    rho_v = [0.]
    effect_size_v = [1]
    data_kwargs_v = [{'seed': None, 'variance': 1}]
    use_true_contour_v = [False]
    n_boot_v = [500]
    df_result = []
    param_grid = {'L': L_v, 'level': level_v, 
                  'N': N_v, 'p_p_null': p_p_null_v, 
                  'rho':rho_v, 'effect_size':effect_size_v,
                  'data_kwargs':data_kwargs_v, 'use_true_contour': use_true_contour_v, 'n_boot':n_boot_v
                 }
    param_grid = ParameterGrid(param_grid)
    start = time.time()
    for param in param_grid:
        para_name_dict = {k:str(v) for k, v in param.items()}
        df = pd.DataFrame(para_name_dict, index = [0])
        print(para_name_dict)
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