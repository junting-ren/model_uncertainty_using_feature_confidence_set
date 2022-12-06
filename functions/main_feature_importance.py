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
file_name = 'sim_results_interpretation'+date+'.csv'
if __name__ ==  '__main__': 
    df_total = pd.DataFrame()
    n_sim = 500
    N_v = [50,100,200]
    N_test_v = [200,500,1000]
    level_v = [0]
    error_sd_v = [1]
    p_v = [50]
    p_causal_v = [20]
    L_v = [0.95]
    X_test_v = [None]
    n_boot_v = [500]
    df_result = []
    param_grid = {'N': N_v, 'N_test': N_test_v, 'level': level_v, 'error_sd': error_sd_v, 'p': p_v , 'p_causal': p_causal_v, 'L': L_v,
                  'X_test':X_test_v, 'n_boot':n_boot_v}
    param_grid = ParameterGrid(param_grid)
    start = time.time()
    for param in param_grid:
        pool_obj = multiprocessing.Pool()
        cur_para = (n_sim*((param['N'], param['N_test'], param['level'], param['error_sd'], \
                          param['p'],param['p_causal'],param['L'], param['X_test'], param['n_boot']),))
        df_result.append(np.array(pool_obj.starmap(linear_sim_func.sim_interpretation,cur_para)))
        pool_obj.close()
        pool_obj.join()
        df_ = pd.DataFrame(np.concatenate(df_result, axis = 0), 
                     columns = ["contain", "all_causal", "percent_causal_selected", "N", "N_test",
                                "p", "p_causal","error_sd", "level", "L"])
        df_total = pd.concat([df_total,df_])
        df_total.to_csv(file_name, index = False)
    end = time.time()
    print('Runtime of the program is ' + str(end -start) + ' seconds')