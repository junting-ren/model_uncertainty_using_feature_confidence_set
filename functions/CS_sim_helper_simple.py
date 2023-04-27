from confidence_set_func import prediction_confidence_set, multiple_testing_confidence_set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *


def generate_sim_data(N, p, p_null, rho, effect_size, level,
                      seed = None, variance = 1):
    '''Generate data for simulation
    
    Parameters:
    ----------------------------------------
    N: the number of sample size
    p: the number of parameters
    rho: correlation parameter for Compound Symmetry correlation matrix
    p_null: number of null parameters
    effect_size: the effect size when not under the null
    seed: random seed for data generation
    variance: variance for the mean we are simulating from under the alternative
    c: the null distribution mean
    
    Returns:
    -----------------------------------
    A tuple containing the following:
        the sampled parameters 
        the true parameter mean
    '''
    rng = np.random.default_rng(seed)
    corr_matrix = np.ones((p, p))*rho
    np.fill_diagonal(corr_matrix, 1)
    # Compound symmetry covariance matrix
    cov_matrix = corr_matrix*variance
    mean_true = np.array([level]*p, dtype = 'float')
    index_alt = rng.choice(p,p-p_null, replace = False)
    #import pdb;pdb.set_trace()
    mean_true[index_alt] = level+np.where(rng.binomial(1,p=0.5, size = p-p_null)==1, 1,-1)*effect_size
    #np.put(mean_true, index_alt, np.array([c])+np.where(rng.binomial(1,p=0.5, size = p-p_null)==1, 1,-1)*effect_size)
    X = rng.multivariate_normal(mean_true, cov_matrix, size = N)
    return X, mean_true

def bootstrap_simple(X, n_boot):
    N = X.shape[0]
    G_l = []
    point_est = np.mean(X, axis = 0)
    for i in range(n_boot):
        #import pdb; pdb.set_trace()
        index_boot= np.random.randint(X.shape[0], size=int(X.shape[0]))
        X_boot = X[index_boot]
        G_l.append(np.sqrt(N)*(np.mean(X_boot, axis = 0) - point_est)/np.std(X_boot, axis = 0))
    #import pdb; pdb.set_trace()
    G = np.array(G_l)
    #point_est = np.mean(mean_boot_l, axis = 0)
    #import pdb; pdb.set_trace()
    #import pdb; pdb.set_trace()
    se = np.std(X, axis = 0)/np.sqrt(N)
    return point_est, G, se

def bootstrap_and_CS(L, level,
                     X, mean_true = None, 
                     use_true_contour = False,  n_boot = 200):
    #import pdb;pdb.set_trace()
    N = X.shape[0]
    point_pred, G, se = bootstrap_simple(X, n_boot = n_boot)
    #import pdb; pdb.set_trace()
    # construct confidence set
    (df_res, result_dict) = \
    prediction_confidence_set(L, level, point_pred, se, G, mean_test_true = mean_true, use_true_contour = use_true_contour, test_null = True)
    result_dict['method'] = 'CS_on_mean'
    # maxT step down for constructing confidence set
    multiple_testing = multiple_testing_confidence_set(L, level, point_pred, se, G, mean_true)
    _,result_dict_T = multiple_testing.maxT_step_down_confidence_set()
    result_dict_T['method'] = 'step_maxT_on_mean'
    _,result_dict_Bonf = multiple_testing.Bonf_confidence_set()
    result_dict_Bonf['method'] = 'Bonf_on_mean'
    _,result_dict_Holm = multiple_testing.Holm_confidence_set()
    result_dict_Holm['method'] = 'Holm_on_mean'
    r = [result_dict,result_dict_T,result_dict_Bonf,result_dict_Holm]
    return pd.DataFrame.from_dict(r)

def sim_CS(L, level, N, p_p_null, rho, effect_size, data_kwargs,
           use_true_contour = False,  n_boot = 200):
    p, p_null = p_p_null[0],p_p_null[1]
    # data simulation
    X, mean_true = generate_sim_data(N, p, p_null, rho, effect_size, level, **data_kwargs)
    r = bootstrap_and_CS(L, level, 
                         X, mean_true = mean_true, 
                         use_true_contour = use_true_contour,
                         n_boot = n_boot
                        )
    return r
        
def sim_CS_wrapper(kwargs):
    return sim_CS(**kwargs)
        


