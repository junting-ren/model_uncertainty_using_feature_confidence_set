##############################################
# Functions for linear regression simulations
##############################################

import numpy as np
from scipy.stats import t
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
import pandas as pd
import time
# custom script functions
import confidence_set_func
import bootstrap_func
from scipy import stats

def generate_linear_data(N,N_test = None, beta=None, p = None, error_sd = 3, seed = None, X = None, X_test = None, if_tensor = False):
    rng = np.random.default_rng(seed)
    if beta is None:
        beta = rng.standard_normal(size = p)
    p = len(beta)-1
    # Training dataset 
    if X is None:
        X = rng.standard_normal(size = (N,p))
    X = np.concatenate((np.ones((N, 1)), X), axis = 1)
    y = X @ beta + rng.normal(loc = 0,scale = error_sd, size = (N,))
    # Test dataset
    if X_test is None:
        X_test = rng.standard_normal(size = (N_test,p))
        X_test = np.concatenate((np.ones((N_test, 1)), X_test), axis = 1)
        mean_test_true = X_test @ beta
    else:
        N_test = X_test.shape[0]
        X_test = np.concatenate((np.ones((N_test, 1)), X_test), axis = 1)
        mean_test_true = X_test @ beta
    # delete the intercept
    X = np.delete(X,0, axis = 1)
    X_test = np.delete(X_test, 0, axis = 1)
    return y,X, X_test, mean_test_true

def linear_fit_predict(y, X, X_test, return_X_predict = False):
    #import pdb; pdb.set_trace()
    reg = LinearRegression().fit(X, y)
    if return_X_predict:
        return reg.predict(X_test), reg.predict(X)
    else:
        return  reg.predict(X_test)

def ridge_linear_fit_predict(y, X, X_test, penal_size, return_X_predict = False):
    #reg = RidgeCV(alphas = penal_sizes, store_cv_values = True).fit(X, y)
    reg = Ridge(alpha = penal_size).fit(X, y)
    if return_X_predict:
        return reg.predict(X_test), reg.predict(X)
    else:
        return  reg.predict(X_test)

def sim_linear(N, N_test, p, error_sd, L = 0.925, level= None, use_true_contour = False, n_boot = 500, ridge_penal = None, MC = False, second_stage = False, center_G = False,center_pred = False, residual_boot = False, CV_boot = False, beta = None, X_test = None,  return_range = False):
    if MC:
        mean_boot_l = []
        if beta is None:
            rng = np.random.default_rng(None)
            beta = rng.standard_normal(size = p)
        y,X, X_test, mean_test_true = generate_linear_data(N = N, N_test = N_test, p = p,beta = beta, error_sd = error_sd, X_test = X_test)
        if ridge_penal is not None:
            clf = RidgeCV(alphas = ridge_penal, store_cv_values = True).fit(X, y)
            errors = np.mean(clf.cv_values_, axis = 0)
            best_penal = ridge_penal[np.argmin(errors)]
        for i in range(n_boot):
            y_new,X_new, X_test_trash, mean_test_true_trash = generate_linear_data(N = N, N_test = N_test, p = p,beta = beta, error_sd = error_sd, X_test = X_test)
            if ridge_penal is not None:
                if CV_boot:
                    clf = RidgeCV(alphas = ridge_penal, store_cv_values = True).fit(X_new, y_new)
                    mean_boot_l.append(clf.predict(X_test))
                else:
                    mean_boot_l.append(ridge_linear_fit_predict(y_new, X_new, X_test, best_penal) )
            else:# linear model without penality
                mean_boot_l.append(linear_fit_predict(y_new, X_new, X_test) )
        mean_boot_l = np.array(mean_boot_l)
        se = np.std(mean_boot_l, axis = 0)
        mean = ridge_linear_fit_predict(y, X, X_test, best_penal) if ridge_penal is not None else linear_fit_predict(y, X, X_test) 
    else:
        y,X, X_test, mean_test_true = generate_linear_data(N = N, N_test = N_test, p = p,beta = beta, error_sd = error_sd, X_test = X_test)
        if ridge_penal is not None:
            mean, se, mean_boot_l, mean_train = bootstrap_func.fit_bootstrap_ridge(y = y, 
                                                  X = X, X_test = X_test, n_boot = n_boot,  
                                                penal_sizes = ridge_penal, 
                                                residual_boot= residual_boot, CV_boot = CV_boot)
        else:# linear model without penality
            mean, se, mean_boot_l, mean_train = bootstrap_func.fit_bootstrap(y = y, 
                                                  X = X, X_test = X_test, n_boot = n_boot, residual_boot = residual_boot)
    mae = np.mean(np.abs(mean - mean_test_true))
    mse = np.mean((mean - mean_test_true)**2)
    #import pdb; pdb.set_trace()
    G, mean, se =confidence_set_func.process_boot_samples(mean_boot_l, mean, se, mean_test_true, 
                                      MC, second_stage, center_G, center_pred)
    _, L, U, contain, contain_scb, contain_CS_scb, L1, L2, U1, U2, n_points, range_v,inner_points_num,outer_points_num,true_set_points_num = confidence_set_func.prediction_confidence_set(L, level, mean, se, G, mean_test_true = mean_test_true, use_true_contour = use_true_contour)
    p = X_test.shape[1]
    N_test = X_test.shape[0]
    if return_range:
        return contain, N, N_test, p, error_sd, level, use_true_contour, contain_scb, contain_CS_scb, L, U, L1, L2, U1, U2, n_points, mae, mse, range_v, inner_points_num,outer_points_num,true_set_points_num
    else:
        return contain, contain_scb, contain_CS_scb, L, U, L1, L2, U1, U2, n_points, mae, mse, inner_points_num,outer_points_num,true_set_points_num
    
def safe_sim_linear(N, N_test, p, error_sd, L = 0.925, level= None, use_true_contour = False, n_boot = 500, ridge_penal = None, MC = False, second_stage = False, center_G = False,center_pred = False, residual_boot = False, CV_boot = False, beta = None, X_test = None,  return_range = False):
    try:
        return sim_linear(N, N_test, p, error_sd, L, level, use_true_contour, n_boot, ridge_penal, MC, second_stage, center_G,center_pred,residual_boot, CV_boot ,beta, X_test,  return_range)
    except:
        return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1,-1,-1, -1

    
def CS_t_test(inner_index, outer_index, X_test):
    n_inner = np.sum(inner_index)
    n_comp_outer = np.sum(~outer_index)
    mean1 = np.mean(X_test[inner_index], axis = 0) 
    mean2 = np.mean(X_test[~outer_index], axis = 0)
    sd1 = np.std(X_test[inner_index], axis = 0)
    sd2 = np.std(X_test[~outer_index], axis = 0)
    var_mean1 = sd1**2/n_inner
    var_mean2 = sd2**2/n_comp_outer
    t = (mean1 - mean2 )/np.sqrt(var_mean1+var_mean2)
    v = (var_mean1+var_mean2)**2/(var_mean1/(n_inner-1)+var_mean2/(n_comp_outer-1))
    t_q = stats.t.ppf(0.975, v)
    return t, t_q

def sim_interpretation(N ,N_test, level, error_sd, p, p_causal, L, X_test = None, n_boot = 500 ):
    beta = np.zeros(p)
    index_causal = np.random.choice(p-1,size = p_causal)
    beta[index_causal+1] = np.random.normal(size = p_causal)
    y,X, X_test, mean_test_true = generate_linear_data(N = N, N_test = N_test, p = p,beta = beta, error_sd = error_sd, X_test = X_test)
    mean, se, mean_boot_l = bootstrap_func.fit_bootstrap(fit_predict = linear_fit_predict, y = y, 
                                      X = X, X_test = X_test, n_boot = n_boot)
    result_dict, L, U, contain, contain_scb, contain_CS_scb, L1, L2, U1, U2, n_points, range_v,inner_points_num,outer_points_num,true_set_points_num = confidence_set_func.prediction_confidence_set(L, level, mean, se, mean_boot_l, mean_test_true = mean_test_true, use_true_contour = False)
    t, t_q = CS_t_test(result_dict['inner'], result_dict['outer'], X_test)
    significant_covar = np.where(np.abs(t)>t_q)[0] 
    all_causal_covariate_ind = np.all(np.isin(significant_covar, index_causal))
    percent_causal_covariate = np.sum(np.isin(significant_covar, index_causal))/len(index_causal)
    return contain, all_causal_covariate_ind, percent_causal_covariate, N, N_test, p, p_causal, error_sd, level, L
