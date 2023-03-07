################################
# Functions for bootstrap
################################
import numpy as np
import neural_net_func 
from sklearn.linear_model import RidgeCV
import linear_sim_func


def fit_bootstrap(y, X, X_test, n_boot = 200, residual_boot = False, boot_sample_ratio = 1, pred_on_y = False):
    '''Function for fittting the model and bootstrap 
    Parameters:
    ---------------
    fit_predict: function for fitting and predicting by taking in y, X, X_test
    X: matrix of features
    y: vector of outcomes
    
    Return:
    --------------
    a tuple contains:
        predicted mean on the full data, numpy array of (X_test.shape[0],)
        se for the predicted mean on the full data: basically tau_n*se that shrinks as sample size increases, numpy array of (X_test.shape[0],)
        a matrix containing the bootstraped predicted mean, with rows indicting the sample index, 
        the columns indicating the bootstrap index, numpy array of (n_boot, X_test.shape[0])
    '''
    mean_boot_l = []
    for i in range(n_boot):
        index_boot= np.random.randint(X.shape[0], size=int(X.shape[0]*boot_sample_ratio))
        #import pdb;pdb.set_trace()
        index_val = np.array(list(range(0,X.shape[0])))[~np.isin(list(range(0,X.shape[0])), index_boot)]
        if residual_boot:
            residuals = y - mean_train
            y_boot = mean_train + residuals[index_boot]
            X_boot = X
        else:
            y_boot = y[index_boot]
            X_boot = X[index_boot]
            X_val = X[index_val]
            y_val = y[index_val]
        if pred_on_y:
            preds, pred_train_boot = linear_sim_func.linear_fit_predict(y_boot, X_boot, 
                                                                   [X_test, X_val], 
                                                                   return_X_predict = True) 
            #err = np.concatenate([preds[1]-y_val, pred_train_boot - y_boot])
            err = preds[1]-y_val
            temp = preds[0] + err[np.random.randint(0, len(err), size = (len(preds[0]),) )]
            mean_boot_l.append(temp)
        else:
            mean_boot_l.append(linear_sim_func.linear_fit_predict(y_boot, X_boot, X_test) )
    mean_boot_l = np.array(mean_boot_l)
    mean, mean_train = linear_sim_func.linear_fit_predict(y, X, X_test, return_X_predict = True)
    se = np.std(mean_boot_l, axis = 0)
    return mean, se, mean_boot_l,mean_train


def fit_bootstrap_ridge(y, X, X_test, n_boot = 200,  penal_sizes = [1,0.5,1e-1, 1e-2, 1e-3, 1e-4],residual_boot = False, CV_boot = False, boot_sample_ratio = 1):
    clf = RidgeCV(alphas = penal_sizes, store_cv_values = True).fit(X, y)
    errors = np.mean(clf.cv_values_, axis = 0)
    penal_size = penal_sizes[np.argmin(errors)]
    mean_boot_l = []
    for i in range(n_boot):
        index_boot= np.random.randint(X.shape[0], size=int(X.shape[0]*boot_sample_ratio))
        if residual_boot:
            residuals = y - mean_train
            y_boot = mean_train + residuals[index_boot]
            X_boot = X
        else:
            y_boot = y[index_boot]
            X_boot = X[index_boot]
        if CV_boot:
            clf = RidgeCV(alphas = penal_sizes, store_cv_values = True).fit(X_boot, y_boot)
            mean_boot_l.append(clf.predict(X_test))
        else:
            mean_boot_l.append(linear_sim_func.ridge_linear_fit_predict(y_boot, X_boot, X_test, penal_size) )
    mean_boot_l = np.array(mean_boot_l)
    se = np.std(mean_boot_l, axis = 0)
    mean, mean_train= linear_sim_func.ridge_linear_fit_predict(y, X, X_test, penal_size, return_X_predict = True)
    return mean, se, mean_boot_l, mean_train

def fit_bootstrap_NN(y, X, X_test, input_size, h_sizes, out_size,
                  n_boot = 200, n_iter = 100, lr = 0.01, device = 'cpu', 
                     patience = 10, weight_decay = 0, batchnorm_ind = False,
                    mean_num = 1, return_best = False, pred_on_y = False):
    '''Function for fittting the model and bootstrap 
    Parameters:
    ---------------
    fit_predict: function for fitting and predicting by taking in y, X, X_test
    X: matrix of features
    y: vector of outcomes
    mean_num: number of prediction samples during the gradient descend epoch to take average over for the final prediction.
    
    Return:
    --------------
    a tuple contains:
        predicted mean on the full data, numpy array of (X_test.shape[0],)
        se for the predicted mean on the full data: basically tau_n*se that shrinks as sample size increases, numpy array of (X_test.shape[0],)
        a matrix containing the bootstraped predicted mean, with rows indicting the sample index, 
        the columns indicating the bootstrap index, numpy array of (n_boot, X_test.shape[0])
    '''
    mean_boot_l = []
    model = neural_net_func.FF_neural_net(input_size = input_size, h_sizes = h_sizes, out_size = out_size, batchnorm_ind = batchnorm_ind)
    mean, best_model,val_error = neural_net_func.nn_fit_predict(model, y, X, X_test, n_iter = n_iter, lr = lr, device = device,  
                          patience = patience, weight_decay = weight_decay, mean_num = mean_num, return_best = True) 
    mean_train = best_model.forward(X.to(device)).squeeze(1).cpu().detach().numpy()
    for i in range(n_boot):
        index_boot= np.random.randint(X.shape[0], size=X.shape[0]) 
        index_val = np.array(list(range(0,X.shape[0])))[~np.isin(list(range(0,X.shape[0])), index_boot)]
        y_boot = y[index_boot]
        X_boot = X[index_boot]
        model = neural_net_func.FF_neural_net(input_size = input_size, h_sizes = h_sizes, out_size = out_size, batchnorm_ind = batchnorm_ind)
        mean_pred_boot, best_model_BOOT, best_error_boot = neural_net_func.nn_fit_predict(model, y_boot, 
                                                          X_boot, X_test, n_iter = n_iter, lr = lr, device = device,  
                          patience = patience, weight_decay = weight_decay, mean_num = mean_num, return_best = True) 
        if pred_on_y:
            #import pdb; pdb.set_trace()
            #err = np.concatenate([preds[1]-y_val, pred_train_boot - y_boot])
            best_error_boot = best_error_boot - np.mean(best_error_boot)
            temp = mean_pred_boot + best_error_boot[np.random.randint(0, len(best_error_boot), size = (len(mean_pred_boot),) )]
            mean_boot_l.append(temp)
        else:
            mean_boot_l.append(mean_pred_boot)
    mean_boot_l = np.array(mean_boot_l)
    se = np.std(mean_boot_l, axis = 0)
    if return_best:
        return mean, se, mean_boot_l, mean_train, best_model
    else:
        return mean, se, mean_boot_l, mean_train  
