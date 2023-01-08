################################
# Functions for bootstrap
################################
import numpy as np
import neural_net_func 

def fit_bootstrap(fit_predict, y, X, X_test, n_boot = 200):
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
    mean = fit_predict(y, X, X_test)
    mean_boot_l = []
    for i in range(n_boot):
        index_boot= np.random.randint(X.shape[0], size=X.shape[0]) 
        y_boot = y[index_boot]
        X_boot = X[index_boot]
        mean_boot_l.append(fit_predict(y_boot, X_boot, X_test) )
    mean_boot_l = np.array(mean_boot_l)
    se = np.std(mean_boot_l, axis = 0)
    return mean, se, mean_boot_l


def fit_bootstrap_NN(y, X, X_test, input_size, h_sizes, out_size,
                  n_boot = 200, n_iter = 100, lr = 0.01, device = 'cpu', patience = 10, weight_decay = 0, batchnorm_ind = False):
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
        index_boot= np.random.randint(X.shape[0], size=X.shape[0]) 
        y_boot = y[index_boot]
        X_boot = X[index_boot]
        model = neural_net_func.FF_neural_net(input_size = input_size, h_sizes = h_sizes, out_size = out_size, batchnorm_ind = batchnorm_ind)
        mean_boot_l.append(neural_net_func.nn_fit_predict(model, y_boot, X_boot, X_test, n_iter = n_iter, lr = lr, device = device,  
                          patience = patience, weight_decay = weight_decay) )
    mean_boot_l = np.array(mean_boot_l)
    se = np.std(mean_boot_l, axis = 0)
    mean = np.mean(mean_boot_l, axis = 0)
    return mean, se, mean_boot_l    
