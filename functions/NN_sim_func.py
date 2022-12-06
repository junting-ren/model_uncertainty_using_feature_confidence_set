import numpy as np
import torch
import bootstrap_func
import confidence_set_func
def transform_X(X):
    '''Transform the design matrix using sigmoid, cos, square function
    
    Parameters:
    -----------------
    X: the design matrix without intercpet
    
    Returns:
    ----------------
    A numpy matrix with rows as the sample and column as the feature. The number of feature is X.shape[0]*4+1.
    '''
    N = X.shape[0]
    # sigmoid
    sigmoid_X = 1/(1+np.exp(-X))
    # cosine
    cos_X = np.cos(X)
    # square
    square_X = np.square(X)
    X = np.concatenate((sigmoid_X, cos_X, square_X, X), axis = 1)
    return np.concatenate((np.ones((N, 1)), X), axis = 1)

def transform_X_poly(X):
    N = X.shape[0]
    # square
    square_X = np.square(X)
    # sqrt root
    root_X = np.sqrt(X)
    X = np.concatenate((root_X, square_X, X), axis = 1)
    return np.concatenate((np.ones((N, 1)), X), axis = 1)

def generate_NN_data(N, N_test, p,  transform_func, beta=None, error_sd = 3, seed = None, X = None, X_test = None, if_tensor = False):
    '''Generate neural network data
    
    Parameters:
    ----------------------------------------
    N: the number of training sample size
    N_test: the number of test sample size
    transform_func: the function to transform the design matrix to more complex design matrix with non-linear relationship  
    beta: the coefficients, defaults to None
    p: the number of features
    beta: the coefficients, defaults to None
    error_sd: the irreducible error standard deviation, defaults to 3
    seed: the random number generator seed
    X: the training design matrix, defaults to None
    X_test: the testing design matrix, defaults to None
    if_tensor: whether to transform the return values to tensor
    
    Returns:
    -----------------------------------
    A tuple containing the following:
        The original training design matrix with no transformation (do not include the intercept), # features = p
        The training outcome with irreducible error included
        The original testing design matrix with no transformation (do not include the intercept), # features = p
        The testing outcome with irreducible error included
        The testing true mean without irreducible error included
    '''
    rng = np.random.default_rng(seed)
    if beta is None:
        beta = rng.standard_normal(size = p*4+1)
    # Training dataset 
    if X is None:
        X = rng.standard_normal(size = (N,p))
    X_train_transformed = transform_func(X)
    y = X_train_transformed @ beta + rng.normal(loc = 0,scale = error_sd, size = (N,))
    # Test dataset
    if X_test is None:
        X_test = rng.standard_normal(size = (N_test,p))
    X_test_transformed = transform_func(X_test)
    mean_test_true = X_test_transformed @ beta
    y_test = mean_test_true + rng.normal(loc = 0,scale = error_sd, size = (N_test,))
    if if_tensor:
        y = torch.tensor(y).double()
        X = torch.tensor(X).double()
        X_test = torch.tensor(X_test).double()
    return X, y, X_test, y_test, mean_test_true

def sim_NN(N, N_test, p, error_sd, transform_func, 
           h_sizes, out_size, n_iter = 100, lr = 0.01, device = 'cpu', patience = 10, weight_decay = 0,
           L = 0.925, level= 0, use_true_contour = False, n_boot = 500, beta = None, X_test = None,  return_range = False):
    '''Function for neural network simulation
    
    '''
    
    X, y, X_test, y_test, mean_test_true = generate_NN_data(N = N, N_test = N_test, p = p, transform_func = transform_func, error_sd = error_sd, if_tensor = True)
    mean, se, mean_boot_l  = bootstrap_func.fit_bootstrap_NN(y, X, X_test, p, h_sizes, out_size,
                                              n_boot = n_boot, n_iter = n_iter, lr = lr, 
                                              device = device, patience = patience, weight_decay = weight_decay)
    mae = np.mean(np.abs(mean - mean_test_true))
    #import pdb; pdb.set_trace()
    if level is None:
        level = np.mean(mean)
    _, L, U, contain, contain_scb, contain_CS_scb, L1, L2, U1, U2, n_points, range_v,inner_points_num,outer_points_num,true_set_points_num = confidence_set_func.prediction_confidence_set(L, level, mean, se, mean_boot_l, mean_test_true = mean_test_true, use_true_contour = use_true_contour)
    p = X_test.shape[1]
    N_test = X_test.shape[0]
    if return_range:
        return contain, N, N_test, p, error_sd, level, use_true_contour, contain_scb, contain_CS_scb, L, U, L1, L2, U1, U2, n_points, mae, range_v, inner_points_num,outer_points_num,true_set_points_num
    else:
        return contain, N, N_test, p, error_sd, level, use_true_contour, contain_scb, contain_CS_scb, L, U, L1, L2, U1, U2, n_points, mae, inner_points_num,outer_points_num,true_set_points_num
    
def safe_sim_NN(N, N_test, p, error_sd, transform_func, 
                h_sizes, out_size, n_iter = 100, lr = 0.01, device = 'cpu', patience = 10, weight_decay = 0,
                L = 0.925, level= None, use_true_contour = False, n_boot = 500, beta = None, X_test = None,  return_range = False):
    try:
        return sim_NN(N, N_test, p, error_sd, transform_func, 
                      h_sizes, out_size, n_iter = n_iter, lr = lr, 
                      device = device, patience = patience, weight_decay = weight_decay,
                      L = L, level= level, use_true_contour = use_true_contour,
                      n_boot = n_boot, beta = beta, X_test = X_test,  return_range = return_range)
    except:
        return -1, N, N_test, p, error_sd, level, use_true_contour, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1,-1,-1