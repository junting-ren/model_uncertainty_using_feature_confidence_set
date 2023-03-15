from CS_toolbox import bootstrap,process_boot_samples
from confidence_set_func import prediction_confidence_set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    #root_X = np.sqrt(X)
    X = np.concatenate((square_X, X), axis = 1)
    return np.concatenate((np.ones((N, 1)), X), axis = 1)


def generate_sim_data(N, N_test, p, error_sd, 
                      transform_func = None, binary = False, beta=None, 
                      seed = None, X = None, X_test = None, uniform_range = None
                     ):
    '''Generate data for simulation
    
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
    uniform_range: if None, generate from N(0,1) else it must a list with two numbers indicating the low and high of uniform distribution
    
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
    # Training dataset 
    if X is None:
        if uniform_range is None:
            X = rng.standard_normal(size = (N,p))
        else:
            X = rng.uniform(uniform_range[0], uniform_range[1], size = (N,p))
    if transform_func is None:
        X_train_transformed = X
    else:
        X_train_transformed = transform_func(X)
    if beta is None:
        beta = rng.standard_normal(size = X_train_transformed.shape[1])
    if binary:
        y = X_train_transformed @ beta
        y_prob = 1/(1+np.exp(-y))
        y = rng.binomial(1, p = y_prob)
    else:
        y = X_train_transformed @ beta + rng.normal(loc = 0,scale = error_sd, size = (N,))
        y_prob = None
    # Test dataset
    #import pdb; pdb.set_trace()
    if X_test is None:
        if uniform_range is None:
            X_test = rng.standard_normal(size = (N_test,p))
        else:
            X_test = rng.uniform(uniform_range[0], uniform_range[1], size = (N_test,p))
    if transform_func is None:
        X_test_transformed = X_test
    else:
        X_test_transformed = transform_func(X_test)
    mean_test_true = X_test_transformed @ beta
    if binary:
        y_test = mean_test_true 
        y_prob_test = 1/(1+np.exp(-y_test))
        mean_test_true = y_prob_test
        y_test = rng.binomial(1, p = y_prob_test)
    else:
        y_test = mean_test_true + rng.normal(loc = 0,scale = error_sd, size = (N_test,))
        y_prob_test = None
    return X, y, X_test, y_test, mean_test_true, beta, y_prob, y_prob_test


def bootstrap_and_CS(L, level, model, model_kwargs, 
                     X, y, X_test, y_test, mean_test_true = None, 
                     center_G = True, center_pred = False, 
                     use_true_contour = False,  n_boot = 200, pred_y = True):
    
    point_pred, mean_boot_l, se, mean_boot_y_l, se_y = bootstrap(model, model_kwargs, y, X, X_test, n_boot = 200)
    #import pdb; pdb.set_trace()
    # Process the sample
    G, point_pred = process_boot_samples(mean_boot_l, point_pred, se, mean_test_true = mean_test_true, 
                                         MC = False, center_G = center_G, center_pred = center_pred)
    # construct confidence set
    (df_res, L, U, contain, contain_scb, contain_CS_scb, 
     L1, L2, U1, U2, n_points, range_v, inner_points_num,outer_points_num,true_set_points_num) = \
    prediction_confidence_set(L, level, point_pred, se, G, mean_test_true = mean_test_true, use_true_contour = use_true_contour)
    if pred_y:
        G_y, _ = process_boot_samples(mean_boot_y_l, point_pred, se_y, mean_test_true = mean_test_true, 
                                             MC = False, center_G = center_G, center_pred = center_pred)
        (df_res_y, L_y, U_y, contain_y, contain_scb_y, contain_CS_scb_y, 
         L1_y, L2_y, U1_y, U2_y, n_points_y, range_v_y, inner_points_num_y,outer_points_num_y,true_set_points_num_y) = \
        prediction_confidence_set(L, level, point_pred, se_y, G_y, mean_test_true = y_test, use_true_contour = use_true_contour)
    
        r = [[contain, contain_scb, L, U, L1, L2, U1, U2, n_points, inner_points_num,outer_points_num,true_set_points_num, model, 0],
                [contain_y, contain_scb_y, L_y, U_y, L1_y, L2_y, U1_y, U2_y, n_points_y, inner_points_num_y,outer_points_num_y,true_set_points_num_y, model, 1]
               ]
    else:
        r = [[contain, contain_scb, L, U, L1, L2, U1, U2, n_points, inner_points_num,outer_points_num,true_set_points_num, model, 0
               ]]    
    return pd.DataFrame(np.array(r), columns = ["contain","contain_scb","Lower_bound",
                                                "Upper_bound", "L1", "L2", "U1", "U2", "points_in_e1", 
                                                'inner_points_num', 'outer_points_num' ,'true_set_points_num', 
                                                'model', 'on_y'])

def sim_CS(L, level, models_l, model_kwargs_l, 
           N, N_test, p, error_sd, 
           data_sim_func, data_kwargs,
           center_G = True, center_pred = False, 
           use_true_contour = False,  n_boot = 200):
    # data simulation
    X, y, X_test, y_test, mean_test_true, beta, y_prob, y_prob_test = generate_sim_data(N, N_test, p, error_sd, **data_kwargs)
    if y_prob is not None:
        pred_y = False
    else:
        pred_y = True
    # Construct confidence set for multiple models
    result_list = []
    for model, kwarg in zip(models_l, model_kwargs_l):
        r = bootstrap_and_CS(L, level, model, kwarg,
                             X, y, X_test, y_test, mean_test_true = mean_test_true, 
                             center_G = center_G, center_pred = center_pred, 
                             use_true_contour = use_true_contour,  n_boot = n_boot, pred_y = pred_y)
        result_list.append(r)
    return pd.concat(result_list)
        
def sim_CS_wrapper(kwargs):
    return sim_CS(**kwargs)
        
def check_overfitting(model, model_kwargs, 
                      N, N_test, p, error_sd, 
                      data_sim_func, data_kwargs, plot_index = True):
    # data simulation
    X, y, X_test, y_test, mean_test_true, beta, y_prob, y_prob_test = generate_sim_data(N, N_test, p, error_sd, **data_kwargs)
    #import pdb; pdb.set_trace()
    # point prediction on training data
    model_f = model(**model_kwargs)
    model_f.fit(X, y)
    point_pred = model_f.predict(X)
    point_pred_test = model_f.predict(X_test)
    if y_prob is not None:
        y = y_prob
        y_test = y_prob_test
    train_error = np.mean((point_pred-y)**2)
    test_error = np.mean((point_pred_test-y_test)**2)
    # Plotting
    fig, axs = plt.subplots(2)
    # 1D plotting
    if X.shape[1]>1:
        # Train sort
        y_pred = list(zip(y, point_pred))
        y_pred = sorted(y_pred, key=lambda x: x[0])
        y = np.array([i[0] for i in y_pred])
        point_pred = np.array([i[1] for i in y_pred])
        # Test sort
        y_pred_test = list(zip(y_test, point_pred_test))
        y_pred_test = sorted(y_pred_test, key=lambda x: x[0])
        y_test = np.array([i[0] for i in y_pred_test])
        point_pred_test = np.array([i[1] for i in y_pred_test])
        # X axis
        X = list(range(0, len(y)))
        X_test = list(range(0, len(y_test)))
        x_label = 'sorted index'
    else:
        x_label = 'x'
    if plot_index:
        # Training
        axs[0].scatter(X, y, color = 'red', label = 'True outcome')
        axs[0].scatter(X, point_pred, label = 'Prediction')
        axs[0].legend()
        axs[0].set_xlabel(x_label)
        axs[0].set_title('Training')
        # Testing
        axs[1].scatter(X_test, y_test, color = 'red', label = 'True outcome')
        axs[1].scatter(X_test, point_pred_test, label = 'Prediction')
        axs[1].legend()
        axs[1].set_xlabel(x_label)
        axs[1].set_title('Testing')

        plt.tight_layout()

        plt.show()
    
    # return the training and test error
    return (train_error, test_error)