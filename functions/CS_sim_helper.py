from CS_toolbox import bootstrap,process_boot_samples
from confidence_set_func import prediction_confidence_set, naive_CS_method
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *

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
    #sigmoid_X = 1/(1+np.exp(-X))
    sigmoid_X = 0*X
    # cosine
    cos_X = np.cos(X*3)*2
    # square
    #square_X = np.square(X)
    square_X = X*0
    #root_X = np.sqrt(X)
    X = np.concatenate((sigmoid_X, cos_X, square_X), axis = 1)
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
                      seed = None, X = None, X_test = None, uniform_range_beta = None,uniform_range_x = None
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
x        The training outcome with irreducible error included
        The original testing design matrix with no transformation (do not include the intercept), # features = p
        The testing outcome with irreducible error included
        The testing true mean without irreducible error included
    '''
    rng = np.random.default_rng(seed)
    # Training dataset 
    if X is None:
        if uniform_range_x is None:
            X = rng.standard_normal(size = (N,p))
        else:
            X = rng.uniform(uniform_range_x[0], uniform_range_x[1], size = (N,p))
    if transform_func is None:
        X_train_transformed = X
    else:
        X_train_transformed = transform_func(X)
    if beta is None:
        if uniform_range_beta is None:
            beta = rng.standard_normal(size = X_train_transformed.shape[1])
        else:
            beta = rng.uniform(uniform_range_beta[0], uniform_range_beta[1], size = X_train_transformed.shape[1])
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
        if uniform_range_x is None:
            X_test = rng.standard_normal(size = (N_test,p))
        else:
            X_test = rng.uniform(uniform_range_x[0], uniform_range_x[1], size = (N_test,p))
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
                     center_G = True, center_pred = False, boundary_point_ind = False,
                     use_true_contour = False,  n_boot = 200, pred_y = True, return_plot = False):
    
    point_pred, mean_boot_l, se, mean_boot_y_l, se_y = bootstrap(model, model_kwargs, y, X, X_test, n_boot = n_boot)
    #import pdb; pdb.set_trace()
    # Process the sample
    G, point_pred = process_boot_samples(mean_boot_l, point_pred, se, mean_test_true = mean_test_true, 
                                         MC = False, center_G = center_G, center_pred = center_pred)
    # construct confidence set using the new algorithm
    (df_res, result_dict) = prediction_confidence_set(L, level, point_pred, se, G, 
                                                   mean_test_true = mean_test_true, use_true_contour = use_true_contour, boundary_point_ind = boundary_point_ind)
    result_dict['method'] = 'CS_on_mean'
    # Naive method
    (_, result_dict_naive) = naive_CS_method(L, level, mean_boot_l, mean_test_true)
    result_dict_naive['method'] = 'naive_on_mean'
    r = [result_dict, result_dict_naive]
    #import pdb; pdb.set_trace()
    if pred_y:
        G_y, _ = process_boot_samples(mean_boot_y_l, point_pred, se_y, mean_test_true = mean_test_true, 
                                             MC = False, center_G = center_G, center_pred = center_pred)
        (df_res_y, result_dict_y) = prediction_confidence_set(L, level, point_pred, se_y, G_y, 
                                                           mean_test_true = y_test, use_true_contour = use_true_contour, boundary_point_ind = boundary_point_ind)
        result_dict_y['method'] = 'CS_on_y'
        (_, result_dict_naive_y) = naive_CS_method(L, level, mean_boot_y_l, y_test)
        result_dict_naive_y['method'] = 'naive_on_y'
        r.extend([result_dict_y, result_dict_naive_y])
    if return_plot:#only works for 1D case
        df_res = pd.concat([df_res, pd.DataFrame({'x':np.squeeze(X_test, axis = 1), 'prediction':point_pred, 'y': y_test, 'f(x)': mean_test_true})],axis = 1)
        df_res_y = pd.concat([df_res_y, pd.DataFrame({'x':np.squeeze(X_test, axis = 1), 'prediction':point_pred, 'y': y_test, 'f(x)': mean_test_true})],axis = 1)
        df_res['Sets'] = np.where(df_res.inner, 'inner', 
                                np.where(np.logical_and(df_res.outer, ~df_res.inner), 'uncertain', 'outside outer')
                               )
        df_res_y['Sets'] = np.where(df_res_y.inner, 'inner', 
                                np.where(np.logical_and(df_res_y.outer, ~df_res_y.inner), 'uncertain', 'outside outer')
                               )
        CS_and_plot(df_res, df_res_y, level)
    return pd.DataFrame.from_dict(r)



def CS_and_plot(df_res, df_res_y, level):
    # Plot the first plot without confidence set
    #import pdb; pdb.set_trace()
    df_res_plot = pd.melt(df_res, id_vars = ['x'], value_vars = ['prediction', 'y'], var_name = 'type', value_name = 'y')
    # ggplot() +geom_line(df_res, aes('x', 'true_mean', colour = 'True mean')) + geom_point(df_res_plot, aes('x', 'outcome', color = 'type'))+scale_colour_manual(values = {'True mean':'black'})
    (ggplot() +
     geom_line(df_res, aes(x = 'x', y = 'f(x)', color = "'f(x)'")) +
     geom_point(df_res_plot, aes('x', 'y', color = 'type'))+
     scale_color_manual(name = ' ',values = {'f(x)':'black', 'prediction':'grey', 'y':'brown'},
                      guide = guide_legend(override_aes = {'linetype': ['-', 'None','None'],
                                                            'shape':['None', 'o', 'o']} ) 
                      )+
     theme_light()+theme(legend_position="bottom", legend_box_spacing=.2)+
     labs(title = '', y = 'Outcome')
    ).save('raw_points.jpg', dpi = 300)
    # Confidence set for the true mean
    (ggplot() +
     geom_line(df_res, aes(x = 'x', y = 'f(x)', color = "'f(x)'")) + 
     geom_point(df_res, aes('x', 'prediction', color = 'Sets'))+
     scale_color_manual(name = ' ',values = {'f(x)':'black', 'inner':'red', 'uncertain':'green', 'outside outer':'blue'},
                        guide = guide_legend(override_aes = {'linetype': ['-', 'None','None', 'None'],
                                                            'shape':['None', 'o', 'o', 'o']} )
                      )+
     geom_hline(yintercept = level, linetype = 'dashed')+
     theme_light()+theme(legend_position="bottom", legend_box_spacing=.2)+
     labs(title = 'Confidence sets for f(x)', y = 'Outcome')
    ).save('CS_f.jpg', dpi = 300)
    # Confidence set for the unobserved outcome
    (ggplot() +
     geom_point(df_res_y, aes('x', 'y', color = 'Sets')) + 
     geom_point(df_res_y, aes('x', 'prediction', color = "'prediction'"))+
     scale_color_manual(name = ' ',values = {'prediction':'gray', 'inner':'red', 'uncertain':'green', 'outside outer':'blue'},
                        guide = guide_legend(override_aes = {'linetype': ['None', 'None','None', 'None'],
                                                            'shape':['o', 'o', 'o', 'o']} )
                      )+
     geom_hline(yintercept = level, linetype = 'dashed')+
     theme_light()+theme(legend_position="bottom", legend_box_spacing=.2)+
     labs(title = 'Confidence sets for y', y = 'Outcome')
    ).save('CS_y.jpg', dpi = 300)



def sim_CS(L, level, models_l, model_kwargs_l, 
           N, N_test, p, error_sd, 
           data_sim_func, data_kwargs,
           boundary_point_ind = False, 
           center_G = True, center_pred = False, 
           use_true_contour = False,  n_boot = 200, return_plot = False):
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
                             use_true_contour = use_true_contour,  n_boot = n_boot, pred_y = pred_y, return_plot = return_plot)
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



