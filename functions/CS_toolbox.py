import numpy as np

def bootstrap(model, model_kwargs, y, X, X_test, n_boot = 200):
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
    mean_boot_y_l = []
    for i in range(n_boot):
        index_boot= np.random.randint(X.shape[0], size=int(X.shape[0]))
        index_val = np.array(list(range(0,X.shape[0])))[~np.isin(list(range(0,X.shape[0])), index_boot)]
        y_boot = y[index_boot]
        X_boot = X[index_boot]
        X_val = X[index_val]
        y_val = y[index_val]
        model_b = model(**model_kwargs)
        model_b.fit(X_boot, y_boot)
        pred_test = model_b.predict(X_test)
        pred_val = model_b.predict(X_val)
        err = pred_val - y_val
        pred_test_p_err = pred_test + err[np.random.randint(0, len(err), size = (len(pred_test),) )]
        mean_boot_l.append(pred_test)
        mean_boot_y_l.append(pred_test_p_err)
    mean_boot_l = np.array(mean_boot_l)
    mean_boot_y_l = np.array(mean_boot_y_l)
    # Point prediction
    model_f = model(**model_kwargs)
    model_f.fit(X, y)
    point_pred = model_f.predict(X_test)
    se = np.std(mean_boot_l, axis = 0)
    se_y = np.std(mean_boot_y_l, axis = 0)
    return point_pred, mean_boot_l, se, mean_boot_y_l, se_y



def process_boot_samples(mean_boot_l, point_pred, se, mean_test_true = None, 
                         MC = False, center_G = True, center_pred = True):
    '''
    Preprocessing the bootstrap samples before input into the confidence set function
    
    Parameters
    ----------------------------
    mean_boot_l:bootstrap of the predictions, matrix
    point_pred: point prediction
    se: standard error of point prediction
    mean_test_true: true mean 
    MC: whether mean_boot_l comes from Monte Carlo from true population
    center_G:  we center the bootstrap sample with its mean
    center_pred: we use mean of the bootstrap sample as point prediction but use the original distribution for G
    err: This is the training sample error; if None, then we do not take account of the variance of the irreducible error
    '''
    if MC:
        G = (mean_boot_l-mean_test_true)/se
    else:
        if center_G:
            mean_boot = np.mean(mean_boot_l, axis = 0)# assuming that the estimator is unbiased even for finite sample
            G = (mean_boot_l-mean_boot)/se
        else:
            G = (mean_boot_l-point_pred)/se
        if center_pred:
            point_pred = np.mean(mean_boot_l, axis = 0)
    return G, point_pred


