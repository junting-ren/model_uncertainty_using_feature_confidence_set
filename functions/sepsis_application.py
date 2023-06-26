import sys
sys.path.append('../functions')
from importlib import reload
import sepsis_func
reload(sepsis_func)
from sepsis_func import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils import resample
import multiprocessing


##################################################################################
#Load the best parameters for XGboost
##################################################################################
import json 
def flatten(input_dict, d = {}):    
    for key, val in input_dict.items():
        if isinstance(val, dict):
            flatten(val)
        else:
            d[key] = val
    return d
# best_model = xgb.Booster(model_file = "../data/xgb_model/model1.mdl")
# best_param = json.loads(best_model.save_config())
# best_param = flatten(best_param)


##################################################################################
#Bootstrap
##################################################################################
def bootstrap_time_series(X,y,ids):
    # Get unique IDs
    unique_ids = np.unique(ids)
    
    # Bootstrap IDs
    bootstrapped_ids = resample(unique_ids, replace=True)
    
    # Initialize empty list to store bootstrapped feature data
    bootstrapped_X = []
    bootstrapped_y = []
    # For each ID in bootstrapped IDs
    for id_ in bootstrapped_ids:
        # Get the index of the features associated with the ID
        id_index = np.where(ids == id_)[0]
        
        # Get the features associated with the ID
        id_features = X[id_index]
        id_y = y[id_index]
        # Append the bootstrapped features to the list
        bootstrapped_X.append(id_features)
        bootstrapped_y.append(id_y)
    # Concatenate all bootstrapped feature data
    bootstrapped_X = np.concatenate(bootstrapped_X, axis=0)
    bootstrapped_y = np.concatenate(bootstrapped_y, axis=0)
    return bootstrapped_X, bootstrapped_y

def bootstrap_cross_sectional(X,y):
    index = np.random.choice(len(y),size = len(y), replace = True)
    return X[index], y[index]



# 

def train_boot(train_features, train_labels, train_ids, test_features, test_labels, best_param):
    #X_boot, y_boot = bootstrap_time_series(train_features, train_labels, train_ids)
    X_boot, y_boot = bootstrap_cross_sectional(train_features,train_labels)
    X_train, X_val, y_train, y_val = train_test_split(X_boot, y_boot, test_size = 0.15, random_state = 1)
    model = train_model(X_train, y_train, X_val, y_val,best_param)
    pred_raw = model.predict(test_features, output_margin = True,  iteration_range=(0,model.best_iteration+1))
    y_test_pred = [0 if i <= 0 else 1 for i in pred_raw]
    acc = accuracy_score(test_labels, y_test_pred)
    print('###########################################')
    print('Bootstrap Test dataset acc: ' + str(acc))
    test_auc = roc_auc_score(test_labels, pred_raw)
    sensitivity = np.mean(np.array(y_test_pred)[test_labels==1])
    precision = np.mean(test_labels[np.array(y_test_pred)==1])
    print(f'sensitivity is {sensitivity}')
    print(f'precision is {precision}')
    print('Test dataset AUC: ' + str(test_auc))
    print('###########################################')
    return pred_raw

def train_boot_wrapper(kwargs):
    return train_boot(**kwargs)
    
if __name__ ==  '__main__': 
    n_boot = 100
    ##################################################################################
    #Load and process data
    ##################################################################################
    rng = np.random.default_rng(2023)
    train_nosepsis = np.load('../data/train_nosepsis.npy')[0:3000]
    train_sepsis = np.load('../data/train_sepsis.npy')[0:400]
    test_nosepsis =  np.load('../data/test_nosepsis.npy')
    test_sepsis = np.load('../data/test_sepsis.npy')[0:100]
    # sub set the no sepsis subjects
    train_nosepsis_index =  rng.choice(len(train_nosepsis), size = int(len(train_sepsis)/2), replace = False)
    test_nosepsis_index = rng.choice(len(test_nosepsis), size = int(len(test_sepsis)), replace = False)
    # get the final train and test datset
    train_nosepsis_sub = train_nosepsis[train_nosepsis_index]
    test_nosepsis_sub =  test_nosepsis[test_nosepsis_index]

    train_set = np.concatenate((train_nosepsis_sub,train_sepsis))
    test_set = np.concatenate((test_nosepsis_sub,test_sepsis))

    train_features, train_labels, train_ids = data_process(train_set, "../data/all_dataset_sepsis/")
    test_features, test_labels, test_ids = data_process(test_set, "../data/all_dataset_sepsis/")
    
    np.save('./train_ids.npy', train_ids)
    np.save('./train_features.npy', train_features)
    np.save('./train_labels.npy', train_labels)
    np.save('./test_labels.npy', test_labels)
    np.save('./test_ids.npy', test_ids)
    np.save('./test_features.npy', test_features)
    
    # Bootstrap
    best_param = BO_TPE(train_features, train_labels, test_features, test_labels)
    #best_param = {'max_depth': 3, 'learning_rate': 0.15, 'subsample': 0.7, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 2}

    param = {'train_features':train_features, 'train_labels': train_labels, 'train_ids' :train_ids, 
         'test_features':test_features,'test_labels':test_labels, 'best_param':best_param}
    ctx = multiprocessing.get_context('spawn')
    pool_obj = ctx.Pool()
    cur_para = (n_boot*(param,))
    pred_raw_m = pool_obj.map(train_boot_wrapper, cur_para)
    #import pdb;pdb.set_trace()
    pred_raw_m = np.array(pred_raw_m)
    pool_obj.close()
    pool_obj.join()
    np.save('./sepsis_pred_raw_m.npy', pred_raw_m)