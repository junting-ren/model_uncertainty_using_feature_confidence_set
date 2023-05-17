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
#Load and process data
##################################################################################
rng = np.random.default_rng(2023)
train_nosepsis = np.load('../data/train_nosepsis.npy')
train_sepsis = np.load('../data/train_sepsis.npy')
test_nosepsis = np.load('../data/test_nosepsis.npy')
test_sepsis = np.load('../data/test_sepsis.npy')
# sub set the no sepsis subjects
train_nosepsis_index =  rng.choice(len(train_nosepsis), size = len(train_sepsis), replace = False)
test_nosepsis_index = rng.choice(len(test_nosepsis), size = len(test_sepsis)*2, replace = False)
# get the final train and test datset
train_nosepsis_sub = train_nosepsis[train_nosepsis_index]
test_nosepsis_sub =  test_nosepsis[test_nosepsis_index]

train_set = np.concatenate((train_nosepsis_sub,train_sepsis))
test_set = np.concatenate((test_nosepsis_sub,test_sepsis))

train_features, train_labels, train_ids = data_process(train_set, "../data/all_dataset_sepsis/")
test_features, test_labels, test_ids = data_process(test_set, "../data/all_dataset_sepsis/")

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
best_model = xgb.Booster(model_file = "../data/xgb_model/model1.mdl")
best_param = json.loads(best_model.save_config())
best_param = flatten(best_param)


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

def train_boot(train_features, train_labels, train_ids, test_features, best_param):
    X_boot, y_boot = bootstrap_time_series(train_features, train_labels, train_ids)
    X_train, X_val, y_train, y_val = train_test_split(X_boot, y_boot, test_size = 0.15, random_state = 1)
    model = train_model(X_train, y_train, X_val, y_val,best_param)
    pred_raw = model.predict(test_features, output_margin = True)
    return pred_raw

def train_boot_wrapper(kwargs):
    return train_boot(**kwargs)
    
n_boot = 10
param = {'train_features':train_features, 'train_labels': train_labels, 'train_ids' :train_ids, 'test_features':test_features, 'best_param':best_param}
if __name__ ==  '__main__': 
    ctx = multiprocessing.get_context('spawn')
    pool_obj = ctx.Pool()
    cur_para = (n_boot*(param,))
    #import pdb;pdb.set_trace()
    pred_raw_m = np.array(pool_obj.map(train_boot_wrapper, cur_para))
    pool_obj.close()
    pool_obj.join()
    np.save('./sepsis_pred_raw_m.npy', pred_raw_m)