import os
import pandas as pd
import numpy as np
from itertools import chain

###########################################################################
# Data Process functions
###########################################################################

sep_index = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST',
             'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
             'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
             'Bilirubin_total', 'Hct', 'Hgb', 'PTT', 'WBC', 'Platelets']
con_index = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']

def feature_informative_missingness(case, sep_columns):
    """
    informative missingness features reflecting measurement frequency
        or time interval of raw variables
    differential features, defined by calculating the difference between
        the current record and the previous measurement value
    :param case: one patient's EHR data
    :param sep_columns: selected variables
    :return: calculated features
    """
    temp_data = np.array(case)
    for sep_column in sep_columns:
        sep_data = np.array(case[sep_column])
        nan_pos = np.where(~np.isnan(sep_data))[0]
        # Measurement frequency sequence
        interval_f1 = sep_data.copy()
        # Measurement time interval
        interval_f2 = sep_data.copy()
        if len(nan_pos) == 0:
            interval_f1[:] = 0
            temp_data = np.column_stack((temp_data, interval_f1))
            interval_f2[:] = -1
            temp_data = np.column_stack((temp_data, interval_f2))
        else:
            interval_f1[: nan_pos[0]] = 0
            for p in range(len(nan_pos)-1):
                interval_f1[nan_pos[p]: nan_pos[p+1]] = p + 1
            interval_f1[nan_pos[-1]:] = len(nan_pos)
            temp_data = np.column_stack((temp_data, interval_f1))

            interval_f2[:nan_pos[0]] = -1
            for q in range(len(nan_pos) - 1):
                length = nan_pos[q+1] - nan_pos[q]
                for l in range(length):
                    interval_f2[nan_pos[q] + l] = l

            length = len(case) - nan_pos[-1]
            for l in range(length):
                interval_f2[nan_pos[-1] + l] = l
            temp_data = np.column_stack((temp_data, interval_f2))

        # Differential features
        diff_f = sep_data.copy()
        diff_f = diff_f.astype(float)
        if len(nan_pos) <= 1:
            diff_f[:] = np.NaN
            temp_data = np.column_stack((temp_data, diff_f))
        else:
            diff_f[:nan_pos[1]] = np.NaN
            for p in range(1, len(nan_pos)-1):
                diff_f[nan_pos[p] : nan_pos[p+1]] = sep_data[nan_pos[p]] - sep_data[nan_pos[p-1]]
            diff_f[nan_pos[-1]:] = sep_data[nan_pos[-1]] - sep_data[nan_pos[-2]]
            temp_data = np.column_stack((temp_data, diff_f))

    return temp_data

def feature_slide_window(temp, con_index):
    """
    Calculate dynamic statistics in a six-hour sliding window
    :param temp: data after using a forward-filling strategy
    :param con_index: selected variables
    :return: time-series features
    """
    sepdata = temp[:, con_index]
    max_values = [[0 for col in range(len(sepdata))]
                  for row in range(len(con_index))]
    min_values = [[0 for col in range(len(sepdata))]
                  for row in range(len(con_index))]
    mean_values = [[0 for col in range(len(sepdata))]
                   for row in range(len(con_index))]
    median_values = [[0 for col in range(len(sepdata))]
                     for row in range(len(con_index))]
    std_values = [[0 for col in range(len(sepdata))]
                  for row in range(len(con_index))]
    diff_std_values = [[0 for col in range(len(sepdata))]
                       for row in range(len(con_index))]

    for i in range(len(sepdata)):
        if i < 6:
            win_data = sepdata[0:i + 1]
            for ii in range(6 - i):
                win_data = np.row_stack((win_data, sepdata[i]))
        else:
            win_data = sepdata[i - 6: i + 1]

        for j in range(len(con_index)):
            dat = win_data[:, j]
            if len(np.where(~np.isnan(dat))[0]) == 0:
                max_values[j][i] = np.nan
                min_values[j][i] = np.nan
                mean_values[j][i] = np.nan
                median_values[j][i] = np.nan
                std_values[j][i] = np.nan
                diff_std_values[j][i] = np.nan
            else:
                max_values[j][i] = np.nanmax(dat)
                min_values[j][i] = np.nanmin(dat)
                mean_values[j][i] = np.nanmean(dat)
                median_values[j][i] = np.nanmedian(dat)
                std_values[j][i] = np.nanstd(dat)
                diff_std_values[j][i] = np.std(np.diff(dat))

    win_features = list(chain(max_values, min_values, mean_values,
                              median_values, std_values, diff_std_values))
    win_features = (np.array(win_features)).T

    return win_features

def feature_empiric_score(dat):
    """
    empiric features scoring for
    heart rate (HR), systolic blood pressure (SBP), mean arterial pressure (MAP),
    respiration rate (Resp), temperature (Temp), creatinine, platelets and total bilirubin
    according to the scoring systems of NEWS, SOFA and qSOFA
    """
    scores = np.zeros((len(dat), 8))
    for ii in range(len(dat)):
        HR = dat[ii, 0]
        if HR == np.nan:
            HR_score = np.nan
        elif (HR <= 40) | (HR >= 131):
            HR_score = 3
        elif 111 <= HR <= 130:
            HR_score = 2
        elif (41 <= HR <= 50) | (91 <= HR <= 110):
            HR_score = 1
        else:
            HR_score = 0
        scores[ii, 0] = HR_score

        Temp = dat[ii, 2]
        if Temp == np.nan:
            Temp_score = np.nan
        elif Temp <= 35:
            Temp_score = 3
        elif Temp >= 39.1:
            Temp_score = 2
        elif (35.1 <= Temp <= 36.0) | (38.1 <= Temp <= 39.0):
            Temp_score = 1
        else:
            Temp_score = 0
        scores[ii, 1] = Temp_score

        Resp = dat[ii, 6]
        if Resp == np.nan:
            Resp_score = np.nan
        elif (Resp < 8) | (Resp > 25):
            Resp_score = 3
        elif 21 <= Resp <= 24:
            Resp_score = 2
        elif 9 <= Resp <= 11:
            Resp_score = 1
        else:
            Resp_score = 0
        scores[ii, 2] = Resp_score

        Creatinine = dat[ii, 19]
        if Creatinine == np.nan:
            Creatinine_score = np.nan
        elif Creatinine < 1.2:
            Creatinine_score = 0
        elif Creatinine < 2:
            Creatinine_score = 1
        elif Creatinine < 3.5:
            Creatinine_score = 2
        else:
            Creatinine_score = 3
        scores[ii, 3] = Creatinine_score

        MAP = dat[ii, 4]
        if MAP == np.nan:
            MAP_score = np.nan
        elif MAP >= 70:
            MAP_score = 0
        else:
            MAP_score = 1
        scores[ii, 4] = MAP_score

        SBP = dat[ii, 3]
        Resp = dat[ii, 6]
        if SBP + Resp == np.nan:
            qsofa = np.nan
        elif (SBP <= 100) & (Resp >= 22):
            qsofa = 1
        else:
            qsofa = 0
        scores[ii, 5] = qsofa

        Platelets = dat[ii, 30]
        if Platelets == np.nan:
            Platelets_score = np.nan
        elif Platelets <= 50:
            Platelets_score = 3
        elif Platelets <= 100:
            Platelets_score = 2
        elif Platelets <= 150:
            Platelets_score = 1
        else:
            Platelets_score = 0
        scores[ii, 6] = Platelets_score

        Bilirubin = dat[ii, 25]
        if Bilirubin == np.nan:
            Bilirubin_score = np.nan
        elif Bilirubin < 1.2:
            Bilirubin_score = 0
        elif Bilirubin < 2:
            Bilirubin_score = 1
        elif Bilirubin < 6:
            Bilirubin_score = 2
        else:
            Bilirubin_score = 3
        scores[ii, 7] = Bilirubin_score

    return scores

def feature_extraction(case):
    labels = np.array(case['SepsisLabel'])
    # drop three variables due to their massive missing values
    pid = case.drop(columns=['Bilirubin_direct', 'TroponinI', 'Fibrinogen', 'SepsisLabel'])

    temp_data = feature_informative_missingness(pid, con_index + sep_index)
    temp = pd.DataFrame(temp_data)
    # Missing values used a forward-filling strategy
    temp = temp.fillna(method='ffill')
    # 62 informative missingness features, 31 differential features
    # and 37 raw variables
    feature_A = np.array(temp)
    # Statistics in a six-hour window for the selected measurements
    # [0, 1, 3, 4, 6] = ['HR', 'O2Sat', 'SBP', 'MAP', 'Resp']
    # 30 statistical features in the window
    feature_B = feature_slide_window(feature_A, [0, 1, 3, 4, 6])
    # 8 empiric features
    feature_C = feature_empiric_score(feature_A)
    # A total of 168 features were obtained
    features = np.column_stack((feature_A, feature_B, feature_C))

    return  features, labels

def data_process(data_set, data_path_dir):
    """
    Feature matrix across all patients in the data_set
    """
    frames_features = []
    frames_labels = []
    frames_ids = []
    for i,psv in enumerate(data_set):
        patient = pd.read_csv(os.path.join(data_path_dir, psv), sep='|')
        features, labels = feature_extraction(patient)
        features = pd.DataFrame(features)
        labels = pd.DataFrame(labels)
        ids = pd.DataFrame(np.repeat(i, len(labels)))
        frames_features.append(features)
        frames_labels.append(labels)
        frames_ids.append(ids)

    dat_features = np.array(pd.concat(frames_features))
    dat_labels = (np.array(pd.concat(frames_labels)))[:, 0]
    dat_ids = (np.array(pd.concat(frames_ids)))[:, 0]
    
    index = [i for i in range(len(dat_labels))]
    np.random.shuffle(index)
    dat_features = dat_features[index]
    dat_labels = dat_labels[index]
    dat_ids = dat_ids[index]

    return dat_features, dat_labels, dat_ids


###########################################################################
# Model training functions
###########################################################################
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
from hyperopt import STATUS_OK, hp, fmin, tpe

def BO_TPE(X_train, y_train, X_val, y_val):
    "Hyperparameter optimization"
    train = xgb.DMatrix(X_train, label=y_train)
    val = xgb.DMatrix(X_val, label=y_val)
    X_val_D = xgb.DMatrix(X_val)

    def objective(params):
        xgb_model = xgb.train(params, dtrain=train, num_boost_round=1000, evals=[(val, 'eval')],
                              verbose_eval=False, early_stopping_rounds=80)
        y_vd_pred = xgb_model.predict(X_val_D, ntree_limit=xgb_model.best_ntree_limit)
        y_val_class = [0 if i <= 0.5 else 1 for i in y_vd_pred]

        acc = accuracy_score(y_val, y_val_class)
        loss = 1 - acc

        return {'loss': loss, 'params': params, 'status': STATUS_OK}

    max_depths = [3, 4]
    learning_rates = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]
    subsamples = [0.5, 0.6, 0.7, 0.8, 0.9]
    colsample_bytrees = [0.5, 0.6, 0.7, 0.8, 0.9]
    reg_alphas = [0.0, 0.005, 0.01, 0.05, 0.1]
    reg_lambdas = [0.8, 1, 1.5, 2, 4]

    space = {
        'max_depth': hp.choice('max_depth', max_depths),
        'learning_rate': hp.choice('learning_rate', learning_rates),
        'subsample': hp.choice('subsample', subsamples),
        'colsample_bytree': hp.choice('colsample_bytree', colsample_bytrees),
        'reg_alpha': hp.choice('reg_alpha', reg_alphas),
        'reg_lambda': hp.choice('reg_lambda', reg_lambdas),
    }

    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20)

    best_param = {'max_depth': max_depths[(best['max_depth'])],
                  'learning_rate': learning_rates[(best['learning_rate'])],
                  'subsample': subsamples[(best['subsample'])],
                  'colsample_bytree': colsample_bytrees[(best['colsample_bytree'])],
                  'reg_alpha': reg_alphas[(best['reg_alpha'])],
                  'reg_lambda': reg_lambdas[(best['reg_lambda'])]
                  }

    return best_param

def train_model(X_train, y_train, X_val, y_val, best_param):
    xgb_model = xgb.XGBClassifier(max_depth = best_param['max_depth'],
                                  eta = best_param['learning_rate'],
                                  n_estimators = 1000,
                                  subsample = best_param['subsample'],
                                  colsample_bytree = best_param['colsample_bytree'],
                                  reg_alpha = best_param['reg_alpha'],
                                  reg_lambda = best_param['reg_lambda'],
                                  objective = "binary:logistic", 
                                  eval_metric='error',
                                  early_stopping_rounds=80, 
                                  )

    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],verbose=False)
    y_tr_pred = (xgb_model.predict_proba(X_train, iteration_range=(0,xgb_model.best_iteration+1)))[:, 1]
    train_auc = roc_auc_score(y_train, y_tr_pred)
    print('training dataset AUC: ' + str(train_auc))
    y_tr_class = [0 if i <= 0.5 else 1 for i in y_tr_pred]
    acc = accuracy_score(y_train, y_tr_class)
    print('training dataset acc: ' + str(acc))
    y_vd_pred = (xgb_model.predict_proba(X_val, iteration_range=(0,xgb_model.best_iteration+1)))[:, 1]
    valid_auc = roc_auc_score(y_val, y_vd_pred)
    print('validation dataset AUC: ' + str(valid_auc))
    y_val_class = [0 if i <= 0.5 else 1 for i in y_vd_pred]
    acc = accuracy_score(y_val, y_val_class)
    print('validation dataset acc: ' + str(acc))
    print('************************************************************')
    return xgb_model
    # save the model
    # save_model_path = save_model_dir + 'model{}.mdl'.format(k + 1)
    # xgb_model.get_booster().save_model(fname=save_model_path)
    
    
###########################################################################
# Model testing functions
###########################################################################
import pandas as pd
import numpy as np, os, sys
import xgboost as xgb

def save_challenge_predictions(file, scores, labels):
    with open(file, 'w') as f:
        f.write('PredictedProbability|PredictedLabel\n')
        for (s, l) in zip(scores, labels):
            f.write('%g|%d\n' % (s, l))

def save_challenge_testlabel(file, labels):
    with open(file, 'w') as f:
        f.write('SepsisLabel\n')
        for l in labels:
            f.write('%d\n' % l)

def load_model_predict(X_test, k_fold, path):
    "ensemble the five XGBoost models by averaging their output probabilities"
    test_pred = np.zeros((X_test.shape[0], k_fold))
    X_test = xgb.DMatrix(X_test)
    for k in range(k_fold):
        model_path_name = path + 'model{}.mdl'.format(k+1)
        xgb_model = xgb.Booster(model_file = model_path_name)
        y_test_pred = xgb_model.predict(X_test)
        test_pred[:, k] = y_test_pred
    test_pred = pd.DataFrame(test_pred)
    result_pro = test_pred.mean(axis=1)

    return result_pro

def predict(data_set,
            data_dir,
            save_prediction_dir,
            save_label_dir,
            model_path,
            risk_threshold
            ):
    for psv in data_set:
        patient = pd.read_csv(os.path.join(data_dir, psv), sep='|')
        features, labels = feature_extraction(patient)

        predict_pro = load_model_predict(features, k_fold = 5, path = './' + model_path + '/')
        PredictedProbability = np.array(predict_pro)
        PredictedLabel = [0 if i <= risk_threshold else 1 for i in predict_pro]

        save_prediction_name = save_prediction_dir + psv
        save_challenge_predictions(save_prediction_name, PredictedProbability, PredictedLabel)
        save_testlabel_name = save_label_dir + psv
        save_challenge_testlabel(save_testlabel_name, labels)

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         raise Exception('Include the model directory as arguments, '
#                         'e.g., python test.py Submit_model')

#     test_set = np.load('./data/test_set.npy')
#     test_data_path = "./data/all_dataset/"
#     prediction_directory = './prediction/'
#     label_directory = './label/'
#     model_path = sys.argv[1]

#     predict(test_set, test_data_path, prediction_directory, label_directory, model_path, 0.525)

#     auroc, auprc, accuracy, f_measure, utility = evaluate_sepsis_score(label_directory, prediction_directory)
#     output_string = 'AUROC|AUPRC|Accuracy|F-measure|Utility\n{}|{}|{}|{}|{}'.format(
#                      auroc, auprc, accuracy, f_measure, utility)
#     print(output_string)
