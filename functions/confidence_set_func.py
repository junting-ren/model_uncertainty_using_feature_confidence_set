##############################################
# Functions for finding the confidence sets
##############################################

import numpy as np
from scipy.stats import t
from sklearn.linear_model import LinearRegression
import pandas as pd
import time


def second_stage_boot(mean_boot_l, n_sample = None):
    '''
    mean_boot_l: bootstrap of the predictions, matrix
    '''
    #import pdb; pdb.set_trace()
    n_boot = mean_boot_l.shape[0]
    n_test = mean_boot_l.shape[1]
    mean_matrix = []
    if n_sample is None:
        n_sample = n_boot
    for i in range(n_boot):
        index_boot= np.random.randint(n_boot, size=n_sample)
        mean_matrix.append(np.mean(mean_boot_l[index_boot,:], axis = 0))
    mean_matrix = np.array(mean_matrix)
    se =  np.std(mean_matrix, axis = 0)
    #import pdb; pdb.set_trace()
    return mean_matrix,se


def process_boot_samples(mean_boot_l, mean, se, mean_test_true = None, MC = False, 
                         second_stage = False, center_G = True,
                        center_pred = True,return_mean_boot_l = False):
    '''
    Preprocessing the bootstrap samples before input into the confidence set function
    
    Parameters
    ----------------------------
    mean_boot_l:bootstrap of the predictions, matrix
    mean: point prediction
    se: standard error of point prediction
    mean_test_true: true mean 
    MC: whether mean_boot_l comes from Monte Carlo from true population
    second_stage: whether to decrease the variance using the bootstrap mean distribution and point prediction
    center_G:  we center the bootstrap sample with its mean
    center_pred: we use mean of the bootstrap sample as point prediction but use the original distribution for G
    err: This is the training sample error; if None, then we do not take account of the variance of the irreducible error
    '''
    # if err is not None:
    #     n_rep = 10
    #     mean_boot_l = np.tile(mean_boot_l,[n_rep, 1])
    #     temp = err[np.random.randint(0, len(err), size = (mean_boot_l.shape[0], mean_boot_l.shape[1]) )].reshape((mean_boot_l.shape))
    #     mean_boot_l = mean_boot_l-temp
    #     se = np.std(mean_boot_l, axis = 0)
    #import pdb; pdb.set_trace()
    if MC:
        G = (mean_boot_l-mean_test_true)/se
    elif second_stage:
        mean_matrix, se = second_stage_boot(mean_boot_l) 
        mean_boot = np.mean(mean_boot_l, axis = 0)
        if center_G:
            G = (mean_matrix - mean_boot)/se
        else:
            G = (mean_matrix - mean)/se
        if center_pred:
            mean = mean_boot
    else:
        if center_G:
            mean_boot = np.mean(mean_boot_l, axis = 0)# assuming that the estimator is unbiased even for finite sample
            G = (mean_boot_l-mean_boot)/se
        else:
            G = (mean_boot_l-mean)/se
        if center_pred:
            mean = np.mean(mean_boot_l, axis = 0)
    if return_mean_boot_l:
        return G, mean, se, mean_boot_l
    else:
        return G, mean, se

class cal_thres_at_q(object):
    def __init__(self, q, L, d, d_pos_sorted, d_neg_sorted, G, e1 = None, e2 = None, blur_boundary = True):
        '''Initialize the function to calculate the threshold when lower bound equal to L at q quantile of the distances

        Parameters:
        -----------------
        q: quantile to use to obtain the current e values. Between 0 and 1.
        L: the targeted lower bound to aim, with this, the threshold 'a' is determined.
        d: non sorted original standardized distance, index same as the data.
        d_pos_sorted: positive distance sorted from small to large.
        d_neg_sorted: negative distance (absolute value) sorted from small to large.
        e1: the inflated distance for above the level of interest
        e2: the inflated distance for below the level of interest
        G：The G statistics
        blur_boundary: whether to take the absolute value around the boundary so 
        '''
        #import pdb; pdb.set_trace()
        self.G = G
        self.L = L
        if e1 is None or e2 is None:
            e1 = np.quantile(d_pos_sorted, q, method = 'closest_observation')
            e2 = np.quantile(d_pos_sorted, q, method = 'closest_observation')
        (self.index_up1, self.index_up2, self.index_lo1, self.index_lo2, 
         self.r_up1, self.r_up2, self.r_lo1, self.r_lo2, self.index_set_len) = self.cal_quantities(d, e1, e2)
        self.n_up1 = len(self.r_up1)
        self.n_lo1 = len(self.r_lo1)
        self.n_up2 = len(self.r_up2)
        self.n_lo2 = len(self.r_lo2)
        # fixed constants on the right hand side of the probability
        self.r_inf_up1 = np.min(self.r_up1) if self.n_up1>0 else None
        self.r_inf_lo1 = np.min(self.r_lo1) if self.n_lo1>0 else None
        self.r_inf_up2 = np.min(self.r_up2) if self.n_up2>0 else None
        self.r_inf_lo2 = np.min(self.r_lo2) if self.n_lo2>0 else None
        self.r_sup_up1 = np.max(self.r_up1) if self.n_up1>0 else None
        self.r_sup_lo1 = np.max(self.r_lo1) if self.n_lo1>0 else None
        # The G statistics
        #import pdb; pdb.set_trace()
        self.G_up2 = self.G[:, self.index_up2] if self.n_up2>0 else None
        self.G_lo2 = self.G[:, self.index_lo2] if self.n_lo2>0 else None
        # The inf or sup of G
        self.inf_up1 = np.min(self.G[:, self.index_up1],axis = 1) if self.n_up1>0 else None
        self.sup_lo1 = np.max(self.G[:, self.index_lo1],axis = 1) if self.n_lo1>0 else None
        self.inf_up2 = np.min(self.G[:, self.index_up2],axis = 1) if self.n_up2>0 else None
        self.sup_lo2 = np.max(self.G[:, self.index_lo2],axis = 1) if self.n_lo2>0 else None
        
        self.blur_boundary = blur_boundary
    
    def binary_search(self):
        '''Binary search for the threshold a
        Returns:
        ----------------
        a tuple contains the following:
            the threshold 
            the minimum of the two lower bounds
            lowerbound1 specified in the paper
            lowerbound2 specified in the paper (probability minus the cardinality)
            upperbound1 specified in the paper
        '''
        a_high = 5
        a_low = 0.1
        a_med = (a_high+a_low)/2
        lower_b_med,lower_bound1,lower_bound2 = self.cal_lower_bound(a_med)
        j = 1
        #import pdb; pdb.set_trace()
        while np.abs(lower_b_med-self.L)>0.001 and j < 100:
            if lower_b_med > self.L:
                a_high = a_med
            else:
                a_low = a_med
            a_med = (a_high+a_low)/2
            lower_b_med, lower_bound1, lower_bound2 = self.cal_lower_bound(a_med)
            j += 1
        upper_bound1 = self.cal_upper_bound(a_med)
        return a_med, lower_b_med, lower_bound1, lower_bound2, upper_bound1
    
    def cal_quantities(self, d, e1, e2):
        ''' Function to calculate vectors of index and quantities of interest for both side of the boundary

        Parameters:
        ----------------
        d: standardized distance to the level of interest
        e1: the inflated distance for above the level of interest
        e2: the inflated distance for below the level of interest

        Returns:
        ----------------
        A tuple contains the following:
            the vector of boolean indicating the positions of samples such that its d is less than e1 above the level
            the vector of boolean indicating the positions of samples such that its d is greater than e1 above the level
            the vector of boolean indicating the positions of samples such that its d is greater than e2 below the level
            the vector of boolean indicating the positions of samples such that its d is less than e2 below the level
            the vector of absolute distance values for samples such that its d is less than e1 above the level
            the vector of absolute distance values for samples such that its d is greater than e1 above the level
            the vector of absolute distance values for samples such that its d is greater than e2 below the level
            the vector of absolute distance values for samples such that its d is less than e2 below the level
            the number of samples in the first part for both above and blow the level of interest.
        '''
        index_up1 = (d >= 0) & (d <= e1)
        index_up2 = (d > e1)
        index_lo1 = (d >= -e2) & (d <0)
        index_lo2 = (d < -e2)
        # Four differences (we are using the estimated mean here)
        r_up1 = np.abs(d[index_up1])
        r_up2 = np.abs(d[index_up2])
        r_lo1 = np.abs(d[index_lo1])
        r_lo2 = np.abs(d[index_lo2])
        index_set_len = np.sum(index_up1)+np.sum(index_lo1)
        return index_up1, index_up2, index_lo1, index_lo2, r_up1, r_up2, r_lo1, r_lo2, index_set_len
    
    def cal_lower_bound(self, a):
        '''Calculate the lower bound for a given threshold 'a'

        Parameters:
        ------------------
        a: the threshold quantile to calculate the lower bound at.

        Returns:
        ------------------
        a tuple contains the following:
            the maximum of lowerbound1 and lowerbound2
            lowerbound1 specified in the paper
            lowerbound2 specified in the paper (probability minus the cardinality)
        '''
        #import pdb;pdb.set_trace()
        if self.blur_boundary:
            #import pdb; pdb.set_trace()
            if self.n_up1>0 and self.n_lo1>0:
                sup_1 = np.max( np.abs(np.concatenate([np.expand_dims(self.inf_up1,1), np.expand_dims(self.sup_lo1,1)], axis = 1)), 1)
                #import pdb; pdb.set_trace()
                r_inf_1 = min(self.r_inf_up1, self.r_inf_lo1)
            else: 
                if self.n_up1>0:
                    #import pdb;pdb.set_trace()
                    sup_1 = np.abs(self.inf_up1)
                    r_inf_1 = self.r_inf_up1
                else:
                    sup_1 = np.abs(self.sup_lo1)
                    r_inf_1 = self.r_inf_lo1
            lower_bound_p1 = np.mean(sup_1 < a + r_inf_1)
        else:
            if self.n_up1>0 and self.n_lo1>0:
                lower_bound_p1 = np.mean(np.logical_and((self.inf_up1 >= -a- self.r_inf_up1), (self.sup_lo1 < a +self.r_inf_lo1)))
            else:
                if self.n_up1>1:
                    lower_bound_p1 = np.mean((self.inf_up1 >= -a- self.r_inf_up1))
                else:
                    lower_bound_p1 = np.mean((self.sup_lo1 >= -a- self.r_inf_lo1))
        if self.n_up2>0 and self.n_lo2>0:
            lower_bound1 = lower_bound_p1+np.mean(np.logical_and((np.min(self.G_up2,axis = 1)  >= -a- self.r_inf_up2), (np.max(self.G_lo2,axis = 1) < a +self.r_inf_lo2)))-1
            lower_bound2 = lower_bound_p1+(np.sum( np.mean((self.G_up2 >= -a- self.r_up2), axis =0 ))+np.sum(np.mean((self.G_lo2 < a+ self.r_lo2),axis = 0)))-(len(self.r_up2)+len(self.r_lo2))
        else:
            if self.n_up2>0:
                lower_bound1 = lower_bound_p1+np.mean((np.min(self.G_up2,axis = 1)  >= -a- self.r_inf_up2))-1
                lower_bound2 = lower_bound_p1+np.sum( np.mean((self.G_up2 >= -a- self.r_up2), axis =0 ))-len(self.r_up2)
            elif self.n_lo2>0:
                lower_bound1 = lower_bound_p1+np.mean((np.max(self.G_lo2,axis = 1) < a +self.r_inf_lo2))-1
                lower_bound2 = lower_bound_p1+np.sum(np.mean((self.G_lo2 < a+ self.r_lo2),axis = 0))-len(self.r_lo2)
            else:
                lower_bound1 = lower_bound_p1
                lower_bound2 = lower_bound_p1
        return lower_bound2, lower_bound1, lower_bound2

    def cal_upper_bound(self,a):
        '''Calculate the upper bound in the paper

        Parameters:
        -----------------
        a: the threshold quantile to calculate the upper bound at.

        Returns:
        -----------------
        the upper bound evaluated at the specific quantile of the distances.
        '''
        if self.n_up1>0 and self.n_lo1>0:
            upper_bound = np.mean(np.logical_and((self.inf_up1 >= -a- self.r_sup_up1), (self.sup_lo1 < a +self.r_sup_lo1)))
        else:
            if self.n_up1>0:
                upper_bound = np.mean((self.inf_up1 >= -a- self.r_sup_up1))
            else:
                upper_bound = np.mean((self.sup_lo1 < a +self.r_sup_lo1))
        return upper_bound


def distance_search(L, d, d_pos_sorted, d_neg_sorted, d_abs_sorted, G):
    '''Search function of the distance such that the range between the lower and upper bound is minimized
    
    Parameters:
    -----------------
    L: the targeted lower bound
    d: the distance for the samples with signs
    d_pos_sorted: absolute value for the positive side of the distance sorted in ascending order
    d_neg_sorted: absolute value for the negative side of the distance sorted in ascending order
    d_abs_sorted: absolute value for distance sorted in ascending order
    G: standardized G statistics for all the samples
    
    Returns:
    -----------------
    a tuple contains the following:
        Threshold a 
        Lower bound
        Upper bound
        Lower bound1
        Lower bound2
        Upper bound1
        Upper bound2
        The total number of points in both side of the boundary for the first part
        The range between the upper and lower bound
    '''
    # the smallest distance possible
    e1 = d_pos_sorted[0] if len(d_pos_sorted)>0 else 0
    e2 = d_neg_sorted[0] if len(d_neg_sorted)>0 else 0
    max_smallest = max(e1,e2)
    largest_pos = max(d_pos_sorted) if len(d_pos_sorted)>0 else 0
    largest_neg = max(d_neg_sorted) if len(d_neg_sorted)>0 else 0
    search_func_min = cal_thres_at_q(None, L, d, d_pos_sorted, d_neg_sorted, G, e1, e2)
    # grid search
    range_min = float('inf')
    range_v = []
    q = 0
    #for q in np.arange(0.01,0.6,0.01):
    i = 0
    patience = 0
    max_patience = 300
    for e in d_abs_sorted:
        i += 1
        #import pdb; pdb.set_trace()
        # if e < max_smallest: # if this is true, there is one side without any point
        #     continue
        # if e >= largest_pos and e >= largest_neg:#if there is no points in the second part for both positive and negative distance
        #     break
        # if e >= largest_pos:# e1 stays below largest positive so that we have something in the second part for positive side
        #     e1 = largest_pos -0.01
        #     e2 = e
        # elif e >=  largest_neg:# e2 stays below largest negative so that we have something in the second part for negative side
        #     e1 = e
        #     e2 = largest_neg - 0.01
        # else: # increase both distance, only one side will include one more point
        e1, e2 = e,e
        # e1 = d_pos_sorted[i]
        # e2 =  d_neg_sorted[i]
        search_func_cur = cal_thres_at_q(q, L, d, d_pos_sorted, d_neg_sorted, G, e1, e2)
        a_q, lowerb, lowerb1, lowerb2, upperb1 = search_func_cur.binary_search()
        upperb2 = search_func_min.cal_upper_bound(a_q)
        #upperb = min(upperb1,upperb2)
        if i %100==0:
            print(f'finished %s' %(i/len(d_abs_sorted)))
        upperb = upperb1
        range_ = upperb - lowerb
        range_v.append((e, range_))
        patience += 1
        if range_ < range_min:
            patience = 0
            range_min = range_
            a = a_q
            L1 = lowerb1
            L2 = lowerb2
            U1 = upperb1
            U2 = upperb2
            L = lowerb
            U = upperb
            n_points = i
        if patience> max_patience:
            break
    return a, L, U, L1, L2, U1,U2, n_points, range_v
    # initialize the golden search parameters
    # q0 = 0
    # q3 = 1
    # gr = 0.618
    # #import pdb; pdb.set_trace()
    # while np.abs(q3-q0)>0.05:
    #     g = gr*(q3-q0)
    #     q1 = q0 + g
    #     q2 = q3 - g
    #     a_q_v = []
    #     R = []
    #     LB = []
    #     UB = []
    #     for q in (q1, q2):
    #         a_q, lowerb, lowerb1, lowerb2, upperb1 = cal_thres_at_q(q, L, d, d_pos_sorted, d_neg_sorted, G)
    #         upperb2 = cal_upper_bound(a_q, inf_up1_min, sup_lo1_min, r_sup_up1_min, r_sup_lo1_min)
    #         upperb = min(upperb1,upperb2)
    #         range_ = upperb - lowerb
    #         a_q_v.append(a_q)
    #         R.append(range_)
    #         LB.append(lowerb)
    #         UB.append(upperb)
    #     if R[0] < R[1]:
    #         q0, a, L, U = q2, a_q_v[0], LB[0], UB[0]
    #     else:
    #         q3, a, L, U = q1, a_q_v[1], LB[1], UB[1]
        #print('goldsec once')
    # return a, L, U
    
def prediction_confidence_set(L, level, mean, se, G, mean_test_true = None, use_true_contour = False, test_null = False):
    '''Function for finding the inner and outer confidence set
    
    Parameters:
    ----------------
    L: the targeted lower bound for the coverage rate.
    level: the targeted level such that all the samples' true mean is greater than.
    mean: the predicted values from the model.
    se: the predicted values' standard error.
    G: the standardized matrix of predicted values from new fitted models on the bootstraped samples.
    mean_test_true: the true mean for the test data.
    use_true_contour: indicator whether to use the true mean to calculate the distances. 
    Returns:
    ---------------
    A tuple contains the following:
        Dataframe contains the predicted mean, lower interval, upper interval, inner set index, outer set index
        Lower bound
        Upper bound
        whether the true set contains the inner set and the outer set contains the true set
        the coverage rate of the SCB for all points
        the covarage rate of the confidence est contructed using SCB
        Lower bound1
        Lower bound2
        Upper bound1
        Upper bound2
        Number of points in either side of the boundary for the first part
        The range between the upper and lower bound
        The number of points in the inner set
        The number of points in the outer set
        The number of points in the true set
    '''
    if use_true_contour and mean_test_true is not None:
        d = (mean_test_true - level)/se
    else:
        d = (mean - level)/se
    # Getting the number of points in the true set
    if mean_test_true is not None:
        true_set_points_up_num = np.sum(mean_test_true>=level)
        true_set_points_low_num = np.sum(mean_test_true<level)
    else:
        true_set_points_num = None
        true_set_points_low_num = None
    positive_indx = d >= 0
    negative_indx = d < 0
    d_pos_sorted = np.sort(d[positive_indx])
    d_neg_sorted = np.sort(np.abs(d[negative_indx]))
    d_abs_sorted = np.sort(np.abs(d))
    #import pdb; pdb.set_trace()
    a, L, U, L1, L2, U1, U2, n_points, range_v = distance_search(L, d, d_pos_sorted, d_neg_sorted,d_abs_sorted, G)
    low = mean - a*se
    high = mean + a*se
    dict_ = {"mean": mean, "low": low, "high": high, "inner": low >= level, "outer": high >= level}
    inner_points_num = np.sum(dict_['inner'])
    outer_points_num = np.sum(dict_['outer'])
    # Whether there is true mean or not
    if mean_test_true is None:
        return pd.DataFrame(dict_), L, U, None, None, None
    else:# if there is True mean
        true_set = mean_test_true >= level
        # For classifying whether it is greater than c
        if np.sum(dict_["inner"].astype(int) )>0:
            precision_inner = 1-np.sum((dict_["inner"].astype(int) - true_set.astype(int))>=1)/np.sum(dict_["inner"].astype(int) )
        else:
            precision_inner = 1
        # out of the all the true positive, percentage of them classified as positive
        sensitivity_inner = 1-np.sum((true_set.astype(int) - 
                                      dict_["inner"].astype(int) )>=1)/np.sum(true_set.astype(int) ) if np.sum(true_set.astype(int) )>0 else None
        # For classifying whether it is less than c
        compl_outer = 1 - dict_["outer"].astype(int)# complement of outer set
        true_set_less_c = (mean_test_true < level).astype(int)
        if np.sum(compl_outer)>0:
            precision_outer = 1-np.sum((compl_outer - true_set_less_c)>=1)/np.sum(compl_outer)
        else:
            precision_outer = 1
        sensitivity_outer = 1-np.sum((true_set_less_c - compl_outer)>=1)/np.sum(true_set_less_c) if np.sum(true_set_less_c)>0 else None
        # Confidence set containment
        if test_null:
            #import pdb;pdb.set_trace()
            null_set = mean_test_true==level
            signi_set = np.logical_or(dict_["inner"],~dict_["outer"])
            if np.all( ((1-signi_set.astype(int)) - null_set.astype(int)) >=0):
                contain = True
            else:
                contain = False
        else:
            if np.all( (true_set.astype(int) - dict_["inner"].astype(int)) >= 0 ) and np.all( (dict_["outer"].astype(int) - true_set.astype(int)) >= 0 ):
                contain = True
            else:
                contain = False
        # Getting the containment for the whole set SCB
        r_max_l = np.max(np.abs(G),axis = 1)
        thres = np.quantile(r_max_l, q = 1 - 0.05)
        low = mean - thres*se
        high = mean + thres*se
        inner = low >= level
        outer = high >= level
        contain_scb = np.all(np.logical_and(low<= mean_test_true, high>= mean_test_true))
        contain_CS_scb = True if np.all( (true_set.astype(int) - inner.astype(int)) >= 0 ) and np.all( (outer.astype(int) - true_set.astype(int)) >= 0 ) else False
        agg_dict = {"contain":contain, "contain_scb":contain_scb,'contain_CS_scb':contain_CS_scb,
                    'Lower_bound':L, 'Upper_bound': U, 
                    'L1':L1, 'L2': L2, 'U1': U1, 'U2':U2, 'points_in_e1': n_points,
                    'inner_points_num': inner_points_num,'outer_points_num':outer_points_num,
                    'true_set_points_up_num':true_set_points_up_num, 'true_set_points_low_num':true_set_points_low_num,
                    'precision_inner':precision_inner, 'sensitivity_inner':sensitivity_inner,
                    'precision_outer':precision_outer,'sensitivity_outer':sensitivity_outer}
        return pd.DataFrame(dict_), agg_dict
    
def naive_CS_method(L, level, mean_boot_l, mean_test_true = None):
    #import pdb; pdb.set_trace()
    percent_above = np.mean(mean_boot_l > level, axis = 0)
    index_inner = percent_above > L
    percent_below = np.mean(mean_boot_l < level, axis = 0)
    index_outer = ~(percent_below > L)
    dict_ = {"mean": None, "low": None, "high": None, "inner": index_inner, "outer": index_outer}
    inner_points_num = np.sum(dict_['inner'])
    outer_points_num = np.sum(dict_['outer'])
    # Whether there is true mean or not
    if mean_test_true is None:
        return pd.DataFrame(dict_), L, None, None, None, None
    else:# if there is True mean
        true_set_points_up_num = np.sum(mean_test_true>=level)
        true_set_points_low_num = np.sum(mean_test_true<level)        
        true_set = mean_test_true >= level
        # For classifying whether it is greater than c
        if np.sum(dict_["inner"].astype(int) )>0:
            precision_inner = 1-np.sum((dict_["inner"].astype(int) - true_set.astype(int))>=1)/np.sum(dict_["inner"].astype(int) )
        else:
            precision_inner = 1
        # out of the all the true positive, percentage of them classified as positive
        sensitivity_inner = 1-np.sum((true_set.astype(int) - 
                                      dict_["inner"].astype(int) )>=1)/np.sum(true_set.astype(int) ) if np.sum(true_set.astype(int) )>0 else None
        # For classifying whether it is less than c
        compl_outer = 1 - dict_["outer"].astype(int)# complement of outer set
        true_set_less_c = (mean_test_true < level).astype(int)
        if np.sum(compl_outer)>0:
            precision_outer = 1-np.sum((compl_outer - true_set_less_c)>=1)/np.sum(compl_outer)
        else:
            precision_outer = 1
        sensitivity_outer = 1-np.sum((true_set_less_c - compl_outer)>=1)/np.sum(true_set_less_c) if np.sum(true_set_less_c)>0 else None
        # Confidence set containment
        if np.all( (true_set.astype(int) - dict_["inner"].astype(int)) >= 0 ) and np.all( (dict_["outer"].astype(int) - true_set.astype(int)) >= 0 ):
            contain = True
        else:
            contain = False
        # Getting the containment for the whole set SCB
        contain_scb = None
        contain_CS_scb = None
        agg_dict = {"contain":contain, "contain_scb":contain_scb,'contain_CS_scb':contain_CS_scb,
                    'Lower_bound':L, 'Upper_bound': None, 
                    'L1':None, 'L2': None, 'U1': None, 'U2':None, 'points_in_e1': None,
                    'inner_points_num': inner_points_num,'outer_points_num':outer_points_num,
                    'true_set_points_up_num':true_set_points_up_num, 'true_set_points_low_num':true_set_points_low_num,
                    'precision_inner':precision_inner, 'sensitivity_inner':sensitivity_inner,
                    'precision_outer':precision_outer,'sensitivity_outer':sensitivity_outer}
        return pd.DataFrame(dict_), agg_dict

class multiple_testing_confidence_set(object):
    def __init__(self, L, level, mean, se, G, mean_test_true = None):
        self.L = L
        self.level = level
        self.mean = mean
        self.se = se
        self.G = G
        self.mean_test_true = mean_test_true
        self.alpha = 1-self.L
        self.test_statistics = np.abs(self.mean-self.level)/self.se
        if mean_test_true is not None:
            self.true_set_points_up_num = np.sum(mean_test_true>=level)
            self.true_set_points_low_num = np.sum(mean_test_true<level)
        else:
            self.true_set_points_num = None
            self.true_set_points_low_num = None
        
    
    def p_to_CS(self, adjust_p, test_null = False):
        inner = np.logical_and(self.mean > self.level, adjust_p<self.alpha)
        outer = np.logical_or(inner, adjust_p>self.alpha)
        dict_ = {"mean": self.mean, "low": self.mean, "high": self.mean, "inner": inner, "outer": outer}
        inner_points_num = np.sum(dict_['inner'])
        outer_points_num = np.sum(dict_['outer'])
        if self.mean_test_true is None:
            return pd.DataFrame(dict_), self.L, None, None, None, None
        else:# if there is True mean
            true_set = self.mean_test_true >= self.level
            # For classifying whether it is greater than c
            if np.sum(dict_["inner"].astype(int) )>0:
                precision_inner = 1-np.sum((dict_["inner"].astype(int) - true_set.astype(int))>=1)/np.sum(dict_["inner"].astype(int) )
            else:
                precision_inner = 1
            # out of the all the true positive, percentage of them classified as positive
            sensitivity_inner = 1-np.sum((true_set.astype(int) - 
                                          dict_["inner"].astype(int) )>=1)/np.sum(true_set.astype(int) ) if np.sum(true_set.astype(int) )>0 else None
            # For classifying whether it is less than c
            compl_outer = 1 - dict_["outer"].astype(int)# complement of outer set
            true_set_less_c = (self.mean_test_true < self.level).astype(int)
            if np.sum(compl_outer)>0:
                precision_outer = 1-np.sum((compl_outer - true_set_less_c)>=1)/np.sum(compl_outer)
            else:
                precision_outer = 1
            sensitivity_outer = 1-np.sum((true_set_less_c - compl_outer)>=1)/np.sum(true_set_less_c) if np.sum(true_set_less_c)>0 else None
            # Confidence set containment
            if test_null:
                null_set = self.mean_test_true==self.level
                signi_set = np.logical_or(dict_["inner"],~dict_["outer"])
                if np.all( ((1-signi_set.astype(int)) - null_set.astype(int)) >=0):
                    contain = True
                else:
                    contain = False
            else:
                if np.all( (true_set.astype(int) - dict_["inner"].astype(int)) >= 0 ) and np.all( (dict_["outer"].astype(int) - true_set.astype(int)) >= 0 ):
                    contain = True
                else:
                    contain = False
            agg_dict = {"contain":contain, "contain_scb":None,'contain_CS_scb':None,
                'Lower_bound':self.L, 'Upper_bound': None, 
                'L1':None, 'L2': None, 'U1': None, 'U2': None, 'points_in_e1': None,
                'inner_points_num': inner_points_num,'outer_points_num':outer_points_num,
                'true_set_points_up_num':self.true_set_points_up_num, 'true_set_points_low_num':self.true_set_points_low_num,
                'precision_inner':precision_inner, 'sensitivity_inner':sensitivity_inner,
                'precision_outer':precision_outer,'sensitivity_outer':sensitivity_outer}
        return pd.DataFrame(dict_), agg_dict

    
    def maxT_step_down_confidence_set(self):
        # order the statistics and get the index ordering from smallest to largest
        sorted_index = np.argsort(self.test_statistics)
        # number of tests
        I = self.G.shape[1]
        # number of bootstrap
        n_boot = self.G.shape[0]
        # Empty matrix for saving the null statistic distribution
        U_matrix = np.zeros(self.G.shape)
        # Loop over the number bootstraps
            # Loop over the index and get the threshold for each prediction
        #import pdb; pdb.set_trace()
        for b in range(n_boot):
            for j in range(I):
                index = sorted_index[j]
                statistics = abs(self.G[b, index])
                if j==0:
                    U_matrix[b, index] = statistics
                else:
                    U_matrix[b, index] = max(U_matrix[b, sorted_index[j-1]],statistics)
        #import pdb; pdb.set_trace()
        # vector of p-values
        p_values = np.zeros(I)
        # Calculate the p value
        for j in range(I):
            p_values[j] = np.mean(U_matrix[:,j]>=self.test_statistics[j])
        # Monotonicity constraint
        #import pdb; pdb.set_trace()
        for j in range(2,I+1):
            p_values[sorted_index[I-j]] = max(p_values[sorted_index[I-j]], p_values[sorted_index[I-j+1]])
        if self.mean_test_true is None:
            return pd.DataFrame(dict_), L, None, None, None, None
        else:# if there is True mean
            df_cs_r, agg_dict = self.p_to_CS(p_values)
            return df_cs_r, agg_dict

    def Bonf_confidence_set(self):
        # number of tests
        I = self.G.shape[1]
        # number of bootstrap
        n_boot = self.G.shape[0]
        p_values = np.zeros(I)
        for j in range(I):
            p_values[j] = np.mean(self.test_statistics[j]<=np.abs(self.G[:,j]))
        # Bonferroni confidence set
        p_values_Bonf = p_values*I
        df_cs_r, r_dict_Bonf = self.p_to_CS(p_values_Bonf)
        return df_cs_r, r_dict_Bonf
    
    def Holm_confidence_set(self):
        # number of tests
        I = self.G.shape[1]
        # number of bootstrap
        n_boot = self.G.shape[0]
        p_values = np.zeros(I)
        for j in range(I):
            p_values[j] = np.mean(self.test_statistics[j]<=np.abs(self.G[:,j]))
        # Holm confidence set
        sorted_index = np.argsort(p_values)# smallest to largest
        p_values_Holm = np.zeros(I)
        for j in range(I):
            p_values_Holm[sorted_index[j]] = (I-j)*p_values[sorted_index[j]]
        for j in range(1,I):
            if p_values_Holm[sorted_index[j]]< p_values_Holm[sorted_index[j-1]]:
                p_values_Holm[sorted_index[j]]= p_values_Holm[sorted_index[j-1]]
        df_cs_r, r_dict_Holm = self.p_to_CS(p_values_Holm)
        return df_cs_r, r_dict_Holm

   
    