##############################################
# Functions for finding the confidence sets
##############################################

import numpy as np
from scipy.stats import t
from sklearn.linear_model import LinearRegression
import pandas as pd
import time

def second_stage_boot(mean_boot_l, point_pred,se):
    '''
    mean_boot_l: bootstrap of the predictions, matrix
    point_pred: the prediction estimators we are using
    se_pred: the se estimate for the predictions
    '''
    #import pdb; pdb.set_trace()
    n_boot = mean_boot_l.shape[0]
    n_test = mean_boot_l.shape[1]
    mean_matrix = []
    for i in range(n_boot):
        index_boot= np.random.randint(n_boot, size=n_boot)
        mean_matrix.append(np.mean(mean_boot_l[index_boot,:], axis = 0))
    mean_matrix = np.array(mean_matrix)
    #import pdb; pdb.set_trace()
    return (mean_matrix - point_pred)/se
    

class cal_thres_at_q(object):
    def __init__(self, q, L, d, d_pos_sorted, d_neg_sorted, G, e1 = None, e2 = None):
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

        '''
        #import pdb; pdb.set_trace()
        self.G = G
        self.L = L
        if e1 is None or e2 is None:
            e1 = np.quantile(d_pos_sorted, q, method = 'closest_observation')
            e2 = np.quantile(d_pos_sorted, q, method = 'closest_observation')
        (self.index_up1, self.index_up2, self.index_lo1, self.index_lo2, 
         self.r_up1, self.r_up2, self.r_lo1, self.r_lo2, self.index_set_len) = self.cal_quantities(d, e1, e2)
        # fixed constants on the right hand side of the probability
        self.r_inf_up1 = np.min(self.r_up1)
        self.r_inf_lo1 = np.min(self.r_lo1)
        self.r_inf_up2 = np.min(self.r_up2)
        self.r_inf_lo2 = np.min(self.r_lo2)
        self.r_sup_up1 = np.max(self.r_up1)
        self.r_sup_lo1 = np.max(self.r_lo1)
        # The G statistics
        self.G_up2 = self.G[:, self.index_up2]
        self.G_lo2 = self.G[:, self.index_lo2]
        # The inf or sup of G
        self.inf_up1 = np.min(self.G[:, self.index_up1],axis = 1) 
        self.sup_lo1 = np.max(self.G[:, self.index_lo1],axis = 1) 
        self.inf_up2 = np.min(self.G[:, self.index_up2],axis = 1) 
        self.sup_lo2 = np.max(self.G[:, self.index_lo2],axis = 1) 
    
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
        lower_bound_p1 = np.mean(np.logical_and((self.inf_up1 >= -a- self.r_inf_up1), (self.sup_lo1 < a +self.r_inf_lo1)))
        lower_bound1 = lower_bound_p1+np.mean(np.logical_and((np.min(self.G_up2,axis = 1)  >= -a- self.r_inf_up2), (np.max(self.G_lo2,axis = 1) < a +self.r_inf_lo2)))-1
        lower_bound2 = lower_bound_p1+(np.sum( np.mean((self.G_up2 >= -a- self.r_up2), axis =0 ))+np.sum(np.mean((self.G_lo2 < a+ self.r_lo2),axis = 0)))-(len(self.r_up2)+len(self.r_lo2))
        return max(lower_bound1, lower_bound2), lower_bound1, lower_bound2

    def cal_upper_bound(self,a):
        '''Calculate the upper bound in the paper

        Parameters:
        -----------------
        a: the threshold quantile to calculate the upper bound at.

        Returns:
        -----------------
        the upper bound evaluated at the specific quantile of the distances.
        '''
        return np.mean(np.logical_and((self.inf_up1 >= -a- self.r_sup_up1), (self.sup_lo1 < a +self.r_sup_lo1)))
    


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
    e1 = d_pos_sorted[0]
    e2 = d_neg_sorted[0]
    max_smallest = max(e1,e2)
    largest_pos = max(d_pos_sorted)
    largest_neg = max(d_neg_sorted)
    search_func_min = cal_thres_at_q(None, L, d, d_pos_sorted, d_neg_sorted, G, e1, e2)
    # grid search
    range_min = float('inf')
    range_v = []
    q = 0
    #for q in np.arange(0.01,0.6,0.01):
    i = 0
    for e in d_abs_sorted:
        i += 1
        #import pdb; pdb.set_trace()
        if e < max_smallest: # if this is true, there is one side without any point
            continue
        if e >= largest_pos and e >= largest_neg:#if there is no points in the second part for both positive and negative distance
            break
        if e >= largest_pos:# e1 stays below largest positive so that we have something in the second part for positive side
            e1 = largest_pos -0.01
            e2 = e
        elif e >=  largest_neg:# e2 stays below largest negative so that we have something in the second part for negative side
            e1 = e
            e2 = largest_neg - 0.01
        else: # increase both distance, only one side will include one more point
            e1, e2 = e,e
        # e1 = d_pos_sorted[i]
        # e2 =  d_neg_sorted[i]
        search_func_cur = cal_thres_at_q(q, L, d, d_pos_sorted, d_neg_sorted, G, e1, e2)
        a_q, lowerb, lowerb1, lowerb2, upperb1 = search_func_cur.binary_search()
        upperb2 = search_func_min.cal_upper_bound(a_q)
        upperb = min(upperb1,upperb2)
        range_ = upperb - lowerb
        range_v.append((e, range_))
        if range_ < range_min:
            range_min = range_
            a = a_q
            L1 = lowerb1
            L2 = lowerb2
            U1 = upperb1
            U2 = upperb2
            L = lowerb
            U = upperb
            n_points = i
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
    
def prediction_confidence_set(L, level, mean, se, mean_boot_l, mean_test_true = None, use_true_contour = False, MC = False, center_G= True):
    '''Function for finding the inner and outer confidence set
    
    Parameters:
    ----------------
    L: the targeted lower bound for the coverage rate.
    level: the targeted level such that all the samples' true mean is greater than.
    mean: the predicted values from the model.
    se: the predicted values' standard error.
    mean_boot_l: the matrix of predicted values from new fitted models on the bootstraped samples.
    mean_test_true: the true mean for the test data.
    use_true_contour: indicator whether to use the true mean to calculate the distances. 
    MC: whether the input mean_boot_l is from Monte Carlo simulation instead of bootstrap
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
    if MC:
        G = (mean_boot_l-mean_test_true)/se
    else:
        if center_G:
            mean_boot = np.mean(mean_boot_l, axis = 0)# assuming that the estimator is unbiased even for finite sample
            G1 = (mean_boot_l-mean_boot)/se
            G2 = second_stage_boot(mean_boot_l, mean, se)
            G = []
            conv_total = 5
            n_boot = G1.shape[0]
            for i in range(conv_total):
                random_index1 = np.random.randint(n_boot, size=n_boot)
                random_index2 = np.random.randint(n_boot, size=n_boot)
                G.append(G1[random_index1,:]+G2[random_index2,:])
            #import pdb; pdb.set_trace()
            G = np.concatenate(G, axis = 0)
        else:
            G = (mean_boot_l-mean)/se
    if use_true_contour and mean_test_true is not None:
        d = (mean_test_true - level)/se
    else:
        d = (mean - level)/se
    # Getting the number of points in the true set
    if mean_test_true is not None:
        true_set_points_num = np.sum(mean_test_true>=level)
    else:
        true_set_points_num = None
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
        if np.all( (true_set.astype(int) - dict_["inner"].astype(int)) >= 0 ) and np.all( (dict_["outer"].astype(int) - true_set.astype(int)) >= 0 ):
            contain = True
        else:
            contain = False
        # Getting the containment for the whole set SCB
        if MC:
            r_max_l = np.max(np.abs(G),axis = 1)
        else:
            r_max_l = np.max(np.abs(G),axis = 1)
        thres = np.quantile(r_max_l, q = 1 - 0.05)
        low = mean - thres*se
        high = mean + thres*se
        inner = low >= level
        outer = high >= level
        contain_scb = np.all(np.logical_and(low<= mean_test_true, high>= mean_test_true))
        contain_CS_scb = True if np.all( (true_set.astype(int) - inner.astype(int)) >= 0 ) and np.all( (outer.astype(int) - true_set.astype(int)) >= 0 ) else False
        return pd.DataFrame(dict_), L, U, contain, contain_scb, contain_CS_scb, L1, L2, U1, U2, n_points, range_v, inner_points_num,outer_points_num,true_set_points_num
    

