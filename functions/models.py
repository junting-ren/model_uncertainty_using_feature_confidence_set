from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor

############################################################################
# Logistic Regression
############################################################################
class logistic_regression(object):
    def __init__(self, solver = 'lbfgs'):
        self.model = LogisticRegression(solver = solver)
    
    def fit(self, X, y):
        self.model.fit(X,y)
        
        
    def predict(self, X):
        return self.model.predict_proba(X)[:,1]
    
    def predict_class(self, X):
        return self.model.predict(X)

############################################################################
# Neural Networks
############################################################################
import torch
from torch import nn
import numpy as np
import copy

class FF_neural_net(nn.Module):
    def __init__(self, input_size, h_sizes, out_size = 1, out_act = 'linear', batchnorm_ind = False):
        ''' Initialize the feedfoward neural network
        
        Parameters:
        -------------------
        input_size: integer, the feature size x
        h_sizes: list, hidden layer parameter size
        out_size: integer, the number of linear outputs
        '''
        super().__init__()
        layer_sizes = [input_size] + h_sizes
        self.layers = nn.ModuleList()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.out_act = out_act
        for k in range(len(layer_sizes)-1):
            self.layers.append(nn.Linear(layer_sizes[k], layer_sizes[k+1]).double())
            if batchnorm_ind and k != len(layer_sizes)-2:
                self.layers.append(nn.BatchNorm1d(layer_sizes[k+1]).double())
        self.out_layer = nn.Linear(layer_sizes[k+1], out_size).double()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        #import pdb; pdb.set_trace()
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = self.relu(layer(x))
            else:# batchnorm layer
                x = layer(x) 
        if self.out_act == 'linear':
            return self.out_layer(x)
        elif self.out_act == 'sigmoid':
            return self.sigmoid(self.out_layer(x))
    
class NueralNet(object):
    def __init__(self, input_size, h_sizes, out_size = 1, out_act = 'linear', batchnorm_ind = False, n_iter = 100, lr = 0.01, device = 'cpu', verbose = False, patience = 10, weight_decay = 0, loss_fn = nn.MSELoss()):
        self.model = FF_neural_net(input_size, h_sizes, out_size, out_act, batchnorm_ind)
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = weight_decay)
        self.device = 'cpu'
        self.n_iter = n_iter
        self.patience = patience
        self.verbose = verbose
    def train(self, X,y):
        self.model.train()
        pred = self.model(X)
        #import pdb; pdb.set_trace()
        loss = self.loss_fn(pred.squeeze(1), y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def val(self, X, y):
        self.model.eval()
        pred = self.model(X)
        #import pdb; pdb.set_trace()
        error = pred.squeeze(1) - y
        loss = self.loss_fn(pred.squeeze(1), y)
        return loss.item(), error
    
    def fit(self, X, y):
        X = torch.tensor(X).double().to(self.device)
        y = torch.tensor(y).double().to(self.device)
        split = X.shape[0]//4
        y_sorted = torch.sort(y)
        #import pdb; pdb.set_trace()
        index_val = np.linspace(start=0, stop=X.shape[0]-1, num=split, dtype = int)
        index_val = y_sorted[1][index_val]
        y_val = y[index_val]
        X_val = X[index_val,:]
        index_val = torch.isin(y_sorted[1], index_val)
        y_train = y[~index_val]
        X_train = X[~index_val,:]

        min_loss_val = float('inf')
        patience_pass = 0
        list_pred = []
        for i in range(self.n_iter):
            loss_train = self.train(X_train, y_train)
            loss_val, error = self.val(X_val, y_val)
            if min_loss_val > loss_val:
                self.model_best = copy.deepcopy(self.model)
                self.best_error = error
                min_loss_val = loss_val
            else:
                patience_pass += 1
            if patience_pass > self.patience:
                break
            if self.verbose:
                print(f'Loss is {loss} at iteration {i}')
    
    def predict(self, X):
        X = torch.tensor(X).double().to(self.device)
        return self.model_best.forward(X.to(self.device)).squeeze(1).cpu().detach().numpy()
