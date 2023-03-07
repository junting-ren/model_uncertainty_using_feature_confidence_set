import torch
from torch import nn
import numpy as np
import copy

class FF_neural_net(nn.Module):
    def __init__(self, input_size, h_sizes, out_size = 1, batchnorm_ind = False):
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
        for k in range(len(layer_sizes)-1):
            self.layers.append(nn.Linear(layer_sizes[k], layer_sizes[k+1]).double())
            if batchnorm_ind and k != len(layer_sizes)-2:
                self.layers.append(nn.BatchNorm1d(layer_sizes[k+1]).double())
        self.out_layer = nn.Linear(layer_sizes[k+1], out_size).double()
    
    def forward(self, x):
        #import pdb; pdb.set_trace()
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = self.relu(layer(x))
            else:# batchnorm layer
                x = layer(x) 
        return self.out_layer(x)

def train(X, y, model, loss_fn, optimizer, device = 'cpu'):
    model.train()
    X, y = X.to(device), y.to(device)
    pred = model(X)
    #import pdb; pdb.set_trace()
    loss = loss_fn(pred.squeeze(1), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def val(X, y, model, loss_fn, device):
    model.eval()
    X, y = X.to(device), y.to(device)
    pred = model(X)
    #import pdb; pdb.set_trace()
    error = pred.squeeze(1) - y
    loss = loss_fn(pred.squeeze(1), y)
    return loss.item(), error


def nn_fit_predict(model, y, X, X_test, n_iter = 100, lr = 0.01, device = 'cpu', verbose = False, patience = 10, weight_decay = 0, mean_num = 1, return_best = False):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    # split the training data into 20% validation and 80% training
    #import pdb; pdb.set_trace()
    split = X.shape[0]//3
    y_sorted = torch.sort(y)
    index_val = np.linspace(start=0, stop=X.shape[0]-1, num=split, dtype = int)
    y_val = y[index_val]
    X_val = X[index_val,:]
    index_val = torch.isin(y, y_val)
    y_train = y[~index_val]
    X_train = X[~index_val,:]
    
    # index = np.random.permutation(X.shape[0])
    # split = X.shape[0]//3
    # train_idx, val_idx = index[split:], index[:split]
    # X_train = X[train_idx,:]
    # y_train = y[train_idx]
    # X_val = X[val_idx,:]
    # y_val = y[val_idx]
    
    min_loss_val = float('inf')
    patience_pass = 0
    list_pred = []
    for i in range(n_iter):
        loss_train = train(X_train, y_train, model, loss_fn, optimizer, device = device)
        loss_val, error = val(X_val, y_val, model, loss_fn, device = device)
        list_pred.append(model.forward(X_test.to(device)).squeeze(1).cpu().detach().numpy())
        if min_loss_val > loss_val:
            model_best = copy.deepcopy(model)
            best_error = error
            min_loss_val = loss_val
        else:
            patience_pass += 1
        if patience_pass > patience:
            break
        if verbose:
            print(f'Loss is {loss} at iteration {i}')
    #import pdb; pdb.set_trace()
    if return_best:
        return model_best.forward(X_test.to(device)).squeeze(1).cpu().detach().numpy(), model_best, best_error.cpu().detach().numpy()
    else:
        return model_best.forward(X_test.to(device)).squeeze(1).cpu().detach().numpy(),best_error.cpu().detach().numpy()
# def nn_fit_predict(model, y, X, X_test, n_iter = 100, lr = 0.01, device = 'cpu', verbose = False, patience = 10, weight_decay = 0):
#     loss_fn = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
#     # split the training data into 20% validation and 80% training
#     #import pdb; pdb.set_trace()
#     index = np.random.permutation(X.shape[0])
#     split = X.shape[0]//5
#     train_idx, val_idx = index[split:], index[:split]
#     X_train = X[train_idx,:]
#     y_train = y[train_idx]
#     X_val = X[val_idx,:]
#     y_val = y[val_idx]
    
#     min_loss_val = float('inf')
#     patience_pass = 0
#     for i in range(n_iter):
#         loss_train = train(X_train, y_train, model, loss_fn, optimizer, device = device)
#         loss_val = val(X_val, y_val, model, loss_fn, device = device)
#         if min_loss_val > loss_val:
#             model_best = copy.deepcopy(model)
#         else:
#             patience_pass += 1
#         if patience_pass > patience:
#             break
#         if verbose:
#             print(f'Loss is {loss} at iteration {i}')
#     return model_best.forward(X_test.to(device)).squeeze(1).cpu().detach().numpy()