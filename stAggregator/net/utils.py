#!/usr/bin/env python

import numpy as np
import torch


def onehot(y, n):
    """
    Make the input tensor one hot tensors
    
    Parameters
    ----------
    y
        input tensors
    n
        number of classes
        
    Return
    ------
    Tensor
    """
    if (y is None) or (n < 2):
        return None
    assert torch.max(y).item() < n
    y = y.view(y.size(0), 1)
    y_cat = torch.zeros(y.size(0), n).to(y.device)
    y_cat.scatter_(1, y.data, 1)
    return y_cat


class EarlyStopping:
    """
    Early stops the training if loss doesn't improve after a given patience.
    """

    def __init__(self, patience=10, switch=True, verbose=False, checkpoint_file=''):
        """
        Parameters
        ----------
        patience 
            How long to wait after last time loss improved. Default: 10
        verbose
            If True, prints a message for each loss improvement. Default: False
        """
        self.patience = patience
        self.switch = switch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.checkpoint_file = checkpoint_file

    def __call__(self, loss, model):
        if not self.switch:
            self.early_stop = False
            return

        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.checkpoint_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''
        Saves model when loss decrease.
        '''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_file)
        self.loss_min = loss


def clear_fig(fig):
    if fig:
        fig.axes[0].set_xlabel(None)
        fig.axes[0].set_ylabel(None)
        fig.tight_layout()
    else:
        pass
    return fig


def permutation(feature):
    # fix_seed(FLAGS.random_seed)
    # node permutrated
    # # 1
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    # 2
    # perm = torch.randperm(feature.shape[0])
    # feature_permutated = feature[perm]
    # feature permutrated
    # # 1
    # feature_id = np.arange(feature.shape[1])
    # feature_id = np.random.permutation(feature_id)
    # feature_permutated = feature_permutated[:, feature_id]
    # 2
    # perm = torch.randperm(feature.shape[1])
    # feature_permutated = feature_permutated[:, perm]
    return feature_permutated
