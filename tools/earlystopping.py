import numpy as np
import torch,tqdm
import torch.nn.functional as F
type_len = 6
from tools.evaluate import *
from Process.construct_H_graph import edge_matrix
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):

        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """

        self.accs = None
        self.test_accs = None
        self.test_ac = 0.0
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.F1 = 0
        self.F2 = 0
        self.F3 = 0
        self.F4 = 0
        self.val_loss_min = np.Inf


    def __call__(self, accs,F1,F2,F3,F4,model,modelname,str):

        if self.accs is None:
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
            self.counter = 0
            self.save_checkpoint(model,modelname,str)
        elif accs < self.accs:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                # print("BEST Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}"
                #       .format(self.accs,self.F1,self.F2,self.F3,self.F4))
        else:
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
            self.save_checkpoint(model,modelname,str)
            self.counter = 0

    def save_checkpoint(self, model,modelname,str):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min,val_loss))
        torch.save(model.state_dict(),modelname+str+'.m')
        # self.val_loss_min = val_loss








