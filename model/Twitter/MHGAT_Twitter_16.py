import sys, os

sys.path.append('/home/lgy/MGAT/')

from Process.process import *
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.data_split import *
from Process.construct_H_graph import edge_matrix
from model.model import MHGAT
from tools.evaluate import *

type_len = 6
edge_lst = {}
type_x = []  # 6*6
# map_index = []
type_edge = []  # 6*6 * 2
type_3, type_4 = [], []
type_node = []
parent_x = []
type_index = []


def check_result(epoch, test_loader, model, testdroprate, mode):
    temp_val_losses = []
    temp_val_accs = []
    temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
    temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
    temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
    temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    model.eval()
    tqdm_test_loader = tqdm(test_loader)
    all_3, all_4 = 0, 0
    for Batch_data in tqdm_test_loader:
        type_edge, type_x, root_id, parent_id = edge_matrix(Batch_data.edge_index.tolist(), len(Batch_data.batch),
                                                            Batch_data.rootindex.tolist(),
                                                            Batch_data.neighbor_row.tolist(),
                                                            Batch_data.neighbor_col.tolist(), testdroprate)

        edge_index = Batch_data.edge_index.to(device)

        all_3 = all_3 + len(type_x[3])
        all_4 = all_4 + len(type_x[4])

        len_3, len_4 = [], []
        for i in type_x[3]:
            len_3.append(len(i))
        for i in type_x[4]:
            len_4.append(len(i))

        for i in range(type_len):
            for j in range(type_len):
                type_edge[i][j] = type_edge[i][j].to(device)
        x = Batch_data.x.to(device)
        y = Batch_data.y.to(device)

        val_out = model(type_edge, x, edge_index, type_x, root_id, parent_id, Batch_data.batch, Batch_data.rootindex)

        val_loss = F.nll_loss(val_out, y)
        temp_val_losses.append(val_loss.item())
        _, val_pred = val_out.max(dim=1)
        correct = val_pred.eq(y).sum().item()
        val_acc = correct / len(y)
        Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
            val_pred, y)
        temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
            Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
        temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
            Recll2), temp_val_F2.append(F2), \
        temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
            Recll3), temp_val_F3.append(F3), \
        temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
            Recll4), temp_val_F4.append(F4)
        temp_val_accs.append(val_acc)
    print(all_3, 'all_3', all_4, 'all_4')
    print("Epoch {:05d} | Loss {:.4f}| {}_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses), mode,
                                                                  np.mean(temp_val_accs)))

    # res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
    #        'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
    #                                                np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
    #        'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
    #                                                np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
    #        'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
    #                                                np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
    #        'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
    #                                                np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
    # print('results:', res)
    return np.mean(temp_val_Acc_all), np.mean(temp_val_F1), np.mean(temp_val_F2), np.mean(temp_val_F3), np.mean(
        temp_val_F4)


def train_MHGAT(treeDic, x_val, x_test, x_train, traindroprate, testdroprate, lr, weight_decay, patience, n_epochs,
                batchsize, dataname, iter):
    model = MHGAT(in_feats=5000, hid_feats=128, ntype=type_len, nclass=4, dropout=0.8).to(device)
    print(model)
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses = []
    train_accs = []
    accs = 0.0
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(n_epochs):
        model.train()
        traindata_list, testdata_list, valdata_list = loadBiData(dataname, treeDic, x_train, x_test, x_val)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        val_loader = DataLoader(valdata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        # for _ in train_loader:
        #     print(_)
        tqdm_train_loader = tqdm(train_loader)
        # train
        all_3, all_4 = 0, 0
        for Batch_data in tqdm_train_loader:
            type_edge, type_x, root_id, parent_id = edge_matrix(Batch_data.edge_index.tolist(), len(Batch_data.batch),
                                                                Batch_data.rootindex.tolist(),
                                                                Batch_data.neighbor_row.tolist(),
                                                                Batch_data.neighbor_col.tolist(), traindroprate)

            for i in range(type_len):
                for j in range(type_len):
                    type_edge[i][j] = type_edge[i][j].to(device)
            # Batch_data.to(device)
            edge_index = Batch_data.edge_index.to(device)
            x = Batch_data.x.to(device)
            y = Batch_data.y.to(device)

            out_labels = model(type_edge, x, edge_index, type_x, root_id, parent_id, Batch_data.batch,
                               Batch_data.rootindex)
            finalloss = F.nll_loss(out_labels, y)
            loss = finalloss
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(y).sum().item()
            train_acc = correct / len(y)
            avg_acc.append(train_acc)
            print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,
                                                                                                               epoch,
                                                                                                               batch_idx,
                                                                                                               loss.item(),
                                                                                                               train_acc))
            batch_idx = batch_idx + 1

        print(all_3, 'all_3', all_4, 'all_4')
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        test_accs, F1, F2, F3, F4 = check_result(epoch, test_loader, model, testdroprate, 'test')

        if test_accs > accs:
            accs = test_accs
            val_accs, F1, F2, F3, F4 = check_result(epoch, val_loader, model, valdroprate, 'val')
            early_stopping(val_accs, F1, F2, F3, F4, model, 'MHGAT', dataname)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    del (model)
    return early_stopping.accs, early_stopping.F1, early_stopping.F2, early_stopping.F3, early_stopping.F4


lr = 0.0001
weight_decay = 1e-4
patience = 20
n_epochs = 160
batchsize = 64
traindroprate = 0.01
testdroprate = 0
valdroprate = 0

# datasetname=sys.argv[1] #"Twitter15"、"Twitter16"
datasetname = 'Twitter15'
# iterations=int(sys.argv[2])
iterations = 50
model = "GCN"
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
test_accs = []
NR_F1 = []
FR_F1 = []
TR_F1 = []
UR_F1 = []
for iter in range(iterations):
    x_val, x_test, x_train = splitData(datasetname)
    treeDic = loadTree(datasetname)
    accs, F1, F2, F3, F4 = train_MHGAT(treeDic, x_val, x_test, x_train,
                                       traindroprate, testdroprate,
                                       lr, weight_decay,
                                       patience,
                                       n_epochs,
                                       batchsize,
                                       datasetname,
                                       iter)

    print('val acc: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}'.format(accs, F1, F2, F3, F4))

    test_accs.append(accs)
    NR_F1.append(F1)
    FR_F1.append(F2)
    TR_F1.append(F3)
    UR_F1.append(F4)
print("Total_Val_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
    sum(test_accs) / iterations, sum(NR_F1) / iterations, sum(FR_F1) / iterations, sum(TR_F1) / iterations,
    sum(UR_F1) / iterations))


