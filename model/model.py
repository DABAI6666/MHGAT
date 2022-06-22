import torch.nn as nn
from torch_geometric.nn import GCNConv
import numpy as np
import torch,copy
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from model.model_block import *





class MHGAT(nn.Module):
    def __init__(self, in_feats, hid_feats ,ntype,nclass,dropout):
        super(MHGAT, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(2*hid_feats,hid_feats)
        self.mau_len_1 = 13
        self.mau_len_2 = 30
        self.ntype = ntype
        self.nclass = nclass
        # self.self_atten = SelfAttention(in_feats, in_feats, in_feats)
        self.mh_attention = nn.ModuleList()
        self.mh_attention.append(MultiheadAttention(input_size=hid_feats, output_size=hid_feats))
        self.mh_attention.append(MultiheadAttention(input_size=hid_feats, output_size=hid_feats))


        in_feats_list = [hid_feats,hid_feats,hid_feats,hid_feats,hid_feats,hid_feats]
        self.gc1 = GraphAttentionConvolution(in_feats_list, hid_feats,ntype, gamma=0.1)
        self.at1 = nn.ModuleList()
        self.at2 = nn.ModuleList()
        for t in range(self.ntype):
            self.at1.append(TypeAttention(hid_feats, t, hid_feats))
            self.at2.append(TypeAttention(hid_feats, t, hid_feats))
        self.gc2 = nn.ModuleList()
        self.gc2.append(GraphConvolution(hid_feats, hid_feats, bias=True))
        self.nonlinear = F.relu_
        self.dropout = dropout

        self.attention_1_2 = nn.ModuleList()
        self.attention_1_2.append(Attention(hid_feats,hid_feats))
        self.attention_1_2.append(Attention(hid_feats,hid_feats))


        self.attention = nn.ModuleList()
        self.attention.append(Attention(hid_feats,hid_feats))
        self.attention.append(Attention(hid_feats,hid_feats))
        self.linear_fuse = nn.ModuleList()
        self.linear_fuse.append(nn.Linear(2*hid_feats,1))
        self.linear_fuse.append(nn.Linear(2*hid_feats,1))
        self.linear_fuse_1_2 = nn.ModuleList()
        self.linear_fuse_1_2.append(nn.Linear(2*hid_feats, 1))
        self.linear_fuse_1_2.append(nn.Linear(2*hid_feats, 1))
        self.fin_linear = nn.Linear(hid_feats,4)


    def forward(self,type_edge,x,edge_index,type_x,root_id,parent_id,batch,rootindex):

        node_len = len(x)

        x = self.conv1(x, edge_index)

        rootindex = rootindex
        root_extend = torch.zeros(len(batch), x.size(1)).to(DEVICE)
        batch_size = max(batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(batch, num_batch))
            root_extend[index] = x[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)


        x0 = []
        x0.append(torch.stack([x[id] for id in type_x[0]]))#0类特征


        for i in range(2):#1，2类特征
            if len(type_x[1 + i]) == 0:
                type_x[1 + i].append([node_len])
                root_id[1 + i].append(node_len)
                parent_id[1+i].append(node_len)
            tmp_lst = type_x[1+i]
            tmp_lst = torch.eye(node_len + 1, node_len)[tmp_lst].to(DEVICE)
            tmp_lst = torch.matmul(tmp_lst, x)
            tmp_root = torch.from_numpy(np.array(root_id[1+i])).to(DEVICE)
            # print(tmp_root)
            tmp_root = torch.eye(node_len + 1,node_len)[tmp_root].to(DEVICE)
            tmp_root = torch.matmul(tmp_root,x)

            tmp_parent = torch.from_numpy(np.array(parent_id[1 + i])).to(DEVICE)
            tmp_parent = torch.eye(node_len + 1, node_len)[tmp_parent].to(DEVICE)
            tmp_parent = torch.matmul(tmp_parent, x)

            tmp_id = torch.cat([tmp_parent.unsqueeze(1),tmp_root.unsqueeze(1)],dim = 1)

            tmp_att = self.attention_1_2[i](tmp_lst,tmp_id)
            X_fuse = torch.cat([tmp_lst, tmp_att], dim=-1)
            alpha = torch.sigmoid(self.linear_fuse_1_2[i](X_fuse))
            tmp_lst = alpha * tmp_lst + (1 - alpha) * tmp_att
            x0.append(tmp_lst)

        for i in range(2):
            tmp_lst = []
            if len(type_x[3+i]) == 0:
                type_x[3+i].append([node_len])
                root_id[3 + i].append(node_len)

            for j in type_x[3 + i]:
                if i == 0:
                    tmp_lst.append(max(0,self.mau_len_1 - len(j)) * [node_len] + j[:self.mau_len_1])
                if i == 1:
                    tmp_lst.append(np.random.choice(j,self.mau_len_2))
                # tmp_lst.append(j)


            tmp_lst = torch.from_numpy(np.asarray(tmp_lst)).to(DEVICE)
            tmp_lst = torch.eye(node_len + 1, node_len)[tmp_lst].to(DEVICE)
            tmp_lst = torch.matmul(tmp_lst,x)
            tmp_lst = self.mh_attention[i](tmp_lst,tmp_lst,tmp_lst)

            tmp_root = torch.from_numpy(np.array(root_id[3+i])).to(DEVICE)
            tmp_root = torch.eye(node_len + 1,node_len)[tmp_root].to(DEVICE)
            tmp_root = torch.matmul(tmp_root,x)
            # print(i,root_id[i].shape,tmp_lst.shape,'i,root_id[i].shape,tmp_lst.shape')
            X_att = self.attention[i](tmp_root,tmp_lst)
            X_fuse = torch.cat([tmp_root,X_att],dim = -1)
            alpha = torch.sigmoid(self.linear_fuse[i](X_fuse))
            tmp_lst = alpha*tmp_root + (1-alpha) * X_att

            x0.append(tmp_lst)

        x0.append(torch.stack([x[id] for id in type_x[0]]))#5类特征

        del(tmp_id)
        del(tmp_root)
        del(tmp_parent)
        del(edge_index)
        del(x)
        del(tmp_lst)
        #type_edge = cal_sim_con(self.ntype,type_edge, x0, self.beta)
        torch.cuda.empty_cache()
        x1 = [None for _ in range(self.ntype)]
        x1_in = self.gc1(x0, type_edge)
        del(x0)

        for t1 in range(len(x1_in)):
            x_t1 = x1_in[t1]
            x_t1, weights = self.at1[t1](torch.stack(x_t1, dim=1))

            x_t1 = self.nonlinear(x_t1)
            x_t1 = F.dropout(x_t1,self.dropout, training = self.training)
            x1[t1] = x_t1


        x2 = [None for _ in range(self.ntype)]

        for t1 in range(self.ntype):
            x_t1 = []
            for t2 in range(self.ntype):
                if type_edge[t1][t2] is None:
                    continue
                idx = 0
                x_t1.append(self.gc2[idx](x1[t2],type_edge[t1][t2]))

            x_t1, weights = self.at2[t1](torch.stack(x_t1,dim = 1))

            x2[t1] = x_t1
            # x2[t1] = F.log_softmax(x_t1,dim = 1)


        return F.log_softmax(self.fin_linear(x2[5]))




















