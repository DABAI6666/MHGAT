


import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
from math import sqrt
class MultiheadAttention(nn.Module):

    def __init__(self, input_size, output_size, d_k=16, d_v=16, num_heads=8, is_layer_norm=False, attn_dropout=0.0):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.W_q = nn.Parameter(torch.Tensor(input_size, num_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, num_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, num_heads * d_v))
        self.W_o = nn.Parameter(torch.Tensor(d_v*num_heads, input_size))

        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.linear3 = nn.Linear(input_size, output_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.W_q)
        init.xavier_uniform_(self.W_k)
        init.xavier_uniform_(self.W_v)
        init.xavier_uniform_(self.W_o)
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)
        init.xavier_uniform_(self.linear3.weight)

    def feed_forword_layer(self, X):
        lay1 = F.relu(self.linear1(X))
        lay1 = self.dropout(lay1)

        output = self.linear2(lay1)
        return output

    def scaled_dot_product_attention(self, Q, K, V, key_padding_mask, episilon=1e-6):
        '''
        :param Q: (*, max_q_words, num_heads, input_size)
        :param K: (*, max_k_words, num_heads, input_size)
        :param V: (*, max_v_words, num_heads, input_size)
        :param episilon:
        :return:
        '''
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)

        if key_padding_mask is not None:
            bsz, src_len = Q.size(0) // self.num_heads, Q.size(1)
            tgt_len = V.size(1)
            Q_K = Q_K.view(bsz, self.num_heads, tgt_len, src_len)
            key_padding_mask = key_padding_mask.unsqueeze(dim=1).unsqueeze(dim=2)
            Q_K = Q_K.masked_fill(key_padding_mask, -2 ** 32 + 1)
            Q_K = Q_K.view(bsz * self.num_heads, tgt_len, src_len)

        Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_q_words, max_k_words)
        Q_K_score = self.dropout(Q_K_score)

        V_att = Q_K_score.bmm(V)  # (*, max_q_words, input_size)
        return V_att


    def multi_head_attention(self, Q, K, V, key_padding_mask):
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.num_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.num_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.num_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.num_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.num_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.num_heads, q_len, self.d_v)

        V_att = self.scaled_dot_product_attention(Q_, K_, V_, key_padding_mask)
        V_att = V_att.view(bsz, self.num_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.num_heads*self.d_v)

        output = self.dropout(V_att.matmul(self.W_o)) # (batch_size, max_q_words, input_size)
        return output


    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        '''
        :param query: (batch_size, max_q_words, input_size)
        :param key: (batch_size, max_k_words, input_size)
        :param value: (batch_size, max_v_words, input_size)
        :return:  output: (batch_size, max_q_words, input_size)  same size as Q
        '''
        bsz, src_len, _ = query.size()

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        V_att = self.multi_head_attention(query, key, value, key_padding_mask)

        if self.is_layer_norm:
            X = self.layer_morm(query + V_att)  # (batch_size, max_r_words, embedding_dim)
            output = self.layer_morm(self.feed_forword_layer(X) + X)
        else:
            X = query + V_att
            output = self.feed_forword_layer(X) + X

        output = self.linear3(output)
        return output



class DownSample2x(nn.Sequential):
    def __init__(self, _in, _out):
        super().__init__(
            nn.Conv1d(_in, _out, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )


class SELayer(nn.Module):
    def __init__(self, _in, _hidden=64):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(_in, _hidden),
            nn.PReLU(),
            nn.Linear(_hidden, _in),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class ResConv1d(nn.Module):
    def __init__(self, _in, _out):
        super(ResConv1d, self).__init__()

        self.cal = nn.Sequential(
            nn.Conv1d(_in, _out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(_out),
            nn.ReLU(),
            nn.Conv1d(_out, _out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(_out),
        )
        self.se = SELayer(_out, _out)
        self.conv = nn.Conv1d(_in, _out, kernel_size=1, padding=0, stride=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(_out)

    def forward(self, x):
        res = self.cal(x)
        res = self.se(res)

        x = self.bn(self.conv(x))

        return self.relu(res + x)



class TypeAttention(nn.Module):
    def __init__(self, in_features, idx, hidden_dim):
        super(TypeAttention, self).__init__()
        self.idx = idx
        self.linear = torch.nn.Linear(in_features, hidden_dim) #hidden_dim = 50
        self.a = nn.Parameter(torch.FloatTensor(2 * hidden_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / sqrt(self.a.size(1))
        self.a.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        # inputs size:  node_num * 3 * in_features
        x = self.linear(inputs).transpose(0, 1) #
        self.n = x.size()[0]
        x = torch.cat([x, torch.stack([x[self.idx]] * self.n, dim=0)], dim=2)
        #
        U = torch.matmul(x, self.a).transpose(0, 1)#
        U = F.leaky_relu_(U)#
        weights = F.softmax(U, dim=1) #
        outputs = torch.matmul(weights.transpose(1, 2), inputs).squeeze(1)
        #weights.transpose(1,2).shape(40,1,3) * (40,3,512)
        return outputs, weights

# weight_introduction
def cal_sim_con(ntype,type_edge, x_feature,beta):
    adj, weight_matrix = type_edge
    for i in range(ntype):
        node_lst = []
        pos_lst = []
        for j in range(len(x_feature[i])):
            node_lst.append([])
            pos_lst.append([])
        for k in range(ntype):
            for pos in range(len(adj[i][k][0])):
                u,v = adj[i][k][0][pos],adj[i][k][1][pos]
                node_lst[u].append(x_feature[k][v])
                pos_lst[u].append((i,k,pos))

        for j in range(len(x_feature[i])):
            emb_lst = torch.stack(node_lst[j])

            sim_c = torch.cosine_similarity(x_feature[i][j].reshape(1, -1), emb_lst)
            weight_lst = F.log_softmax(sim_c)

            for w_index in range(len(weight_lst)):
                u,v,pos = pos_lst[j][w_index]
                weight_matrix[u][v][pos] *= beta * weight_lst[w_index]
                    # elif i <= 2

    tmp_type_edge = []
    for i in range(ntype):
        tmp_type_edge.append([])
        for j in range(ntype):
            tmp_type_edge[i].append(
                torch.sparse.FloatTensor(torch.LongTensor(adj[i][j]), torch.FloatTensor(weight_matrix[i][j]),
                                         torch.Size([len(x_feature[i]), len(x_feature[j])])))
            # tmp_type_edge[i].append(torch.sparse.FloatTensor(torch.LongTensor(type_edge[i][j]),torch.FloatTensor([1.0]* len(type_edge[i][j][0])),torch.Size([len(type_x[i]),len(type_x[j])])))

    type_edge = tmp_type_edge
    for i in range(ntype):
        for j in range(ntype):
            type_edge[i][j] = type_edge[i][j]

    return type_edge


class Attention_NodeLevel(nn.Module):
    def __init__(self, dim_features, gamma=0.1):
        super(Attention_NodeLevel, self).__init__()

        self.dim_features = dim_features

        self.a1 = nn.Parameter(torch.zeros(size=(dim_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(dim_features, 1)))
        nn.init.xavier_normal_(self.a1.data, gain=1.414)
        nn.init.xavier_normal_(self.a2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.gamma = gamma

    def forward(self, input1, input2, adj):
        h = input1  # shape:(40,512)
        g = input2  # shape:(40,512)
        N = h.size()[0]
        M = g.size()[0]

        e1 = torch.matmul(h, self.a1).repeat(1, M)  #
        e2 = torch.matmul(g, self.a2).repeat(1, N).t()  #
        e = e1 + e2  # shape(40 ，40）
        e = self.leakyrelu(e)

        zero_vec = -9e15 * torch.ones_like(e)
        if 'sparse' in adj.type():
            # adj(40,40) adj_desne:（40,40)

            adj_dense = adj.to_dense()
            attention = torch.where(adj_dense > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = torch.mul(attention, adj_dense.sum(1).repeat(M, 1).t())
            attention = torch.add(attention * self.gamma,
                                  adj_dense * (1 - self.gamma))
            del (adj_dense)
        else:
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = torch.mul(attention, adj.sum(1).repeat(M, 1).t())
            attention = torch.add(attention * self.gamma, adj.to_dense() * (1 - self.gamma))
        del (zero_vec)

        h_prime = torch.matmul(attention,g)

        return h_prime


class GraphAttentionConvolution(nn.Module):
    def __init__(self, in_features_list, out_features, ntype, bias=True, gamma=0.1):
        super(GraphAttentionConvolution, self).__init__()
        self.in_features_list = in_features_list
        self.out_features = out_features
        self.weights = nn.ParameterList()
        self.ntype = ntype
        for i in range(self.ntype):
            cache = nn.Parameter(torch.FloatTensor(in_features_list[i], out_features))
            nn.init.xavier_normal_(cache.data, gain=1.414)
            self.weights.append(cache)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            stdv = 1. / sqrt(out_features)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

        self.att_list = nn.ModuleList()
        for i in range(self.ntype):
            self.att_list.append(Attention_NodeLevel(out_features, gamma))

    def forward(self, inputs_list, adj_list, global_W=None):
        h = []
        for i in range(self.ntype):
            h.append(torch.spmm(inputs_list[i], self.weights[i]))
        # h[0].shape : [40,512] h[1].shape[12,512],h[2].shape[93,512]
        if global_W is not None:  # global_w = none
            for i in range(self.ntype):
                h[i] = (torch.spmm(h[i], global_W))
        outputs = []
        for t1 in range(self.ntype):
            x_t1 = []
            for t2 in range(self.ntype):
                # adj has no non-zeros
                if len(adj_list[t1][t2]._values()) == 0:
                    x_t1.append(torch.zeros(adj_list[t1][t2].shape[0], self.out_features, device=self.bias.device))
                    continue

                if self.bias is not None:
                    x_t1.append(self.att_list[t1](h[t1], h[t2], adj_list[t1][t2]) + self.bias)
                else:
                    x_t1.append(self.att_list[t1](h[t1], h[t2], adj_list[t1][t2]))  #
            outputs.append(x_t1)

        return outputs

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj, global_W = None):

        if len(adj._values()) == 0:
            return torch.zeros(adj.shape[0], self.out_features, device=inputs.device)

        support = torch.spmm(inputs, self.weight)
        #self.weight.shape:(512,10)
        if global_W is not None:
            support = torch.spmm(support, global_W)
        output = torch.spmm(adj, support)#torch.size([40,10])
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Attention(nn.Module):

    def __init__(self, in_features, hidden_size):
        super(Attention, self).__init__()
        self.linear1 = nn.Linear(in_features*2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, K, V, mask = None):
        '''
        :param K: (batch_size, d)
        :param V: (batch_size, hist_len, d)
        :return: (batch_size, d)
        '''
        K = K.unsqueeze(dim=1).expand(V.size())
        fusion = torch.cat([K, V], dim=-1)

        fc1 = self.activation(self.linear1(fusion))
        score = self.linear2(fc1)

        if mask is not None:
            mask = mask.unsqueeze(dim=-1)
            score = score.masked_fill(mask, -2 ** 32 + 1)

        alpha = F.softmax(score, dim=1)
        alpha = self.dropout(alpha)
        att = (alpha * V).sum(dim=1)
        return att


class FusionAttentionUnit(nn.Module):
    def __init__(self, in_features, out_features, user_features):
        super(FusionAttentionUnit, self).__init__()
        self.linear_doc = nn.Linear(in_features, out_features)
        self.linear_user = nn.Linear(user_features, out_features)
        self.W = nn.Parameter(torch.FloatTensor(out_features, out_features))
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear_doc.weight)
        init.xavier_normal_(self.linear_doc.weight)
        init.xavier_normal_(self.W)

    def forward(self, X_doc, X_user):
        '''
        :param X_doc: (bsz, max_sents, in_features)
        :param X_user: (bsz, max_sents, D)
        :return:
        '''
        X_doc = self.linear_doc(X_doc)
        X_user = self.linear_user(X_user)

        X_doc = X_doc * F.sigmoid(X_user)  # (bsz, max_sents, in_features)
        X_user = X_user * F.sigmoid(X_doc) # (bsz, max_sents, in_features)
        attentive_mat = F.tanh(torch.einsum("bsd,dd,brd->bsr", X_doc, self.W, X_user) )  # (bsz, max_sents, max_sents)

        score_d = F.softmax(attentive_mat.mean(2), dim=1).unsqueeze(-1)
        score_u = F.softmax(attentive_mat.mean(1), dim=1).unsqueeze(-1)

        attention_d = torch.sum(score_d*X_doc, dim=1)
        attention_u = torch.sum(score_u*X_user, dim=1)
        return attention_d, attention_u