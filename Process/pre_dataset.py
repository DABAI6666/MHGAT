import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
from collections import Counter

type_len = 6

edge_lst = {}
type_x = []#6*6
# map_index = []
type_edge = []#6*6 * 2
type_3,type_4,root_id,parent_id = [],[],[],[]

type_node,type_index = [],[]

class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, tddroprate=0,budroprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        #self.fold_x 就是树编号[tid]
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)
        #

    def __getitem__(self, index):
        #用了dataloader以后 这个index会是一组index，所以后面返回的数组 是一组batchsize的数组
        id =self.fold_x[index]#利用打乱的index 来取tid
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        # #就是树的边 两个元素 第一个是src 第二个是end
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        # new_edgeindex = edgeindex



        # print(new_edgeindex,'new_edgeindex')
        # print(len(new_edgeindex),'len(new_edgeindex)')
        # print(data['neighbor_lst'],'data[neighbor_lst]')
        # print(len(data['neighbor_lst']),'len(data[neighbor_lst])')
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),#data['root'] 表示了root的的feture
             rootindex=torch.LongTensor([int(data['rootindex'])]),neighbor_row = torch.LongTensor(data['neighbor_row']),
                    neighbor_col = torch.LongTensor(data['neighbor_col']),
                    neighbor_weight = torch.FloatTensor(data['neighbor_weight']))#这个才是rootindex

# neighbor_weight = torch.tensor([data['neighbor_weight']])
#


class BiGraphDataset_Weibo(Dataset):
    def __init__(self, fold_x, treeDic, lower=2, upper=100000, tddroprate=0, budroprate=0,
                 data_path=os.path.join('..', '..', 'data', 'Weibograph')):

        # lst = [3574613797373183,3574311804066048,3918798316397729,3558607675129825,3551685860964318,3475816099213996,3501712684038955]
        # self.fold_x = list(
        #     filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and int(id) not in lst and len(treeDic[id]) <= upper, fold_x))[:10]
        self.fold_x = list(
            filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        # self.fold_x 就是树编号[tid]
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)
        #

    def __getitem__(self, index):
        # 用了dataloader以后 这个index会是一组index，所以后面返回的数组 是一组batchsize的数组
        id = self.fold_x[index]  # 利用打乱的index 来取tid
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        # #就是树的边 两个元素 第一个是src 第二个是end
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex


        return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
                    y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
                    # data['root'] 表示了root的的feture
                    rootindex=torch.LongTensor([int(data['rootindex'])]))  # 这个才是rootindex


def dfs(now ,fa ,rootindex ,root_i):
    if now == rootindex:  # 如果root是bidroot也不用管，因为有5号点
        type_node[now] = 0
        type_index[now] = len(type_x[type_node[now]])  # 当前now点在所在type中的下标
        type_x[type_node[now]].append(now)
        root_id[0].append(rootindex)
    else:


        if now not in edge_lst or len(edge_lst[now]) <= 1:
            type_node[now] = 1  # 当前now点所在type
        else:
            type_node[now] = 2

        # 如果当前是新的4类点，那么往4类新加点，更新type_4 和 type_x[4],type_4存的是4类点的下标，每个下标是一个[]
        # 每个[]用来mau和cnn生成一个点

        type_index[now] = len(type_x[type_node[now]])  # 当前now点在所在type中的下标
        type_x[type_node[now]].append(now)

        root_id[type_node[now]].append(rootindex)
        parent_id[type_node[now]].append(fa)


        #所有非根节点和父节点互联，这是0，1，2之间的连接
        type_edge[type_node[fa]][type_node[now]][0].append(type_index[fa])  # 所有非根节点和父节点互联
        type_edge[type_node[fa]][type_node[now]][1].append(type_index[now])

        # 前面处理 0，1，2类 后面处理3，4类 ，生成并连接 这下面部分都是0，1，2和3，4的连接，不包含0，1，2之间的连接，这一块包含3，4之间的连接 ，0，1，2之间的连接在上面
        #起始type_3 虽然是[]，但是最多只有一个数
        if type_node[fa] == 0 and type_node[now] == 1:#如果fa是 根结点，那么更新所属类型，并把新的类型和根相连
            #新增加3
            type_x[3].append([now])
            type_3[now].append(len(type_x[3]) - 1)
            #自身和3新增加的3互联
            type_edge[1][3][0].append(type_index[now])
            type_edge[1][3][1].append(len(type_x[3]) - 1)
            #父节点和新增加的3互联
            type_edge[0][3][0].append(type_index[fa])
            type_edge[0][3][1].append(len(type_x[3]) - 1)
            #新增加的3和5相连
            type_edge[3][5][0].append(len(type_x[3]) - 1)
            type_edge[3][5][1].append(root_i)
            #新增加的3属于哪个rootindex
            root_id[3].append(rootindex)
            parent_id[3].append(fa)

            ###让5和该树的所有点相连
            type_edge[5][3][0].append(root_i)
            type_edge[5][3][1].append(len(type_x[3]) - 1)
            ###



        elif type_node[fa] == 0 and type_node[now] == 2:
            #新增加4
            type_x[4].append([now])
            type_4[now].append(len(type_x[4]) - 1)
            #自身和新增加的4互联
            type_edge[2][4][0].append(type_index[now])
            type_edge[2][4][1].append(len(type_x[4]) - 1)

            #父节点和新增加的4互联
            type_edge[0][4][0].append(type_index[fa])
            type_edge[0][4][1].append(len(type_x[4]) - 1)

            #增加的4和5相连
            type_edge[4][5][0].append(len(type_x[4]) - 1)
            type_edge[4][5][1].append(root_i)
            #新增加的4属于哪个rootindex
            root_id[4].append(rootindex)
            parent_id[4].append(fa)

            ###让5和该树的所有点相连
            type_edge[5][4][0].append(root_i)
            type_edge[5][4][1].append(len(type_x[4]) - 1)


        elif type_node[fa] == 1 and type_node[now] == 1:  # 3类的非起始节点，属于父节点的3类且与父节点的3，4互联
            # 加入父节点的3，并增加连边
            type_x[3][type_3[fa][0]].append(now)  # 点加入父节点的所在3类
            type_3[now].append(type_3[fa][0])
            type_edge[type_node[now]][3][0].append(type_index[now])
            type_edge[type_node[now]][3][1].append(type_3[fa][0])

            for i in type_4[fa]: #type_4[fa]存的是fa所在4在type_x[4]中的下标
                # 加入父节点属于的4，并添加连边
                type_x[4][i].append(now)
                type_4[now].append(i)
                type_edge[type_node[now]][4][0].append(type_index[now])
                type_edge[type_node[now]][4][1].append(i)

        elif type_node[fa] == 2 and type_node[now] == 1:
            # 那么这个点是新的3类点起始，type_3只有它自己,不加入前面的3，不过会加入父节点的4，并且会和前面最后的4相连/目前是这样
            # 增加3，更新自己属于的3(type_3)
            type_x[3].append([now])
            type_3[now].append(len(type_x[3]) - 1)
            #用当前1更新所在3
            type_edge[1][3][0].append(type_index[now])
            type_edge[1][3][1].append(len(type_x[3]) - 1)
            #把父节点连到当前的3
            type_edge[2][3][0].append(type_index[fa])
            type_edge[2][3][1].append(len(type_x[3]) - 1)
            #新增加的3和5相连
            type_edge[3][5][0].append(len(type_x[3]) - 1)
            type_edge[3][5][1].append(root_i)
            #新增加的3属于哪个rootindex
            root_id[3].append(rootindex)
            parent_id[3].append(fa)

            #让5和所有的点相连：
            type_edge[5][3][0].append(root_i)
            type_edge[5][3][1].append(len(type_x[3]) - 1)


            # if len(type_4) > 0:  # 父节点必然是4,(4,3)连线,这个3会和前面所有的4相连
            #     type_edge[4][3][0].append(type_4[fa][-1])
            #     type_edge[4][3][1].append(type_3[now][-1])





            for i in type_4[fa]:  # 把该节点和所属的4连接，并加入到4 # 父节点必然是4,(4,3)连线,这个3会和前面所有的4相连
                type_x[4][i].append(now)
                type_4[now].append(i)
                type_edge[type_node[now]][4][0].append(type_index[now])
                type_edge[type_node[now]][4][1].append(i)

                type_edge[3][4][0].append(type_3[now][-1])
                type_edge[3][4][1].append(i)



        elif type_node[now] == 2:
            ##把该节点的2 和父所属的4连接，并加入到4,先这么做是因为我要保证type_4最后的数是自己新建的4
            for i in type_4[fa]:
                type_x[4][i].append(now)
                type_4[now].append(i)
                type_edge[type_node[now]][4][0].append(type_index[now])
                type_edge[type_node[now]][4][1].append(i)

            # 增加4，更新自己属于的4(type_4)
            type_x[4].append([now])
            type_4[now].append(len(type_x[4]) - 1)
            type_edge[2][4][0].append(type_index[now])
            type_edge[2][4][1].append(len(type_x[4]) - 1)
            #增加的4和5相连
            type_edge[4][5][0].append(len(type_x[4]) - 1)
            type_edge[4][5][1].append(root_i)

            #新增加的4属于哪个rootindex
            root_id[4].append(rootindex)
            parent_id[4].append(fa)

            #让5和所有的点相连
            type_edge[5][4][0].append(root_i)
            type_edge[5][4][1].append(len(type_x[4]) - 1)

            # 当前节点的4要和 父节点类型(0,1,2）相连
            type_edge[type_node[fa]][4][0].append(type_index[fa])
            type_edge[type_node[fa]][4][1].append(len(type_x[4]) - 1)

            #当前节点的 4要和父节点的3，4互连
            for i in type_3[fa]:
                type_edge[3][4][0].append(i)
                type_edge[3][4][1].append(len(type_x[4]) - 1)
            for i in type_4[fa]:
                type_edge[4][4][0].append(i)
                type_edge[4][4][1].append(len(type_x[4]) - 1)


    type_edge[type_node[now]][5][0].append(type_index[now])  # 所有点都要和5联
    type_edge[type_node[now]][5][1].append(root_i)

    ###树上所有的点都要被5连
    type_edge[5][type_node[now]][0].append(root_i)
    type_edge[5][type_node[now]][1].append(type_index[now])

    if now in edge_lst:
        for v in edge_lst[now]:
            dfs(v, now, rootindex, root_i)



def edge_matrix(edge_index,node_len,root_index,neighbor_row,neighbor_col):
    # edge_index = np.asarray([[0, 1, 2, 3, 4, 4, 5, 5], [1, 2, 3, 4, 5, 6, 7, 8]])
    global edge_lst,type_x,type_edge,type_3,type_4,type_node,type_index,root_id,parent_id
    edge_lst,type_x,type_edge,type_3,type_4,type_node,type_index,root_id,parent_id = {},[],[],[],[],[],[],[],[]

    for i in range(len(edge_index[0])):
        if edge_index[0][i] not in edge_lst:
            edge_lst[edge_index[0][i]] = []
        edge_lst[edge_index[0][i]].append(edge_index[1][i])

    type_3, type_4, type_x,root_id,parent_id= [], [], [],[],[]  # 当前节点所属的type3 index 当前节点所属的 type4 index
    # map_index = []
    # type_x 存放每个类型的点的index，type_edge 是在type_x对应下标的连线
    for i in range(type_len):
        type_x.append([])
        # map_index.append([])

    for i in range(node_len):
        type_3.append([])  # 当前节点所属的type_x[3] index
        type_4.append([])  # 当前节点所属的 type_x[4] index

    for i in range(type_len):
        type_edge.append([])
        root_id.append([])
        parent_id.append([])
        for j in range(type_len):
            type_edge[i].append([])
            type_edge[i][j].append([])
            type_edge[i][j].append([])

    type_node = np.asarray([-1] * node_len)
    type_index = np.asarray([-1] * node_len)

    root_map= {}#用于5的连边
    for root_i, root in enumerate(root_index):
        type_x[5].append(root)
        root_id[5].append(root_index)
        root_map[root] = root_i
        dfs(root, -1, root, root_i)

    for i in range(type_len):
        for j in range(len(type_x[i])):
            type_edge[i][i][0].append(j)
            type_edge[i][i][1].append(j)

    #做5的连边
    user_map = {}
    key_lst = list(Counter(neighbor_row))
    id_map = {}
    for i,j in enumerate(key_lst):
        id_map[j] = i

    for i in range(len(neighbor_col)):
        t_id = id_map[neighbor_row[i]]
        if neighbor_col[i] not in user_map:
            user_map[neighbor_col[i]] = set()
        user_map[neighbor_col[i]].add(t_id)

    for i in user_map:
        tmp_lst = list(user_map[i])
        for j in range(len(tmp_lst)):
            for k in range(len(tmp_lst)):
                if j == k: continue
                type_edge[5][5][0].append(tmp_lst[j])
                type_edge[5][5][1].append(tmp_lst[k])

    tmp_type_edge = []
    for i in range(type_len):
        tmp_type_edge.append([])
        for j in range(type_len):
            tmp_type_edge[i].append(torch.sparse.FloatTensor(torch.LongTensor(type_edge[i][j]),torch.FloatTensor([1.0]* len(type_edge[i][j][0])),torch.Size([len(type_x[i]),len(type_x[j])])))
    #上面一块是5和5之间的连边
    type_edge = tmp_type_edge
    #     for j in range(type_len):
    #         print('i, j :', i, j)
    #         print(type_edge[i][j])
    #
    # print('type_x')
    # for i in range(type_len):
    #     print(type_x[i])

    return type_edge,type_x,root_id,parent_id





