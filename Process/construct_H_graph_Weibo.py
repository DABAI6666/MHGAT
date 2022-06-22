import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
from collections import Counter

type_len = 6




def dfs(now, fa, rootindex, root_i,pr,edge_lst,type_x,type_edge,type_3, type_4, root_id, parent_id,type_node,type_index,num_node):
    num_node = num_node + 1
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

        # 所有非根节点和父节点互联，这是0，1，2之间的连接
        type_edge[type_node[fa]][type_node[now]][0].append(type_index[fa])  # 所有非根节点和父节点互联
        type_edge[type_node[fa]][type_node[now]][1].append(type_index[now])

        # 前面处理 0，1，2类 后面处理3，4类 ，生成并连接 这下面部分都是0，1，2和3，4的连接，不包含0，1，2之间的连接，这一块包含3，4之间的连接 ，0，1，2之间的连接在上面
        # 起始type_3 虽然是[]，但是最多只有一个数
        if type_node[fa] == 0 and type_node[now] == 1:  # 如果fa是 根结点，那么更新所属类型，并把新的类型和根相连
            # 新增加3
            type_x[3].append([now])
            type_3[now].append(len(type_x[3]) - 1)
            # 自身和3新增加的3互联
            type_edge[1][3][0].append(type_index[now])
            type_edge[1][3][1].append(len(type_x[3]) - 1)
            # 父节点和新增加的3互联
            type_edge[0][3][0].append(type_index[fa])
            type_edge[0][3][1].append(len(type_x[3]) - 1)
            # 新增加的3和5相连
            type_edge[3][5][0].append(len(type_x[3]) - 1)
            type_edge[3][5][1].append(root_i)
            # 新增加的3属于哪个rootindex
            root_id[3].append(rootindex)
            parent_id[3].append(fa)

            ###让5和该树的所有点相连
            type_edge[5][3][0].append(root_i)
            type_edge[5][3][1].append(len(type_x[3]) - 1)
            ###



        elif type_node[fa] == 0 and type_node[now] == 2:
            # 新增加4
            type_x[4].append([now])
            type_4[now].append(len(type_x[4]) - 1)
            # 自身和新增加的4互联
            type_edge[2][4][0].append(type_index[now])
            type_edge[2][4][1].append(len(type_x[4]) - 1)

            # 父节点和新增加的4互联
            type_edge[0][4][0].append(type_index[fa])
            type_edge[0][4][1].append(len(type_x[4]) - 1)

            # 增加的4和5相连
            type_edge[4][5][0].append(len(type_x[4]) - 1)
            type_edge[4][5][1].append(root_i)
            # 新增加的4属于哪个rootindex
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

            for i in type_4[fa]:  # type_4[fa]存的是fa所在4在type_x[4]中的下标
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
            # 用当前1更新所在3
            type_edge[1][3][0].append(type_index[now])
            type_edge[1][3][1].append(len(type_x[3]) - 1)

            # 新增加的3和5相连
            type_edge[3][5][0].append(len(type_x[3]) - 1)
            type_edge[3][5][1].append(root_i)
            # 新增加的3属于哪个rootindex
            root_id[3].append(rootindex)
            parent_id[3].append(fa)

            # add新增加的3 和 父节点的3连接
            for i in type_3[fa]:
                type_edge[3][3][0].append(i)
                type_edge[3][3][1].append(len(type_x[3]) - 1)
            # add

            # 让5和所有的点相连：
            type_edge[5][3][0].append(root_i)
            type_edge[5][3][1].append(len(type_x[3]) - 1)


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
            # add该节点加入父节点所在的3
            if type_node[fa] == 1:
                for i in type_3[fa]:
                    type_x[3][i].append(now)
                    type_3[now].append(i)
                    type_edge[type_node[now]][3][0].append(type_index[now])
                    type_edge[type_node[now]][3][1].append(i)
            # add

            # 增加4，更新自己属于的4(type_4)
            type_x[4].append([now])
            type_4[now].append(len(type_x[4]) - 1)
            type_edge[2][4][0].append(type_index[now])
            type_edge[2][4][1].append(len(type_x[4]) - 1)
            # 增加的4和5相连
            type_edge[4][5][0].append(len(type_x[4]) - 1)
            type_edge[4][5][1].append(root_i)

            # 新增加的4属于哪个rootindex
            root_id[4].append(rootindex)
            parent_id[4].append(fa)

            # 让5和所有的点相连
            type_edge[5][4][0].append(root_i)
            type_edge[5][4][1].append(len(type_x[4]) - 1)

            # 当前节点的4要和 父节点类型(0,1,2）相连
            type_edge[type_node[fa]][4][0].append(type_index[fa])
            type_edge[type_node[fa]][4][1].append(len(type_x[4]) - 1)

            # 当前节点的 4要和父节点的3，4互连
            # 和add有关
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
            # if num_node >= 600: break
            p = random.random()
            if (p > pr):
                edge_lst,type_x,type_edge,type_3, type_4, root_id, parent_id,type_node,type_index,num_node = \
                    dfs(v, now, rootindex, root_i,pr,edge_lst,type_x,type_edge,type_3, type_4, root_id, parent_id,type_node,type_index,num_node)
#add
    if now in edge_lst:
        for v in edge_lst[now]:
            for k in edge_lst[now]:
                if v != k:
                    for i in type_3[v]:
                        for j in type_3[k]:
                            type_edge[3][3][0].append(i)
                            type_edge[3][3][1].append(j)
                        for j in type_4[k]:
                            type_edge[3][4][0].append(i)
                            type_edge[3][4][1].append(j)
                    for i in type_4[v]:
                        for j in type_3[k]:
                            type_edge[4][3][0].append(i)
                            type_edge[4][3][1].append(j)
                        for j in type_4[k]:
                            type_edge[4][4][0].append(i)
                            type_edge[4][4][1].append(j)
    return edge_lst,type_x,type_edge,type_3, type_4, root_id, parent_id,type_node,type_index,num_node
#add


def edge_matrix(pr,Batch_data):
    edge_index, node_len, root_index = Batch_data.edge_index.tolist(), len(Batch_data.batch),Batch_data.rootindex.tolist()
    # edge_index = np.asarray([[0, 1, 2, 3, 4, 4, 5, 5], [1, 2, 3, 4, 5, 6, 7, 8]])
    edge_lst, type_x, type_edge, type_3, type_4, type_node, type_index, root_id, parent_id,num_node = {}, [], [], [], [], [], [], [], [],0
    # 当前节点所属的type3 index 当前节点所属的 type4 index

    for i in range(len(edge_index[0])):
        if edge_index[0][i] not in edge_lst:
            edge_lst[edge_index[0][i]] = []
        edge_lst[edge_index[0][i]].append(edge_index[1][i])

     # 当前节点所属的type3 index 当前节点所属的 type4 index
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

    root_map = {}  # 用于5的连边
    for root_i, root in enumerate(root_index):
        num_node = 0
        type_x[5].append(root)
        root_id[5].append(root)
        root_map[root] = root_i
        edge_lst,type_x,type_edge,type_3, type_4, root_id, parent_id,type_node,type_index,num_node = \
            dfs(root, -1, root, root_i,pr,edge_lst,type_x,type_edge,type_3, type_4, root_id, parent_id,type_node,type_index,num_node)
    for i in range(type_len):
        for j in range(len(type_x[i])):
            type_edge[i][i][0].append(j)
            type_edge[i][i][1].append(j)




    weights = []
    for i in range(type_len):
        weights.append([])
        for j in range(type_len):
            weights[i].append([])

    for i in range(type_len):
        for j in range(type_len):
            mp = {}
            for k, h in zip(type_edge[i][j][0], type_edge[i][j][1]):
                if h not in mp:
                    mp[h] = 0
                mp[h] += 1
            for k, h in zip(type_edge[i][j][0], type_edge[i][j][1]):
                weights[i][j].append(1.0 / mp[h])

    tmp_type_edge = []
    for i in range(type_len):
        tmp_type_edge.append([])
        for j in range(type_len):
            tmp_type_edge[i].append(
                torch.sparse.FloatTensor(torch.LongTensor(type_edge[i][j]), torch.FloatTensor(weights[i][j]),
                                         torch.Size([len(type_x[i]), len(type_x[j])])))

    type_edge = tmp_type_edge


    return type_edge, type_x, root_id, parent_id





