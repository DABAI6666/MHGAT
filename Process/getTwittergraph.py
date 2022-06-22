# -*- coding: utf-8 -*-
import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
from collections import Counter
import sys
# cwd=os.getcwd()
cwd = '../'
class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None

def str2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq=float(pair.split(':')[1])
        index=int(pair.split(':')[0])
        if index<=5000:
            wordFreq.append(freq)
            wordIndex.append(index)
    return wordFreq, wordIndex

def constructMat(tree):
    index2node = {}
    for i in tree:
        #因为是dict i是下标，下面是初始化这么多个index2node点
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'])
        nodeC.index = wordIndex
        nodeC.word = wordFreq

        ## not root node ##
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)



        ## root node ##
        else:
            rootindex=indexC-1 #rootindex所在范围是1～5000 要映射回 0～4999 所以-1
            root_index=nodeC.index
            root_word=nodeC.word
    rootfeat = np.zeros([1, 5000]) #一行5000列
    if len(root_index)>0:
        rootfeat[0, np.array(root_index)] = np.array(root_word) #把root_index 所在列 赋值，权重=频数
    matrix=np.zeros([len(index2node),len(index2node)])
    row=[]
    col=[]
    x_word=[]
    x_index=[]
    #因为上面定义node的时候是 index范围是1到5000 不是从0开始的，所以下面都要加1，但是matrix 不加1是因为要映射到邻接矩阵 0～4999内
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i+1].children != None and index2node[index_j+1] in index2node[index_i+1].children:
                matrix[index_i][index_j]=1
                row.append(index_i)
                col.append(index_j)
        x_word.append(index2node[index_i+1].word) #把index1～5000 映射到0～4999了
        x_index.append(index2node[index_i+1].index)
    edgematrix=[row,col]
    #返回的 一棵树和 节点顺序index对对应的x_word,x_index, 还有边的邻接矩阵，root的特征（1*5000），root的index
    return x_word, x_index, edgematrix,rootfeat,rootindex#add

def getfeature(x_word,x_index):
    x = np.zeros([len(x_index), 5000])
    for i in range(len(x_index)):
        if len(x_index[i])>0:
            x[i, np.array(x_index[i])] = np.array(x_word[i])
    return x

def main(obj):
    neighbor_list = {}
    neighbor_weight = {}
    graphPath = os.path.join(cwd,'data/' + obj +'/'+obj+ '_graph.txt')
    #
    with open(graphPath, 'r', encoding='utf-8') as input:
        for line in input.readlines():
            tmp = line.strip().split()
            src = tmp[0]
            neighbor_list[src] = []
            neighbor_weight[src] = []
            for dst_ids_ws in tmp[1:]:
                dst, w = dst_ids_ws.split(":")
                neighbor_list[src].append(int(dst)) #用int存dst float存w
                neighbor_weight[src].append(float(w))



    treePath = os.path.join(cwd, 'data/' + obj + '/data.TD_RvNN.vol_5000.txt')
    print("reading twitter tree")
    treeDic = {}
    # map_root ={}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]



        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
            # map_root[eid] = 0
        # if map_root[eid] > 100: continue
        # map_root[eid]+=1
        treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}

    print('tree no:', len(treeDic))

    labelPath = os.path.join(cwd, "data/" + obj + "/" + obj + "_label_All.txt")
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']

    print("loading tree label")
    event, y = [], []
    l1 = l2 = l3 = l4 = 0
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        label, eid = line.split('\t')[0], line.split('\t')[2]
        label = label.lower()
        event.append(eid)
        if label in labelset_nonR:
            labelDic[eid] = 0
            l1 += 1
        if label in labelset_f:
            labelDic[eid] = 1
            l2 += 1
        if label in labelset_t:
            labelDic[eid] = 2
            l3 += 1
        if label in labelset_u:
            labelDic[eid] = 3
            l4 += 1
    print(len(labelDic))
    print(l1, l2, l3, l4)
    global count
    count = []
    def loadEid(event,id,y,neighbor_lst,neighbor_weight): #add
        if event is None:
            return None
        if len(event) < 2:
            return None
        if len(event)>1:
            if neighbor_lst is None:
                neighbor_lst = []
                neighbor_weight = []
            neighbor_row = []
            neighbor_col = []
            for uid in neighbor_lst:
                neighbor_row.append(int(id))
                neighbor_col.append(int(uid))
            # neighbor_lst = [neighbor_row,neighbor_col]
            x_word, x_index, tree, rootfeat, rootindex = constructMat(event)#add
            x_x = getfeature(x_word, x_index)#然后x_word和x_index又能得到 tree_size*5000的 x_x特征
            rootfeat, tree, x_x, rootindex, y = np.array(rootfeat), np.array(tree), np.array(x_x), np.array(
                rootindex), np.array(y)
            count.append(len(x_x))
            #rootfeat 是单独的（1，5000） 子节点特征是是（x_x）（len(index),5000) rootindex 是下标,x_x里面也又root的feature
            # np.savez(os.path.join(cwd, 'data/'+obj+'graph/'+id+'.npz'), x=x_x,root=rootfeat,edgeindex=tree,rootindex=rootindex,y=y,neighbor_row = neighbor_row,
            #          neighbor_col = neighbor_col,neighbor_weight = neighbor_weight)
            #这就是格式，所以特征是点*5000 edgeindex = tree 也就是边的邻接矩阵[src,end]
            return None



    print("loading dataset", )
    # Parallel(n_jobs=30, backend='threading')(delayed(loadEid)(treeDic[eid] if eid in treeDic else None,eid,labelDic[eid]) for eid in tqdm(event))#tqdm只有进度条作用
    Parallel(n_jobs=30, backend='threading')(delayed(loadEid)(treeDic[eid] if eid in treeDic else None,eid,labelDic[eid],neighbor_list[eid] if eid in neighbor_list else None,neighbor_weight[eid] if eid in neighbor_list else None) for eid in tqdm(event))
    print(np.percentile(count,90),'np.percentile(count,90)')
    print(pd.DataFrame(count,columns=['count']).describe())


    return

if __name__ == '__main__':
    obj= sys.argv[1]
    # obj = 'Twitter16'
    main(obj)