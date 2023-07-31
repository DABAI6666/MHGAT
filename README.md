Paper：
# A Heterogeneous Propagation Graph Model for Rumor Detection Under the Relationship Among Multiple Propagation Subtrees
Guoyi Li, Jingyuan Hu, Yulei Wu, Xiaodan Zhang, Wei Zhou & Honglei Lyu 

# Datasets:  
The datasets used in the experiments were based on the three publicly available Weibo and Twitter datasets released by Ma et al. (2016) and Ma et al. (2017):

Jing Ma, Wei Gao, Prasenjit Mitra, Sejeong Kwon, Bernard J Jansen, Kam-Fai Wong, and Meeyoung Cha. Detecting rumors from microblogs with recurrent neural networks. In Proceedings of IJCAI 2016.

Jing Ma, Wei Gao, Kam-Fai Wong. Detect Rumors in Microblog Posts Using Propagation Structure via Kernel Learning. ACL 2017.

In the 'data' folder we provide the pre-processed data files used for our experiments. The raw datasets can be respectively downloaded from https://www.dropbox.com/s/46r50ctrfa0ur1o/rumdect.zip?dl=0. and https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0.

The Weibo datafile 'weibotree.txt' is in a tab-sepreted column format, where each row corresponds to a weibo. Consecutive columns correspond to the following pieces of information:  
1: root-id -- an unique identifier describing the tree (weiboid of the root);  
2: index-of-parent-weibo -- an index number of the parent weibo for the current weibo;  
3: index-of-the-current-weibo -- an index number of the current weibo;  
4: list-of-index-and-counts -- the rest of the line contains space separated index-count pairs, where a index-count pair is in format of "index:count", E.g., "index1:count1 index2:count2" (extracted from the "text" field in the json format from Weibo raw datasets)  

For a detailed description of Twitter datafile 'data.TD_RvNN.vol_5000.txt' can be seen at [RvNN](https://github.com/majingCUHK/Rumor_RvNN).

# Dependencies:  
python==3.5.2  
numpy==1.18.1  
torch==1.4.0  
torch_scatter==1.4.0  
torch_sparse==0.4.3  
torch_cluster==1.4.5  
torch_geometric==1.3.2  
tqdm==4.40.0  
joblib==0.14.1  


## Usage
Make sure that cuda/bin, cuda/include and cuda/lib64 are in your $PATH, $CPATH and $LD_LIBRARY_PATH respectively before the installation, e.g.:
```
$ echo $PATH
>>> /usr/local/cuda/bin:...

$ echo $CPATH
>>> /usr/local/cuda/include:...
```
and
```
$ echo $LD_LIBRARY_PATH
>>> /usr/local/cuda/lib64
```
on Linux or
```
$ echo $DYLD_LIBRARY_PATH
>>> /usr/local/cuda/lib
```
on macOS. 

* Download Twitter15, Twitter16, Weibo datasets from [BiGCN](https://github.com/TianBian95/BiGCN) and place them in `/data/*`.


# Reproduce the experimental results:  
Run script 
```
$ sh main.sh
```

#Generate graph data and store in /data/Weibograph
```
python ./Process/getWeibograph.py
```
#Generate graph data and store in /data/Twitter15graph
```
python ./Process/getTwittergraph.py Twitter15
```
#Generate graph data and store in /data/Twitter16graph
```
python ./Process/getTwittergraph.py Twitter16
```
#Reproduce the experimental results
```.
python ./model/Weibo/MHGAT_Weibo.py
```
```
python ./model/Twitter/MHGAT_Twitter_15.py
```
```
python ./model/Twitter/MHGAT_Twitter_16.py
```


