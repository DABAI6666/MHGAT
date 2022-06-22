#! /bin/bash
unzip -d ./data/Weibo ./data/Weibo/weibotree.txt.zip
pip install -U torch==1.4.0 numpy==1.18.1
pip install -r requirements.txt
#Generate graph data and store in /data/Weibograph
python ./Process/getWeibograph.py
#Generate graph data and store in /data/Twitter15graph
python ./Process/getTwittergraph.py Twitter15
#Generate graph data and store in /data/Twitter16graph
python ./Process/getTwittergraph.py Twitter16
#Reproduce the experimental results.
python ./model/Weibo/MHGAT_Weibo.py
python ./model/Twitter/MHGAT_Twitter_15.py
python ./model/Twitter/MHGAT_Twitter_16.py

