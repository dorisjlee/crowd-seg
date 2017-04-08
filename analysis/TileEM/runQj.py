# Compute individual worker qualities based on ground truth data as T
import numpy as np
import pandas as pd
from qualityBaseline import *
import pickle as pkl
import shapely
my_BBG  = pd.read_csv("my_ground_truth.csv")
Qj={}
for object_id in tqdm(list(set(my_BBG.object_id))):
    ground_truth_match = my_BBG[my_BBG.object_id==object_id]
    x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
    T = Polygon(zip(x_locs,y_locs))
    DATA_DIR ="output"
    tiles = pkl.load(open(DATA_DIR+"/tiles{}.pkl".format(object_id)))
    indMat = pkl.load(open(DATA_DIR+"/indMat{}.pkl".format(object_id)))
    workers = pkl.load(open(DATA_DIR+"/worker{}.pkl".format(object_id)))
    for wid,j in zip(workers,range(len(workers))):
        Ncorrect=0
        Nwrong = 0
        for k in range(len(tiles)):
            tk = tiles[k]
            ljk = indMat[j][k]
            try:
                overlap = T.intersection(tk).area/T.area>0.8
                tjkInT = T.contains(tk) or overlap
            except(shapely.geos.TopologicalError):
                overlap=True
                tjkInT = T.contains(tk)

            if (ljk ==1 and tjkInT) or (ljk ==0 and (not tjkInT)):
                Ncorrect+=1
            elif (ljk ==1 and (not tjkInT)) or (ljk ==0 and tjkInT):
                Nwrong+=1
        qj = Ncorrect/float(Ncorrect+Nwrong)
        if wid in Qj:
            Qj[wid].append(qj)
        else:
            Qj[wid]=[qj]
pkl.dump(Qj,open("Qj.pkl",'w'))
