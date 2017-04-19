import shapely
from analysis_toolbox import *
import pandas as pd 
import numpy as np 
import pickle as pkl
from tqdm import tqdm 
#DATA_DIR="sampletopworst5"
DATA_DIR="final_all_tiles"
def QjBasic(SAVE=False):
    '''
    Basic Tile EM Worker model 
    Compute the set of Worker qualities
    '''
    my_BBG  = pd.read_csv("my_ground_truth.csv")
    Qj=[]
    os.chdir(DATA_DIR)
    for object_id in tqdm(list(set(my_BBG.object_id))):
        ground_truth_match = my_BBG[my_BBG.object_id==object_id]
        x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
        T = Polygon(zip(x_locs,y_locs))
        tiles = pkl.load(open("vtiles{}.pkl".format(object_id)))
        indMat = pkl.load(open("indMat{}.pkl".format(object_id)))
        workers = pkl.load(open("worker{}.pkl".format(object_id)))
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
            Qj.append([object_id,wid,qj])
    Qj_tbl = pd.DataFrame(Qj,columns=["object_id","worker_id","Qj"])
    if SAVE: pkl.dump(Qj_tbl,open("Qj.pkl",'w'))
    os.chdir("..")
    return Qj
def QjLSA(A_thres,SAVE=False):
    '''
    Large Small Area (LSA) Tile EM Worker model 
    Compute the set of Worker qualities
    A_thres: Area threshold
    '''
    my_BBG  = pd.read_csv("my_ground_truth.csv")
    os.chdir(DATA_DIR)
    Qj1=[]
    Qj2=[]
    for object_id in tqdm(list(set(my_BBG.object_id))):
        ground_truth_match = my_BBG[my_BBG.object_id==object_id]
        x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
        T = Polygon(zip(x_locs,y_locs))
        tiles = pkl.load(open("vtiles{}.pkl".format(object_id)))
        indMat = pkl.load(open("indMat{}.pkl".format(object_id)))
        workers = pkl.load(open("worker{}.pkl".format(object_id)))
        for wid,j in zip(workers,range(len(workers))):
            large_Ncorrect=0
            large_Nwrong = 0
            small_Ncorrect=0
            small_Nwrong = 0
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
                    if tk.area>A_thres:
                        large_Ncorrect+=1
                    else: 
                        small_Ncorrect+=1
                elif (ljk ==1 and (not tjkInT)) or (ljk ==0 and tjkInT):
                    if tk.area>A_thres:
                        large_Nwrong+=1
                    else: 
                        small_Nwrong+=1
            try:
                q1 = large_Ncorrect/float(large_Ncorrect+large_Nwrong)
            except(ZeroDivisionError):
                q1 = -1
            try:
                q2 = small_Ncorrect/float(small_Ncorrect+small_Nwrong)
            except(ZeroDivisionError):
                q2 = -1

            Qj1.append([object_id,wid,q1])
            Qj2.append([object_id,wid,q2])
    Qj1_tbl = pd.DataFrame(Qj1,columns=["object_id","worker_id","Q1"])
    Qj2_tbl = pd.DataFrame(Qj2,columns=["object_id","worker_id","Q2"])
    Qj = Qj1_tbl.merge(Qj2_tbl)
    if SAVE:
        pkl.dump(Qj,open("Qj12_A>{}.pkl".format(A_thres),'w'))
    os.chdir("..")
    return Qj
def pTprimeBasic(objid,Tprime,T):
    '''
    Basic Tile EM Worker model 
    Given a tile combination Tprime, compute likelihood of that T'=T
    '''
    Qj=pkl.load(open("Qj.pkl",'r'))
    tiles = pkl.load(open("vtiles{}.pkl".format(objid)))
    workers = pkl.load(open("worker{}.pkl".format(objid)))
    indicatorMat= pkl.load(open("indMat{}.pkl".format(objid)))
    plk=[]
    for k in Tprime: 
        for j in range(len(workers)):
            tk = tiles[k]
            ljk = indicatorMat[j][k]
            wid=workers[j]
            tjkInT = T.contains(tk) #overlap > threshold
            qj = float(Qj[(Qj["object_id"]==objid)&(Qj["worker_id"]==wid)]["Qj"])
            if (ljk ==1 and tjkInT) or (ljk ==0 and (not tjkInT)):
                plk.append(qj)
            elif (ljk ==1 and (not tjkInT)) or (ljk ==0 and tjkInT):
                plk.append(1-qj)
    pTprime=np.product(plk)
    return pTprime
def pTprimeLSA(objid,Tprime,T,A_thres):
    '''
    Area Based Tile EM Worker model 
    Given a tile combination Tprime, compute likelihood of that T'=T
    '''
    Qj=pkl.load(open("Qj12_A>{}.pkl".format(A_thres),'r'))
    tiles = pkl.load(open("vtiles{}.pkl".format(objid)))
    workers = pkl.load(open("worker{}.pkl".format(objid)))
    indicatorMat= pkl.load(open("indMat{}.pkl".format(objid)))
    plk=[]
    for k in Tprime: 
        for j in range(len(workers)):
            tk = tiles[k]
            ljk = indicatorMat[j][k]
            tjkInT = T.contains(tk) 
            wid=workers[j]
            qj1 = float(Qj[(Qj["object_id"]==objid)&(Qj["worker_id"]==wid)]["Q1"])
            qj2 = float(Qj[(Qj["object_id"]==objid)&(Qj["worker_id"]==wid)]["Q2"])
            if (ljk ==1 and tjkInT) or (ljk ==0 and (not tjkInT)):
                if tk.area>A_thres:
                    plk.append(qj1)
                else: 
                    plk.append(qj2)
            elif (ljk ==1 and (not tjkInT)) or (ljk ==0 and tjkInT):
                if tk.area>A_thres:
                    plk.append(1-qj1)
                else: 
                    plk.append(1-qj2)
    pTprime=np.product(plk)
    return pTprime