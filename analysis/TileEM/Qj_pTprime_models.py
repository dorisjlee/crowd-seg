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
    return Qj_tbl
def correct(ljk,tjkInT):
    if (ljk ==1 and tjkInT) or (ljk ==0 and (not tjkInT)):
        return 1
    elif (ljk ==1 and (not tjkInT)) or (ljk ==0 and tjkInT):
        return 0
def QjArea(SAVE=False):
    '''
    Area weighted worker quality scoring function
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
            numerator=0
            denominator= 0
            for k in range(len(tiles)): 
                tk = tiles[k]
                ljk = indMat[j][k]
                try:
                    overlap = T.intersection(tk).area/T.area>0.8
                    tjkInT = T.contains(tk) or overlap
                except(shapely.geos.TopologicalError):
                    overlap=True
                    tjkInT = T.contains(tk)
                numerator+=tk.area*correct(ljk,tjkInT)
                denominator+=tk.area
            qj =numerator/float(denominator)
            Qj.append([object_id,wid,qj])
    Qj_tbl = pd.DataFrame(Qj,columns=["object_id","worker_id","Qj_area"])
    if SAVE: pkl.dump(Qj_tbl,open("Qj_area.pkl",'w'))
    os.chdir("..")
    return Qj_tbl
def QjLSA(A_percentile,SAVE=False):
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
        tile_area = np.array(indMat[-1])
        A_thres = np.percentile(tile_area,A_percentile)
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
        pkl.dump(Qj,open("Qj12_A>{}%.pkl".format(A_percentile),'w'))
    os.chdir("..")
    return Qj
def QjGTLSA(A_percentile,SAVE=False):
    '''
    GT inclusion, Large Small Area (LSA) Tile EM Worker model 
    Compute the set of Worker qualities
    A_thres: Area threshold
    Qn1,Qp1,Qn1,Qp2
    ngt : not included in ground truth 
    gt : included in ground truth 
    '''
    my_BBG  = pd.read_csv("my_ground_truth.csv")
    os.chdir(DATA_DIR)
    Qp1=[]
    Qp2=[]
    Qn1=[]
    Qn2=[]
    for object_id in tqdm(list(set(my_BBG.object_id))):
        ground_truth_match = my_BBG[my_BBG.object_id==object_id]
        x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
        T = Polygon(zip(x_locs,y_locs))
        tiles = pkl.load(open("vtiles{}.pkl".format(object_id)))
        indMat = pkl.load(open("indMat{}.pkl".format(object_id)))
        workers = pkl.load(open("worker{}.pkl".format(object_id)))
        tile_area = np.array(indMat[-1])
        A_thres = np.percentile(tile_area,A_percentile)
        for wid,j in zip(workers,range(len(workers))):
            large_gt_Ncorrect=0
            large_gt_Nwrong = 0
            small_gt_Ncorrect=0
            small_gt_Nwrong = 0
            large_ngt_Ncorrect=0
            large_ngt_Nwrong = 0
            small_ngt_Ncorrect=0
            small_ngt_Nwrong = 0
            for k in range(len(tiles)): 
                tk = tiles[k]
                ljk = indMat[j][k]
                try:
                    overlap = T.intersection(tk).area/T.area>0.8
                    tjkInT = T.contains(tk) or overlap
                except(shapely.geos.TopologicalError):
                    overlap=True
                    tjkInT = T.contains(tk)
                if tk.area>A_thres:
                    if (ljk ==1 and tjkInT):
                        large_gt_Ncorrect+=1
                    elif (ljk ==0 and tjkInT):
                        large_gt_Nwrong+=1
                    elif (ljk ==0 and (not tjkInT)):
                        large_ngt_Ncorrect+=1
                    elif (ljk ==1 and (not tjkInT)):
                        large_ngt_Nwrong+=1
                else:
                    if (ljk ==1 and tjkInT):
                        small_gt_Ncorrect+=1
                    elif (ljk ==0 and tjkInT):
                        small_gt_Nwrong+=1
                    elif (ljk ==0 and (not tjkInT)):
                        small_ngt_Ncorrect+=1
                    elif (ljk ==1 and (not tjkInT)):
                        small_ngt_Nwrong+=1
            try:
                qp1 = large_gt_Ncorrect/float(large_gt_Ncorrect+large_gt_Nwrong)
            except(ZeroDivisionError):
                qp1 = -1
            try:
                qn1 = large_ngt_Ncorrect/float(large_ngt_Ncorrect+large_ngt_Nwrong)
            except(ZeroDivisionError):
                qn1 = -1
            try:
                qp2 = small_gt_Ncorrect/float(small_gt_Ncorrect+small_gt_Nwrong)
            except(ZeroDivisionError):
                qp2 = -1
            try:
                qn2 = small_ngt_Ncorrect/float(small_ngt_Ncorrect+small_ngt_Nwrong)
            except(ZeroDivisionError):
                qn2 = -1

            Qp1.append([object_id,wid,qp1])
            Qp2.append([object_id,wid,qp2])
            Qn1.append([object_id,wid,qn1])
            Qn2.append([object_id,wid,qn2])
    Qp1_tbl = pd.DataFrame(Qp1,columns=["object_id","worker_id","Qp1"])
    Qp2_tbl = pd.DataFrame(Qp2,columns=["object_id","worker_id","Qp2"])
    Qn1_tbl = pd.DataFrame(Qn1,columns=["object_id","worker_id","Qn1"])
    Qn2_tbl = pd.DataFrame(Qn2,columns=["object_id","worker_id","Qn2"])
    Qp = Qp1_tbl.merge(Qp2_tbl)
    Qn = Qn1_tbl.merge(Qn2_tbl)
    Qj = Qp.merge(Qn)
    if SAVE:
        pkl.dump(Qj,open("Qgt12_A>{}%.pkl".format(A_percentile),'w'))
    os.chdir("..")
    return Qj
def pTprimeGTLSA(objid,Tprime,T,A_percentile):
    '''
    Area Based Tile EM Worker model 
    Given a tile combination Tprime, compute likelihood of that T'=T
    '''
    Qj=pkl.load(open("Qgt12_A>{}%.pkl".format(A_percentile),'r'))
    Qj_obj = Qj[(Qj["object_id"]==objid)]
    tiles = pkl.load(open("vtiles{}.pkl".format(objid)))
    workers = pkl.load(open("worker{}.pkl".format(objid)))
    indMat = pkl.load(open("indMat{}.pkl".format(objid)))
    tile_area = np.array(indMat[-1])
    A_thres = np.percentile(tile_area,A_percentile)
    plk=1    
    for k in Tprime: 
        for j in range(len(workers)):
            tk = tiles[k]
            ljk = indMat[j][k]
            tjkInT = T.contains(tk) 
            wid=workers[j]
            qp1 = float(Qj_obj[Qj_obj["worker_id"]==wid]["Qp1"])
            qp2 = float(Qj_obj[Qj_obj["worker_id"]==wid]["Qp2"])
            qn1 = float(Qj_obj[Qj_obj["worker_id"]==wid]["Qn1"])
            qn2 = float(Qj_obj[Qj_obj["worker_id"]==wid]["Qn2"])
            if tk.area>A_thres:
                if ljk ==1:
                    if tjkInT:
                        plk+=np.log(qp1)
                    else:
                        plk+=np.log(1-qn1)
                else:
                    if tjkInT:
                        plk+=np.log(1-qp1)
                    elif not tjkInT:
                        plk+=np.log(qn1)
            else:
                if ljk ==1:
                    if tjkInT:
                        plk+=np.log(qp2)
                    else:
                        plk+=np.log(1-qn2)    
                else:
                    if tjkInT:
                        plk+=np.log(1-qp2)
                    else:
                        plk+=np.log(qn2)
    return plk

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
def AreaTprimeScore(objid,Tprime,T):
    '''
    Area-Weighted Tile EM Worker model 
    Given a tile combination Tprime, compute area-weighted score for that T'=T
    '''
    Qj=pkl.load(open("Qj_area.pkl",'r'))
    tiles = pkl.load(open("vtiles{}.pkl".format(objid)))
    workers = pkl.load(open("worker{}.pkl".format(objid)))
    indicatorMat= pkl.load(open("indMat{}.pkl".format(objid)))
    TprimeScore=0
    for k in Tprime: 
        for j in range(len(workers)):
            tk = tiles[k]
            ljk = indicatorMat[j][k]
            wid=workers[j]
            tjkInT = T.contains(tk) #overlap > threshold
            qj = float(Qj[(Qj["object_id"]==objid)&(Qj["worker_id"]==wid)]["Qj_area"])
            if tjkInT: 
                if ljk ==1:
                    TprimeScore+=qj
                    #print +qj
                else: 
                    TprimeScore-=qj
                    #print -qj
            else: 
                if ljk==1:
                    TprimeScore-=qj
                    #print -qj
                else: 
                    TprimeScore+=qj
                    #print +qj
        #print "score(T'):",TprimeScore
    return TprimeScore