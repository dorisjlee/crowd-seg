import shapely
from analysis_toolbox import *
import pandas as pd 
import numpy as np 
import pickle as pkl
from tqdm import tqdm 
#DATA_DIR="sampletopworst5"
DATA_DIR="final_all_tiles"
###################################################################################################################################################
###################################################################################################################################################
############################################################### WOKER QUALITY CALCULATIONS ########################################################
###################################################################################################################################################
###################################################################################################################################################
def QjBasic(tiles,indMat,T,j,args):
    '''
    Compute MLE of QJ given fixed T=T'for worker j
    args is a dummy argument
    '''
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
    return qj

def QjLSA(tiles,indMat,T,j,A_percentile):
    '''
    Large Small Area (LSA) Tile EM Worker model 
    '''
    tile_area = np.array(indMat[-1])
    A_thres = np.percentile(tile_area,A_percentile)
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
    return q1,q2
def QjGTLSA(tiles,indMat,T,j,A_thres):
    '''
    GT inclusion, Large Small Area (LSA) Tile EM Worker model 
    Compute the set of Worker qualities
    A_thres: Area threshold
    Qn1,Qp1,Qn1,Qp2
    ngt : not included in ground truth 
    gt : included in ground truth 
    '''
    tile_area = np.array(indMat[-1])
    # if A_percentile!=-1: A_thres = np.percentile(tile_area,A_percentile)
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
	    try:
                tjkInT = T.contains(tk)
	    except(shapely.geos.TopologicalError):
		try:
		    tjkInT = T.buffer(1e-10).contains(tk) 
		except(shapely.geos.TopologicalError):
		    try:
		        tjkInT = T.intersection(tk.buffer(-1e-10)).area/T.area>0.8
		    except(shapely.geos.TopologicalError):
		        print "Problematic containment check"
		        #pkl.dump(tk,open("problematic_containment_{}.pkl".format(k),'w'))
			#pkl.dump(T,open("problematic_T_containment_{}.pkl".format(k),'w'))
		        tjkInT=False
		        pass 
	    except(shapely.geos.PredicateError):
	    	print "PredicateError Problematic containment check"
	    	#pkl.dump(tk,open("problematic_containment_{}.pkl".format(k),'w'))
            	#pkl.dump(T,open("problematic_T_containment_{}.pkl".format(k),'w'))
            	tjkInT=False
            	pass
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
    return qn1,qn2,qp1,qp2

def correct(ljk,tjkInT):
    if (ljk ==1 and tjkInT) or (ljk ==0 and (not tjkInT)):
        return 1
    elif (ljk ==1 and (not tjkInT)) or (ljk ==0 and tjkInT):
        return 0

def QjArea(tiles,indMat,T,j,args):
    '''
    Area weighted worker quality scoring function
    args is a dummy argument
    '''    
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
    return qj
###################################################################################################################################################
###################################################################################################################################################
############################################################### p(T') LIKELIHOOD CALCULATIONS #####################################################
###################################################################################################################################################
###################################################################################################################################################

def pTprimeBasic(objid,Tprime,Qj,tiles,indMat,workers,args):
    '''
    Basic Tile EM Worker model 
    Given a tile combination Tprime, compute likelihood of that T'=T
    args is a dummy argument
    '''
    plk=0 
    for k,tk in enumerate(tiles): 
        for j in range(len(workers)):
            tk = tiles[k]
            ljk = indMat[j][k]
            wid=workers[j]
            tjkInT = Tprime.contains(tk) #overlap > threshold
            qj = Qj[j]
            if (ljk ==1 and tjkInT) or (ljk ==0 and (not tjkInT)):
                plk+=np.log(qj)
            elif (ljk ==1 and (not tjkInT)) or (ljk ==0 and tjkInT):
                plk+=np.log(1-qj)
    return plk

def pTprimeLSA(objid,Tprime,Qj,tiles,indMat,workers,A_percentile):
    '''
    Area Based Tile EM Worker model 
    Given a tile combination Tprime, compute likelihood of that T'=T
    '''
    Q1,Q2=zip(*Qj)
    plk=0

    for k,tk in enumerate(tiles): 
        for j in range(len(workers)):
            ljk = indicatorMat[j][k]
            tjkInT = Tprime.contains(tk) 
            wid=workers[j]
            qj1 = Q1[j]
            qj2 = Q2[j]
            if (ljk ==1 and tjkInT) or (ljk ==0 and (not tjkInT)):
                if tk.area>A_thres:
                    plk+=np.log(qj1)
                else: 
                    plk+=np.log(qj2)
            elif (ljk ==1 and (not tjkInT)) or (ljk ==0 and tjkInT):
                if tk.area>A_thres:
                    plk+=np.log(1-qj1)
                else: 
                    plk+=np.log(1-qj2)
    return plk

def pTprimeGTLSA(objid,Tprime,Qj,tiles,indMat,workers,A_percentile):
    '''
    Area Based Tile EM Worker model 
    Given a tile combination Tprime, compute likelihood of that T'=T
    '''
    Qn1,Qn2,Qp1,Qp2 = zip(*Qj)
    tile_area = np.array(indMat[-1])
    A_thres = np.percentile(tile_area,A_percentile)
    plk=0
    
    for k,tk in enumerate(tiles): 
        for j in range(len(workers)):
            ljk = indMat[j][k]
            try: 
                tjkInT = Tprime.contains(tk) 
            except(shapely.geos.TopologicalError):
                pkl.dump(tk,open("problematic_tk_{}.pkl".format(k),'w'))
                pkl.dump(Tprime,open("problematic_Tprime_{}.pkl".format(k),'w'))
                tjkInT = False
                print "Topological Error Exception, coerce tjkInT to be False" 
            wid=workers[j]
            qp1 = Qp1[j]
            qp2 = Qp2[j]
            qn1 = Qn1[j]
            qn2 = Qn2[j]
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

def AreaTprimeScore(objid,Tprime,Qj,T,tiles,indMat,workers,args):
    '''
    Area-Weighted Tile EM Worker model 
    Given a tile combination Tprime, compute area-weighted score for that T'=T
    args is a dummy argument
    '''
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
