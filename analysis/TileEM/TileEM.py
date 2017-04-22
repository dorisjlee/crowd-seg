import pandas as pd
from analysis_toolbox import *
from TileEM_plot_toolbox import *
from adjacency import *
from tqdm import tqdm
import numpy as np
def estimate_Qj(T,tiles,indMat,workers,Qjfunc,A_percentile):
    Qj=[]
    for wid,j in zip(workers,range(len(workers))):
        Qj.append(Qjfunc(tiles,indMat,T,j,A_percentile))
    return Qj
def computeT(objid,T,tiles,indMat,workers,Tprime_lst,Qj,pTprimefunc,A_percentile,PLOT_LIKELIHOOD=False):
    # Loop through Tprime_lst find the argmax T' s.t pTprime is max given fixed Qj
    pTprime_lst =[]
    for Tprime_idx in Tprime_lst:
        pTprime = pTprimefunc(objid,Tprime_idx,Qj,T,tiles,indMat,workers,A_percentile)
        pTprime_lst.append(pTprime)
    Tidx= np.argmax(pTprime_lst)
    max_likelihood =pTprime_lst[Tidx]
    if PLOT_LIKELIHOOD:
        tidx_score = np.zeros(len(tiles))
        pTprime_lst= [123,412,124]
        Tprime_idx=[1,3,18]
        for i,tidx in enumerate(Tprime_idx):
            tidx_score[Tprime_idx[i]]=pTprime_lst[i]
    	visualizeTilesScore(tiles,tidx_score,INT_Z=False,colorful=True)
    return Tprime_lst[Tidx],join_tiles(Tprime_lst[Tidx],tiles)[0],max_likelihood
###################################################################################################################################################
###################################################################################################################################################
############################################################### Getting Ground Truth  #############################################################
###################################################################################################################################################
###################################################################################################################################################

def initT(tiles,indMat):
    # In the initial step, we pick T to be the top 5 area-vote score
    # where we combine the area and vote in a 1:2 ratio
    area = np.array(indMat[-1])
    votes =indMat[:-1].sum(axis=0)
    norm_area_vote = area/max(area)+2*votes/max(votes)
    tidx = np.argsort(norm_area_vote)[::-1][:5]
    return join_tiles(tidx,tiles)[0]

def ground_truth_T(object_id):
    my_BBG  = pd.read_csv("my_ground_truth.csv")
    ground_truth_match = my_BBG[my_BBG.object_id==object_id]
    x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
    T = Polygon(zip(x_locs,y_locs))
    return T
###################################################################################################################################################
###################################################################################################################################################
############################################################### T prime search strategy  ##########################################################
###################################################################################################################################################
###################################################################################################################################################
def Tprime_snowball_area(objid,indMat,fixedtopk=5, topk = 40,NTprimes=300):
    # Select weighted area-vote score top tiles
    area = np.array(indMat[-1])
    votes =indMat[:-1].sum(axis=0)
    norm_area_vote = area/max(area)+2*votes/max(votes)
    sorted_tidx = np.argsort(norm_area_vote)[::-1]
    fixed_tidx= sorted_tidx[:fixedtopk]
    tile_subset_idx =sorted_tidx[fixedtopk:topk]
    #Creating random subsets from topk tiles
    rand_subset =[]
    flexiblek=topk-fixedtopk
    for i in range(NTprimes): 
        NumTilesInCombo= np.random.randint(1,flexiblek)#at least one tile must be selected
        tidxInCombo= list(np.random.choice(tile_subset_idx,NumTilesInCombo,replace=False))
        tidxInCombo.extend(fixed_tidx)
        rand_subset.append(tidxInCombo)
    return rand_subset

def runTileEM(objid,Tprimefunc,pTprimefunc,Qjfunc,A_percentile,Niter,DEBUG=False):
    '''
    Tfunc : how to get ground truth 
    Tprimefunc : how to pick T'
    pTprimefunc : Model used for computing p(T')
    Qjfunc : Model used for estimating Qj parameters
    objid,A_percentile
    '''
    tiles = pkl.load(open(DATA_DIR+"/vtiles{}.pkl".format(objid)))
    indMat = pkl.load(open(DATA_DIR+"/indMat{}.pkl".format(objid)))
    workers = pkl.load(open(DATA_DIR+"/worker{}.pkl".format(objid)))
    likelihood_lst=[]
    T_lst = []
    if DEBUG: print "Coming up with T' combinations to search through" 
    Tprime_lst = Tprimefunc(objid,indMat,fixedtopk=5, topk = 40,NTprimes=500)
    for _i in tqdm(range(Niter)):
        if _i ==0:
            T=initT(tiles,indMat)
        if DEBUG: print "E-step : Estimate Qj parameters"
        Qjhat = estimate_Qj(T,tiles,indMat,workers,Qjfunc,A_percentile)
        ####potentially adaptive way of getting a new set of T', currently a fixed set of T' #####
        if DEBUG: print "Mstep: Picking the max-likelihood T' " 
        Tidx,T, max_likelihood= computeT(objid,T,tiles,indMat,workers,Tprime_lst,Qjhat,pTprimefunc,A_percentile)
        likelihood_lst.append(max_likelihood)
        T_lst.append(Tidx)
    return T_lst,likelihood_lst

if __name__ =="__main__":
	DATA_DIR="final_all_tiles"
	T_lst,likelihood_lst = runTileEM(2,Tprime_snowball_area,pTprimeGTLSA,QjGTLSA,99,10)