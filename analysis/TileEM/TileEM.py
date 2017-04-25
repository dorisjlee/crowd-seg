import pandas as pd
from Qj_pTprime_models import *
from analysis_toolbox import *
from TileEM_plot_toolbox import *
from adjacency import *
from tqdm import tqdm
import numpy as np
def estimate_Qj(T,tiles,indMat,workers,Qjfunc,A_percentile,DEBUG=False):
    Qj=[]
    for wid,j in zip(workers,range(len(workers))):
        Qj.append(Qjfunc(tiles,indMat,T,j,A_percentile))
    if DEBUG: print "Qj: ",Qj
    return Qj
def computeT(objid,tiles,indMat,workers,Tprime_lst, Tprime_idx_lst,Qj,pTprimefunc,A_percentile,DEBUG=False,PLOT_LIKELIHOOD=False):
    # Loop through Tprime_lst find the argmax T' s.t pTprime is max given fixed Qj
    pTprime_lst =[]
    if DEBUG: print "Looping through T' "
    for Tprime_idx in tqdm(Tprime_lst):
        pTprime = pTprimefunc(objid,Tprime_idx,Qj,tiles,indMat,workers,A_percentile)
        pTprime_lst.append(pTprime)
    Tidx= np.argmax(pTprime_lst)
    max_likelihood =pTprime_lst[Tidx]
    if PLOT_LIKELIHOOD:
    	visualizeTilesScore(tiles,dict(zip(range(len(Tprime_lst)),pTprime_lst)),INT_Z=False,colorful=True)
    if DEBUG: print "Likelihood:{0} ; T={1}".format(max_likelihood,Tprime_idx_lst[Tidx])
    return pTprime_lst,Tprime_idx_lst[Tidx],Tprime_lst[Tidx],max_likelihood
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
    norm_area_vote = area/max(area)+5*votes/max(votes)
    tidx = np.argsort(norm_area_vote)[::-1][:2]
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
def Tprime_snowball_area(objid,tiles,indMat,fixedtopk=3, topk = 40,NTprimes=300):
    # Select weighted area-vote score top tiles
    np.random.seed(111)
    area = np.array(indMat[-1])
    votes =indMat[:-1].sum(axis=0)
    norm_area_vote = area/max(area)+2*votes/max(votes)
    sorted_tidx = np.argsort(norm_area_vote)[::-1]
    fixed_tidx= sorted_tidx[:fixedtopk]
    tile_subset_idx =sorted_tidx[fixedtopk:topk]
    #Creating random subsets from topk tiles
    Tprime_idx_lst =[]
    Tprime_lst =[]
    flexiblek=topk-fixedtopk
    for i in range(NTprimes):
        NumTilesInCombo= np.random.randint(1,flexiblek)#at least one tile must be selected
        tidxInCombo= list(np.random.choice(tile_subset_idx,NumTilesInCombo,replace=False))
        tidxInCombo.extend(fixed_tidx)
        Tprime_lst.append(join_tiles(tidxInCombo,tiles)[0].buffer(0))
        Tprime_idx_lst.append(tidxInCombo)
    return Tprime_lst, Tprime_idx_lst
# def runTileEMGreedy(objid,Tprimefunc,pTprimefunc,Qjfunc,A_percentile,Niter,NTprimes=100,DEBUG=False,PLOT_LIKELIHOOD=False):
#     '''
#     Tfunc : how to get ground truth
#     Tprimefunc : how to pick T'
#     pTprimefunc : Model used for computing p(T')
#     Qjfunc : Model used for estimating Qj parameters
#     objid,A_percentile
#     '''
#     tiles = pkl.load(open(DATA_DIR+"/vtiles{}.pkl".format(objid)))
#     indMat = pkl.load(open(DATA_DIR+"/indMat{}.pkl".format(objid)))
#     workers = pkl.load(open(DATA_DIR+"/worker{}.pkl".format(objid)))
#     T_lst = []
#     pTprime_lst=[]
#     Qj_lst=[]
#     #if DEBUG: print "Coming up with T' combinations to search through"
#     #Tprime_lst, Tprime_idx_lst = Tprimefunc(objid,tiles,indMat,fixedtopk=3, topk = 40,NTprimes=NTprimes)
#     Tstar=[]
#     for _i in tqdm(range(Niter)):
#         if _i ==0:
#             # if DEBUG: print "Initializing tiles "
#             # T=initT(tiles,indMat)
#             if DEBUG: print "Initializing Qjs"
#             qinit = list(np.ones(len(workers))*0.5)
#             Qjhat = np.array([qinit,qinit,qinit,qinit]).T

#         #if DEBUG: print "Mstep: Picking the max-likelihood T' "
#         #pTprimes,Tidx,T, max_likelihood= computeT(objid,tiles,indMat,workers,Tprime_lst, Tprime_idx_lst,Qjhat,pTprimefunc,A_percentile,PLOT_LIKELIHOOD=PLOT_LIKELIHOOD,DEBUG=DEBUG)
#         ############################################################################################################
#         Qn1,Qn2,Qp1,Qp2 = zip(*Qjhat)
#         for tk in tiles:
#             pInT = 0
#             pNotInT = 0
#             for j in range(len(workers)):
#                 ljk = indMat[j][k]
                
#                 wid=workers[j]
#                 qp1 = Qp1[j]
#                 qp2 = Qp2[j]
#                 qn1 = Qn1[j]
#                 qn2 = Qn2[j]
#                 if tk.area>A_thres:
#                     if ljk ==1:
#                         if tjkInT:
#                             pInT+=np.log(qp1)
#                         else:
#                             pNotInT+=np.log(1-qn1)
#                     else:
#                         if tjkInT:
#                             pInT+=np.log(1-qp1)
#                         elif not tjkInT:
#                             pNotInT+=np.log(qn1)
#                 else:
#                     if ljk ==1:
#                         if tjkInT:
#                             pInT+=np.log(qp2)
#                         else:
#                             pNotInT+=np.log(1-qn2)    
#                     else:
#                         if tjkInT:
#                             pInT+=np.log(1-qp2)
#                         else:
#                             pNotInT+=np.log(qn2)
#             if pInT >pNotInT:
#                 # add to Tstar
#                 Tstar = Tstar.union(tk)
#         ############################################################################################################


#         if DEBUG: print "E-step : Estimate Qj parameters"
#         Qjhat = estimate_Qj(T,tiles,indMat,workers,Qjfunc,A_percentile,DEBUG=DEBUG)
#         pTprime_lst.append(pTprimes)
#         T_lst.append(Tidx)
#         Qj_lst.append(Qjhat)
#     return Tprime_idx_lst ,pTprime_lst,Qj_lst,T_lst
def runTileEM(objid,Tprimefunc,pTprimefunc,Qjfunc,A_percentile,Niter,NTprimes=100,DEBUG=False,PLOT_LIKELIHOOD=False):
    '''
    # Doing the M step first 

    Tfunc : how to get ground truth
    Tprimefunc : how to pick T'
    pTprimefunc : Model used for computing p(T')
    Qjfunc : Model used for estimating Qj parameters
    objid,A_percentile
    '''
    tiles = pkl.load(open(DATA_DIR+"/vtiles{}.pkl".format(objid)))
    indMat = pkl.load(open(DATA_DIR+"/indMat{}.pkl".format(objid)))
    workers = pkl.load(open(DATA_DIR+"/worker{}.pkl".format(objid)))
    T_lst = []
    pTprime_lst=[]
    Qj_lst=[]
    if DEBUG: print "Coming up with T' combinations to search through"
    Tprime_lst, Tprime_idx_lst = Tprimefunc(objid,tiles,indMat,fixedtopk=3, topk = 40,NTprimes=NTprimes)
    
    for _i in tqdm(range(Niter)):
        if _i ==0:
            # if DEBUG: print "Initializing tiles "
            # T=initT(tiles,indMat)
            if DEBUG: print "Initializing Qjs"
            qinit = list(np.ones(len(workers))*0.5)
            Qjhat = np.array([qinit,qinit,qinit,qinit]).T

        if DEBUG: print "Mstep: Picking the max-likelihood T' "
        pTprimes,Tidx,T, max_likelihood= computeT(objid,tiles,indMat,workers,Tprime_lst, Tprime_idx_lst,Qjhat,pTprimefunc,A_percentile,PLOT_LIKELIHOOD=PLOT_LIKELIHOOD,DEBUG=DEBUG)
        if DEBUG: print "E-step : Estimate Qj parameters"
        Qjhat = estimate_Qj(T,tiles,indMat,workers,Qjfunc,A_percentile,DEBUG=DEBUG)
        pTprime_lst.append(pTprimes)
        T_lst.append(Tidx)
        Qj_lst.append(Qjhat)
    return Tprime_idx_lst ,pTprime_lst,Qj_lst,T_lst

def runTileEM2(objid,Tprimefunc,pTprimefunc,Qjfunc,A_percentile,Niter,NTprimes=100,DEBUG=False,PLOT_LIKELIHOOD=False):
    '''
    # Doing the E step first 

    Tfunc : how to get ground truth
    Tprimefunc : how to pick T'
    pTprimefunc : Model used for computing p(T')
    Qjfunc : Model used for estimating Qj parameters
    objid,A_percentile
    '''
    tiles = pkl.load(open(DATA_DIR+"/vtiles{}.pkl".format(objid)))
    indMat = pkl.load(open(DATA_DIR+"/indMat{}.pkl".format(objid)))
    workers = pkl.load(open(DATA_DIR+"/worker{}.pkl".format(objid)))
    T_lst = []
    pTprime_lst=[]
    Qj_lst=[]
    if DEBUG: print "Coming up with T' combinations to search through"
    Tprime_lst, Tprime_idx_lst = Tprimefunc(objid,tiles,indMat,fixedtopk=3, topk = 40,NTprimes=NTprimes)
    
    for _i in tqdm(range(Niter)):
        if _i ==0:
            if DEBUG: print "Initializing tiles "
            T=initT(tiles,indMat)
        if DEBUG: print "E-step : Estimate Qj parameters"
        Qjhat = estimate_Qj(T,tiles,indMat,workers,Qjfunc,A_percentile,DEBUG=DEBUG)
        if DEBUG: print "Mstep: Picking the max-likelihood T' "
        pTprimes,Tidx,T, max_likelihood = computeT(objid,tiles,indMat,workers,Tprime_lst, Tprime_idx_lst,Qjhat,pTprimefunc,A_percentile,PLOT_LIKELIHOOD=PLOT_LIKELIHOOD,DEBUG=DEBUG)
        pTprime_lst.append(pTprimes)
        T_lst.append(Tidx)
        Qj_lst.append(Qjhat)
    return Tprime_idx_lst ,pTprime_lst,Qj_lst,T_lst
if __name__ =="__main__":
    #DATA_DIR="final_all_tiles"
    import time
    #Experiments 
    #exp_num=9
    # T initialization with M start
    #objid=3
    #init = time.time()
    #Tprime_lst,pTprime_lst,Qj_lst,T_lst = runTileEM(objid,Tprime_snowball_area,pTprimeGTLSA,QjGTLSA,A_percentile=90,\
    #                                    Niter=5,NTprimes=300,PLOT_LIKELIHOOD=False,DEBUG=True)
    #pkl.dump(Tprime_lst,open("Tprime_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    #pkl.dump(pTprime_lst,open("pTprime_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    #pkl.dump(T_lst,open("T_lst_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    #pkl.dump(Qj_lst,open("Qj_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    #end = time.time()
    #print "Time Elapsed: ",end-init
    # T initialization with E start
    #exp_num=10
    #objid=3
    #Tprime_lst,pTprime_lst,Qj_lst,T_lst = runTileEM2(objid,Tprime_snowball_area,pTprimeGTLSA,QjGTLSA,A_percentile=90,\
    #                                    Niter=5,NTprimes=2000,PLOT_LIKELIHOOD=False,DEBUG=True)
    #pkl.dump(Tprime_lst,open("Tprime_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    #pkl.dump(pTprime_lst,open("pTprime_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    #pkl.dump(T_lst,open("T_lst_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    #pkl.dump(Qj_lst,open("Qj_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    #end2 = time.time()
    #print "Time Elapsed: ",end2-end
    DATA_DIR="output_26"
    exp_num=11
    objid=9
    Tprime_lst,pTprime_lst,Qj_lst,T_lst = runTileEM2(objid,Tprime_snowball_area,pTprimeGTLSA,QjGTLSA,A_percentile=90,\
                                        Niter=5,NTprimes=2000,PLOT_LIKELIHOOD=False,DEBUG=True)
    pkl.dump(Tprime_lst,open("Tprime_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    pkl.dump(pTprime_lst,open("pTprime_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    pkl.dump(T_lst,open("T_lst_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    pkl.dump(Qj_lst,open("Qj_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    end2 = time.time()
    print "Time Elapsed: ",end2-end
