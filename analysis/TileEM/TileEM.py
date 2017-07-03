import pandas as pd
from Qj_pTprime_models import *
from analysis_toolbox import *
from TileEM_plot_toolbox import *
from adjacency import *
from TileEM import *
from tqdm import tqdm
import numpy as np
import pickle as pkl

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
def core(tiles,indMat,topk=1):
    # In the initial step, we pick T to be the top 5 area-vote score
    # where we combine the area and vote in a 1:5 ratio
    area = np.array(indMat[-1])
    votes =indMat[:-1].sum(axis=0)
    norm_area_vote = area/max(area)+5*votes/max(votes)
    tidx = np.argsort(norm_area_vote)[::-1][:topk]
    return join_tiles(tidx,tiles)[0],list(tidx)
def initT(tiles,indMat):
    # In the initial step,
    # pick the tiles that at least 50% of the workers have voted on
    votes =indMat[:-1].sum(axis=0)
    Nworkers = np.shape(indMat)[0]
    tidx =np.where(votes>Nworkers/2.)[0]
    if len(tidx)==0:
        #If no tiles satisfy 50% votes, then just pick the top-1 tile
	topk=1 
        tidx = np.argsort(votes)[::-1][:topk]
    return join_tiles(tidx,tiles)[0],list(tidx)
def ground_truth_T(object_id,reverse_xy = False):
    my_BBG  = pd.read_csv("my_ground_truth.csv")
    ground_truth_match = my_BBG[my_BBG.object_id==object_id]
    if reverse_xy:
	x_locs,y_locs =  process_raw_locs([ground_truth_match["y_locs"].iloc[0],ground_truth_match["x_locs"].iloc[0]])
    else:
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
def QjGTLSA(tiles,indMat,pInT_lst,pNotInT_lst,j,A_thres):
    '''
    GT inclusion, Large Small Area (LSA) Tile EM Worker model 
    Compute the set of Worker qualities
    A_thres: Area threshold
    Qn1,Qp1,Qn1,Qp2
    ngt : not included in ground truth 
    gt : included in ground truth 
    '''
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
        tjkInT = pInT_lst[k]>=pNotInT_lst[k]

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

def estimate_Qj(tiles,indMat,workers,Qjfunc,pInT_lst,pNotInT_lst,DEBUG=False):
    Qj=[]
    # Compute area threshold 
    high_confidence_tiles_area = []
    for k in range(len(tiles)): 
        if pInT_lst[k]>=pNotInT_lst[k]:
            high_confidence_tiles_area.append(tiles[k].area)
    A_thres = np.median(high_confidence_tiles_area)

    for wid,j in zip(workers,range(len(workers))):
        Qj.append(Qjfunc(tiles,indMat,pInT_lst,pNotInT_lst,j,A_thres))
    if DEBUG: print "Qj: ",Qj
    return A_thres,Qj

def runTileAdjacentMLConstruction(objid,workerErrorfunc,Qjfunc,A_percentile,Niter=10,DEBUG=False,PLOT_LIKELIHOOD=False,PLOT=False):
    '''
    Initilaize with majority vote tile , get good Qj estimates based on that
    Input:
    _____
    Tfunc : how to get ground truth
    workerErrorfunc : worker error model for computing p(ljk)
    Qjfunc : Model used for estimating Qj parameters
    objid,A_percentile

    Output:
    _____
    Tstar_idx_lst : list of Tstar index (all tiles that satisfy criterion) at every iteration
    Qj_lst : list of worker qualities at everystep
    Tstar_lst : Tstar at every step
    '''
    # ML Construction with E step as usual
    tiles = pkl.load(open(DATA_DIR+"/vtiles{}.pkl".format(objid)))
    indMat = pkl.load(open(DATA_DIR+"/indMat{}.pkl".format(objid)))
    workers = pkl.load(open(DATA_DIR+"/worker{}.pkl".format(objid)))

    tile_area = np.array(indMat[-1])

    Tstar_lst = []
    Tstar_idx_lst = []

    if DEBUG: print "Compute Core Tiles"
    Tstar,Tidx=core(tiles,indMat,1)
    Tstar_lst.append([Tstar])
    Tstar_idx_lst.append(Tidx)

    if DEBUG: print "Initialize Tiles"
    Tinit,Iidx=initT(tiles,indMat) # 50% MVT solution
    if DEBUG: print "Estimate based on initial tileset and get good Qjs"
    # In the first step our condition for initializing T* is if 
    # number of worker voted for a tile exceeds the number of workers 
    # that did not vote for a tile 
    votes =indMat[:-1].sum(axis=0)
    noVotes = len(workers)*np.ones_like(votes)-votes
    A_thres,Qjhat = estimate_Qj(tiles,indMat,workers,Qjfunc,votes,noVotes,DEBUG=DEBUG)
    Qn1,Qn2,Qp1,Qp2 = zip(*Qjhat)

    
    for i in range(Niter):
        
        if DEBUG: print "Iteration #", i
        plk=0

        if i!=0:
            # print len(tiles)
            # print len(pInT_lst)
            # print len(pNotInT_lst)
            if DEBUG: print "E-step : Estimate Qj parameters"
            A_thres,Qjhat = estimate_Qj(tiles,indMat,workers,Qjfunc,pInT_lst,pNotInT_lst)
            if DEBUG: print "Qj: ",Qjhat
            Qn1,Qn2,Qp1,Qp2 = zip(*Qjhat)
        pInT_lst = []
        pNotInT_lst = []
        
        # Go through all tiles and compute their probabilities
        for k in range(len(tiles)):
            pInT = 0
            pNotInT = 0
            tk = tiles[k]
            # Compute pInT and pNotInT
            for j in range(len(workers)):
                ljk = indMat[j][k] #NOTE k doesn't correspond to k in tiles but in current_shell_tks so this is not good
                wid=workers[j]
                qp1 = Qp1[j]
                qp2 = Qp2[j]
                qn1 = Qn1[j]
                qn2 = Qn2[j]

                if tk.area>A_thres:
                    if ljk ==1:
                        if qp1!=-1:
                            pInT+=np.log(qp1)
                        if qn1!=-1:
                            pNotInT+=np.log(1-qn1)
                    else:
                        if qp1!=-1:
                            pInT+=np.log(1-qp1)
                        if qn1!=-1:
                            pNotInT+=np.log(qn1)
                else:
                    if ljk ==1:
                        if qp2!=-1:
                            pInT+=np.log(qp2)
                        if qn2!=-1:
                            pNotInT+=np.log(1-qn2)
                    else:
                        if qp2!=-1:
                            pInT+=np.log(1-qp2)
                        if qn2!=-1:
                            pNotInT+=np.log(qn2)
            pInT_lst.append(pInT)
            pNotInT_lst.append(pNotInT)
        if DEBUG:print "pInT_lst:", pInT_lst
        if DEBUG:print "pNotInT_lst:",pNotInT_lst
        #Updates
        dump_output(objid,DATA_DIR,i,Qjhat,pInT_lst,pNotInT_lst)
        

def dump_output(objid,DATA_DIR,niter,Qj,pInT_lst,pNotInT_lst):
    pkl.dump(Qj,open("{0}/Qj_obj{1}_iter{2}.pkl".format(DATA_DIR,objid,niter),'w'))
    pkl.dump(pInT_lst,open("{0}/pInT_lst_obj{1}_iter{2}.pkl".format(DATA_DIR,objid,niter),'w'))
    pkl.dump(pNotInT_lst,open("{0}/pNotInT_lst_obj{1}_iter{2}.pkl".format(DATA_DIR,objid,niter),'w'))

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

if __name__ =="__main__":
    #DATA_DIR="final_all_tiles"
    import time
    base_dir = "uniqueTiles" 
    DEBUG=True
    #Experiments
    #mode='test'
    mode='all'
    if mode=="all":
        worker_Nbatches={5:10,10:8,15:6,20:4,25:2,30:1}
        sampleN_lst=worker_Nbatches.keys()
        for Nworker in sampleN_lst:
            for batch_id in range(worker_Nbatches[Nworker]):
                DATA_DIR=base_dir+"/{0}workers_rand{1}".format(Nworker,batch_id)
                print "Working on Batch: ",DATA_DIR
                for objid in object_lst:
		    if not os.path.isfile(DATA_DIR+"/pNotInT_lst_obj{}_iter9.pkl".format(objid)) :#and  objid!=35:
                        try:
                             print "Working on Object #",objid
                             #end = time.time()
			     try:
                                 runTileAdjacentMLConstruction(objid,workerErrorfunc="GTLSA",Qjfunc=QjGTLSA,A_percentile=-1,Niter=10,DEBUG=DEBUG,PLOT_LIKELIHOOD=False)
			     except(IOError):
		 	         if objid==35:
				     pass
                             #end2 = time.time()
                             #print "Time Elapsed: ",end2-end
                        except(shapely.geos.PredicateError):
                             print "Failed Object #",objid
     		    else:
		        print "Already ran: ",DATA_DIR+"/pNotInT_lst_obj{}_iter9.pkl".format(objid)
    else:
        Nworker = sys.argv[1]
        batch_id = sys.argv[2] 
        objid = sys.argv[3]

        # Nworker=5
        # batch_id =0
        # objid=1
        DATA_DIR=base_dir+"/{0}workers_rand{1}".format(Nworker,batch_id)
        print "Working on Batch: ",DATA_DIR
        #for objid in object_lst:
        
        try:
            #print "Working on Object #",objid
            #end = time.time()
            runTileAdjacentMLConstruction(objid,workerErrorfunc="GTLSA",Qjfunc=QjGTLSA,A_percentile=-1,Niter=10,DEBUG=DEBUG,PLOT_LIKELIHOOD=False)
            #end2 = time.time()
            #print "Time Elapsed: ",end2-end
        except(shapely.geos.PredicateError):
            print "Failed Object #",objid 
