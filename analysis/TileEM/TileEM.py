import pandas as pd
from Qj_pTprime_models import *
from analysis_toolbox import *
from TileEM_plot_toolbox import *
from adjacency import *
from TileEM import *
from tqdm import tqdm
import numpy as np
import pickle as pkl
def estimate_Qj(T,tiles,indMat,workers,Qjfunc,A_thres,DEBUG=False):
    Qj=[]
    for wid,j in zip(workers,range(len(workers))):
        Qj.append(Qjfunc(tiles,indMat,T,j,A_thres))
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
    return join_tiles(tidx,tiles)[0],list(tidx)
def find_all_tk_in_shell(tiles,current_shell_idx,exclude_idx=[]):
    # Find all tiles at the shell d=d+1
    # add all tiles adjacent to currentShell front
    filtered_tidxs = np.delete(np.arange(len(tiles)),exclude_idx)

    adjacent_tkidxs =[]
    for ctidx in current_shell_idx:
        ck = tiles[ctidx]
        for tkidx in filtered_tidxs:
            tk = tiles[tkidx]
            if adjacent(tk,ck):
                adjacent_tkidxs.append(tkidx)
    # There might be a lot of duplicate tiles that is adjacent to more than one tile on the current shell front
    return list(set(adjacent_tkidxs))
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

    Qj_lst=[]
    #if DEBUG: print "Coming up with T' combinations to search through"
    #Tprime_lst, Tprime_idx_lst = Tprimefunc(objid,tiles,indMat,fixedtopk=3, topk = 40,NTprimes=NTprimes)
    Tstar_lst = []
    Tstar_idx_lst =[]
    likelihood_lst=[]

    if DEBUG: print "Compute Core Tiles"
    Tstar,Tidx=core(tiles,indMat,1)
    Tstar_lst.append([Tstar])
    Tstar_idx_lst.append(Tidx)

    if DEBUG: print "Initialize Tiles"
    Tinit,Iidx=initT(tiles,indMat)
    if DEBUG: print "Estimate based on initial tileset and get good Qjs"
    if A_percentile!=-1:
        A_thres = np.percentile(tile_area,A_percentile)
    else:
        A_thres = np.median(tile_area[Iidx])
    Qjhat = estimate_Qj(Tinit,tiles,indMat,workers,Qjfunc,A_thres=A_thres,DEBUG=DEBUG)
    Qn1,Qn2,Qp1,Qp2 = zip(*Qjhat)

    for i in tqdm(range(Niter)):
        if DEBUG: print "Iteration #", i
        plk=0
        if i!=0:
            if DEBUG: print "E-step : Estimate Qj parameters"
            A_thres = np.median(tile_area[Tstar_idx_lst[i]])
            print "Median Area Threshold:",A_thres
            Qjhat = estimate_Qj(Tstar_lst[i][0],tiles,indMat,workers,Qjfunc,A_thres=A_thres,DEBUG=DEBUG)
            Qn1,Qn2,Qp1,Qp2 = zip(*Qjhat)

        if DEBUG: print "ML construction of Tstar"
        dPrime = 0

        exclude_idx = set(Tstar_idx_lst[0])
        Tidx_lst = list(exclude_idx)
        good_dPrime_tcount = len(exclude_idx)
        current_shell_tkidxs= Tidx
        past_shell_tkidxs= Tidx
        if DEBUG: print "Add core tiles to first occurence of tk satisfying criterion"
        Tstar_lst.append([Tstar_lst[0][0]])


        while (good_dPrime_tcount!=0 or len(current_shell_tkidxs)!=0):
            ######
            print "Excluding",exclude_idx
            current_shell_tkidxs = find_all_tk_in_shell(tiles,past_shell_tkidxs,list(exclude_idx))

            if DEBUG:
                print "d'={0}; good_dPrime_tcount={1}".format(dPrime,good_dPrime_tcount)
                print "Number of tks in shell: ",len(current_shell_tkidxs)
                print "Current shell index:",current_shell_tkidxs
            good_dPrime_tcount=0

            for k in current_shell_tkidxs:
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
                # Check if tk satisfy constraint
                if pInT<pNotInT:
                    plk+=pNotInT
                elif pInT>=pNotInT:
                    plk+=pInT
                    # if satisfy criterion, then add to Tstar
                    good_dPrime_tcount+=1
                    if DEBUG: print "Adding tk",k
                    try:
                        Tstar_lst[i]=[Tstar_lst[i][0].union(tk)]
                        Tidx_lst.append(k)
                    except(shapely.geos.TopologicalError):
                        try:
                            Tstar_lst[i]=[Tstar_lst[i][0].buffer(0).union(tk.buffer(-1e-10))]
                            Tidx_lst.append(k)
                        except(shapely.geos.TopologicalError):
                            try:
                                Tstar_lst[i]=[Tstar_lst[i][0].buffer(-1e-10).union(tk)]
                                Tidx_lst.append(k)
                            except(shapely.geos.TopologicalError):
                                try:
                                    Tstar_lst[i]=[Tstar_lst[i][0].buffer(-1e-10).union(tk.buffer(-1e-10))]
                                    Tidx_lst.append(k)
                                except(shapely.geos.TopologicalError):
                                    try:
                                        Tstar_lst[i]=[Tstar_lst[i][0].union(tk.buffer(1e-10))]
                                        Tidx_lst.append(k)
                                    except(shapely.geos.TopologicalError):
                                        try:
                                            Tstar_lst[i]=[Tstar_lst[i][0].buffer(1e-10).union(tk)]
                                            Tidx_lst.append(k)
                                        except(shapely.geos.TopologicalError):
                                            try:
                                                Tstar_lst[i]=[Tstar_lst[i][0].buffer(1e-10).union(tk.buffer(1e-10))]
                                                Tidx_lst.append(k)
                                            except(shapely.geos.TopologicalError):
                                                print "Shapely Topological Error: unable to add tk, Tstar unchanged; at k=",k
                                                pkl.dump(Tstar_lst[i][0],open("problematic_Tstar_{0}.pkl".format(k),'w'))
                                                pkl.dump(tk,open("problematic_tk_{0}.pkl".format(k),'w'))
                                                pass

            ############################################################################################################
            if PLOT:
                plt.figure()
                for c in current_shell_tkidxs:plot_coords(tiles[c],color="red",fill_color="red") #current front
                for c in past_shell_tkidxs:plot_coords(tiles[c],color="cyan",linewidth=5,linestyle='--') #past front
                for c in exclude_idx:plot_coords(tiles[c],color="gray",fill_color="gray")#excluded coord
                plot_coords(Tstar_lst[i][0],linestyle="--",linewidth=2,color="blue")#current Tstar
                for c in Tidx_lst:plot_coords(tiles[c],linewidth=2,color="green",fill_color="green")#new Tstar
                plt.ylim(40,100)



            #Updates
            Tstar = Tstar_lst[i][0].buffer(0)
            dPrime+=1
            past_shell_tkidxs= current_shell_tkidxs
            exclude_idx= exclude_idx.union(current_shell_tkidxs)


        #Storage
        Qj_lst.append(Qjhat)
        Tstar_idx_lst.append(Tidx_lst)
        likelihood_lst.append(plk)

    return Tstar_idx_lst , likelihood_lst, Qj_lst,Tstar_lst

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
    #Experiments
    for batch_id in range(9):
	print "Working on Batch #",batch_id
    	DATA_DIR="sample/5worker_rand{}".format(batch_id)
    	for objid in object_lst:
            print "Working on Object #",objid
            try:
                end = time.time()
                Tstar_idx_lst ,likelihood_lst,Qj_lst,Tstar_lst,pInT,pNotInT=runTileAdjacentMLConstruction(objid,workerErrorfunc="GTLSA",Qjfunc=QjGTLSA,A_percentile=-1,Niter=5,DEBUG=True,PLOT_LIKELIHOOD=False)
                pkl.dump(likelihood_lst,open(DATA_DIR+"/likelihood_obj{}.pkl".format(objid),'w'))
                pkl.dump(Tstar_lst,open(DATA_DIR+"/Tstar_obj{}.pkl".format(objid),'w'))
                pkl.dump(Tstar_idx_lst,open(DATA_DIR+"/Tstar_idx_obj{}.pkl".format(objid),'w'))
                pkl.dump(Qj_lst,open(DATA_DIR+"/Qj_obj{}.pkl".format(objid),'w'))
                end2 = time.time()
                print "Time Elapsed: ",end2-end
            except:
                print "Object #{} failed".format(objid)
