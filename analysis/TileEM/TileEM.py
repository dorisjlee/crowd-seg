import pandas as pd
from Qj_pTprime_models import *
from analysis_toolbox import *
from TileEM_plot_toolbox import *
from adjacency import *
from tqdm import tqdm
import numpy as np
import pickle as pkl
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

def initT(tiles,indMat,topk=1):
    # In the initial step, we pick T to be the top 5 area-vote score
    # where we combine the area and vote in a 1:5 ratio
    area = np.array(indMat[-1])
    votes =indMat[:-1].sum(axis=0)
    norm_area_vote = area/max(area)+5*votes/max(votes)
    tidx = np.argsort(norm_area_vote)[::-1][:topk]
    return join_tiles(tidx,tiles)[0],list(tidx)

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
def runTileMLConstruction(objid,workerErrorfunc,Qjfunc,A_percentile,QJINIT=0.6,Niter=10,DEBUG=False,PLOT_LIKELIHOOD=False):
    '''
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
    tiles = pkl.load(open(DATA_DIR+"/vtiles{}.pkl".format(objid)))
    indMat = pkl.load(open(DATA_DIR+"/indMat{}.pkl".format(objid)))
    workers = pkl.load(open(DATA_DIR+"/worker{}.pkl".format(objid)))

    tile_area = np.array(indMat[-1])
    A_thres = np.percentile(tile_area,A_percentile)
    Qj_lst=[]
    #if DEBUG: print "Coming up with T' combinations to search through"
    #Tprime_lst, Tprime_idx_lst = Tprimefunc(objid,tiles,indMat,fixedtopk=3, topk = 40,NTprimes=NTprimes)
    Tstar_lst = []
    Tstar_idx_lst =[]
    likelihood_lst=[]
    for _i in tqdm(range(Niter)):
        if DEBUG: print "Iteration #",_i
        plk=0
        if _i ==0:
            # if DEBUG: print "Initializing tiles "
            # T=initT(tiles,indMat)
            if DEBUG: print "Initializing Qjs"
            qinit = list(np.ones(len(workers))*QJINIT)
            Qjhat = np.array([qinit,qinit,qinit,qinit]).T
            Qj_lst.append(Qjhat)
        if DEBUG: print "ML construction of Tstar"

        Tidx_lst = []
        Qn1,Qn2,Qp1,Qp2 = zip(*Qjhat)
        for k,tk in enumerate(tiles):
            pInT = 0
            pNotInT = 0
            for j in range(len(workers)):
                ljk = indMat[j][k]
                wid=workers[j]
                qp1 = Qp1[j]
                qp2 = Qp2[j]
                qn1 = Qn1[j]
                qn2 = Qn2[j]
                if tk.area>A_thres:
                    if ljk ==1:
                        pInT+=np.log(qp1)
                        pNotInT+=np.log(1-qn1)
                    else:
                        pInT+=np.log(1-qp1)
                        pNotInT+=np.log(qn1)
                else:
                    if ljk ==1:
                        pInT+=np.log(qp2)
                        pNotInT+=np.log(1-qn2)
                    else:
                        pInT+=np.log(1-qp2)
                        pNotInT+=np.log(qn2)
            #Tstar_lst.append([])
            if pInT>=pNotInT:
                # if satisfy criterion, then add to Tstar
                #print Tstar_lst
                try:
                    # print "T[i]:",Tstar_lst[_i]
                    #print k
                    plk+=pInT+pNotInT
                    try:
                        Tstar_lst[_i]=[Tstar_lst[_i][0].union(tk)]
                    except(shapely.errors.TopologicalError):
                        try:
                            Tstar_lst[_i]=[Tstar_lst[_i][0].buffer(0).union(tk.buffer(-1e-10))]
                        except(shapely.errors.TopologicalError):
                            try:
                                Tstar_lst[_i]=[Tstar_lst[_i][0].buffer(-1e-10).union(tk)]
                            except(shapely.errors.TopologicalError):
                                try:
                                    Tstar_lst[_i]=[Tstar_lst[_i][0].buffer(-1e-10).union(tk.buffer(-1e-10))]
                                except(shapely.errors.TopologicalError):
                                    try:
                                        Tstar_lst[_i]=[Tstar_lst[_i][0].union(tk.buffer(1e-10))]
                                    except(shapely.errors.TopologicalError):
                                        try:
                                            Tstar_lst[_i]=[Tstar_lst[_i][0].buffer(1e-10).union(tk)]
                                        except(shapely.errors.TopologicalError):
                                            try:
                                                Tstar_lst[_i]=[Tstar_lst[_i][0].buffer(1e-10).union(tk.buffer(1e-10))]
                                            except(shapely.errors.TopologicalError):
                                                print "Shapely Topological Error: unable to add tk, Tstar unchanged; at k=",k
                                                pkl.dump(Tstar_lst[_i][0],open("problematic_Tstar_{0}.pkl".format(k),'w'))
                                                pkl.dump(tk,open("problematic_tk_{0}.pkl".format(k),'w'))
                                                pass
                                    #return Tstar_lst[_i],tk
                except(IndexError):
                    print "First occurence of tk satisfying criterion"
                    Tstar_lst.append([tk])
                #     Tstar_lst.append([tk])

                Tidx_lst.append(k)

        ############################################################################################################
        Tstar = Tstar_lst[_i][0].buffer(0)
        if DEBUG: print "E-step : Estimate Qj parameters"
        Qjhat = estimate_Qj(Tstar,tiles,indMat,workers,Qjfunc,A_percentile,DEBUG=DEBUG)
        Qj_lst.append(Qjhat)
        Tstar_idx_lst.append(Tidx_lst)
        likelihood_lst.append(plk)
    return Tstar_idx_lst , likelihood_lst, Qj_lst,Tstar_lst

def runTileMLConstruction2(objid,workerErrorfunc,Qjfunc,A_percentile,Niter=10,DEBUG=False,PLOT_LIKELIHOOD=False):
    #ML construction with ground truth Qj initialization
    tiles = pkl.load(open(DATA_DIR+"/vtiles{}.pkl".format(objid)))
    indMat = pkl.load(open(DATA_DIR+"/indMat{}.pkl".format(objid)))
    workers = pkl.load(open(DATA_DIR+"/worker{}.pkl".format(objid)))

    tile_area = np.array(indMat[-1])
    A_thres = np.percentile(tile_area,A_percentile)

    plk=0
    Qjhat = Qjs[['Qn1[A>90%]', 'Qn2[A<90%]','Qp1[A>90%]', 'Qp2[A<90%]']].as_matrix()
    if DEBUG: print "ML construction of Tstar"
    _i=0
    Tidx_lst = []
    Qn1,Qn2,Qp1,Qp2 = zip(*Qjhat)
    Tstar_lst=[]
    for k,tk in enumerate(tiles):
        pInT = 0
        pNotInT = 0
        for j in range(len(workers)):
            ljk = indMat[j][k]
            wid=workers[j]
            qp1 = Qp1[j]
            qp2 = Qp2[j]
            qn1 = Qn1[j]
            qn2 = Qn2[j]
            if tk.area>A_thres:
                if ljk ==1:
                    pInT+=np.log(qp1)
                    pNotInT+=np.log(1-qn1)
                else:
                    pInT+=np.log(1-qp1)
                    pNotInT+=np.log(qn1)
            else:
                if ljk ==1:
                    pInT+=np.log(qp2)
                    pNotInT+=np.log(1-qn2)
                else:
                    pInT+=np.log(1-qp2)
                    pNotInT+=np.log(qn2)
        #Tstar_lst.append([])
        if pInT>=pNotInT:
            # if satisfy criterion, then add to Tstar
            #print Tstar_lst
            try:
                # print "T[i]:",Tstar_lst[_i]
                #print k
                plk+=pInT+pNotInT
                try:
                    Tstar_lst[_i]=[Tstar_lst[_i][0].union(tk)]
                except(shapely.errors.TopologicalError):
                    try:
                        Tstar_lst[_i]=[Tstar_lst[_i][0].buffer(0).union(tk.buffer(-1e-10))]
                    except(shapely.errors.TopologicalError):
                        try:
                            Tstar_lst[_i]=[Tstar_lst[_i][0].buffer(-1e-10).union(tk)]
                        except(shapely.errors.TopologicalError):
                            try:
                                Tstar_lst[_i]=[Tstar_lst[_i][0].buffer(-1e-10).union(tk.buffer(-1e-10))]
                            except(shapely.errors.TopologicalError):
                                try:
                                    Tstar_lst[_i]=[Tstar_lst[_i][0].union(tk.buffer(1e-10))]
                                except(shapely.errors.TopologicalError):
                                    try:
                                        Tstar_lst[_i]=[Tstar_lst[_i][0].buffer(1e-10).union(tk)]
                                    except(shapely.errors.TopologicalError):
                                        try:
                                            Tstar_lst[_i]=[Tstar_lst[_i][0].buffer(1e-10).union(tk.buffer(1e-10))]
                                        except(shapely.errors.TopologicalError):
                                            print "Shapely Topological Error: unable to add tk, Tstar unchanged; at k=",k
                                            pkl.dump(Tstar_lst[_i][0],open("problematic_Tstar_{0}.pkl".format(k),'w'))
                                            pkl.dump(tk,open("problematic_tk_{0}.pkl".format(k),'w'))
                                            pass
                                #return Tstar_lst[_i],tk
            except(IndexError):
                print "First occurence of tk satisfying criterion"
                Tstar_lst.append([tk])
                Tidx_lst.append(k)
        Tstar = Tstar_lst[_i][0].buffer(0)
    return Tstar_lst

def runTileMLConstruction3(objid,workerErrorfunc,Qjfunc,A_percentile,Niter=10,DEBUG=False,PLOT_LIKELIHOOD=False):
    # ML Construction with T init as high confidence tiles 
    tiles = pkl.load(open(DATA_DIR+"/vtiles{}.pkl".format(objid)))
    indMat = pkl.load(open(DATA_DIR+"/indMat{}.pkl".format(objid)))
    workers = pkl.load(open(DATA_DIR+"/worker{}.pkl".format(objid)))

    tile_area = np.array(indMat[-1])
    A_thres = np.percentile(tile_area,A_percentile)
    Qj_lst=[]
    #if DEBUG: print "Coming up with T' combinations to search through"
    #Tprime_lst, Tprime_idx_lst = Tprimefunc(objid,tiles,indMat,fixedtopk=3, topk = 40,NTprimes=NTprimes)
    Tstar_lst = []
    Tstar_idx_lst =[]
    likelihood_lst=[]
    for _i in tqdm(range(Niter)):
        if DEBUG: print "Iteration #",_i
        plk=0
        Tidx_lst = []
        if _i ==0:
            if DEBUG: print "Initializing tiles "
            Tstar,tidx=initT(tiles,indMat)
            Tstar_lst.append([Tstar])
            Tstar_idx_lst.append(tidx)
        else:
            if DEBUG: print "E-step : Estimate Qj parameters"
            Qjhat = estimate_Qj(Tstar,tiles,indMat,workers,Qjfunc,A_percentile,DEBUG=DEBUG)

            if DEBUG: print "ML construction of Tstar"
            Qn1,Qn2,Qp1,Qp2 = zip(*Qjhat)
            for k,tk in enumerate(tiles):
                pInT = 0
                pNotInT = 0
                for j in range(len(workers)):
                    ljk = indMat[j][k]
                    wid=workers[j]
                    qp1 = Qp1[j]
                    qp2 = Qp2[j]
                    qn1 = Qn1[j]
                    qn2 = Qn2[j]
                    if tk.area>A_thres:
                        if ljk ==1:
                            pInT+=np.log(qp1)
                            pNotInT+=np.log(1-qn1)
                        else:
                            pInT+=np.log(1-qp1)
                            pNotInT+=np.log(qn1)
                    else:
                        if ljk ==1:
                            pInT+=np.log(qp2)
                            pNotInT+=np.log(1-qn2)
                        else:
                            pInT+=np.log(1-qp2)
                            pNotInT+=np.log(qn2)
                #Tstar_lst.append([])
                if pInT>=pNotInT:
                    # if satisfy criterion, then add to Tstar
                    #print Tstar_lst
                    try:
                        # print "T[i]:",Tstar_lst[_i]
                        #print k
                        plk+=pInT+pNotInT
                        try:
                            Tstar_lst[_i]=[Tstar_lst[_i][0].union(tk)]
                        except(shapely.errors.TopologicalError):
                            try:
                                Tstar_lst[_i]=[Tstar_lst[_i][0].buffer(0).union(tk.buffer(-1e-10))]
                            except(shapely.errors.TopologicalError):
                                try:
                                    Tstar_lst[_i]=[Tstar_lst[_i][0].buffer(-1e-10).union(tk)]
                                except(shapely.errors.TopologicalError):
                                    try:
                                        Tstar_lst[_i]=[Tstar_lst[_i][0].buffer(-1e-10).union(tk.buffer(-1e-10))]
                                    except(shapely.errors.TopologicalError):
                                        try:
                                            Tstar_lst[_i]=[Tstar_lst[_i][0].union(tk.buffer(1e-10))]
                                        except(shapely.errors.TopologicalError):
                                            try:
                                                Tstar_lst[_i]=[Tstar_lst[_i][0].buffer(1e-10).union(tk)]
                                            except(shapely.errors.TopologicalError):
                                                try:
                                                    Tstar_lst[_i]=[Tstar_lst[_i][0].buffer(1e-10).union(tk.buffer(1e-10))]
                                                except(shapely.errors.TopologicalError):
                                                    print "Shapely Topological Error: unable to add tk, Tstar unchanged; at k=",k
                                                    pkl.dump(Tstar_lst[_i][0],open("problematic_Tstar_{0}.pkl".format(k),'w'))
                                                    pkl.dump(tk,open("problematic_tk_{0}.pkl".format(k),'w'))
                                                    pass
                                        #return Tstar_lst[_i],tk
                    except(IndexError):
                        print "First occurence of tk satisfying criterion"
                        Tstar_lst.append([tk])
                    #     Tstar_lst.append([tk])

                    Tidx_lst.append(k)

            ############################################################################################################
            Tstar = Tstar_lst[_i][0].buffer(0)
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
    # DATA_DIR="output_26"
    # exp_num=11
    # objid=9
    # Tprime_lst,pTprime_lst,Qj_lst,T_lst = runTileEM2(objid,Tprime_snowball_area,pTprimeGTLSA,QjGTLSA,A_percentile=90,\
    #                                     Niter=5,NTprimes=2000,PLOT_LIKELIHOOD=False,DEBUG=True)
    # pkl.dump(Tprime_lst,open("Tprime_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    # pkl.dump(pTprime_lst,open("pTprime_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    # pkl.dump(T_lst,open("T_lst_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    # pkl.dump(Qj_lst,open("Qj_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    # end2 = time.time()
    # print "Time Elapsed: ",end2-end
    #DATA_DIR="output_15"
    #exp_num=13
    #objid=35
    #end = time.time()
    #Tstar_idx_lst ,likelihood_lst,Qj_lst,Tstar_lst=runTileMLConstruction(objid,workerErrorfunc="GTLSA",Qjfunc=QjGTLSA,A_percentile=90,Niter=10,DEBUG=True,PLOT_LIKELIHOOD=False)
    # a,b = runTileMLConstruction(objid,workerErrorfunc="GTLSA",Qjfunc=QjGTLSA,A_percentile=90,Niter=5,DEBUG=True,PLOT_LIKELIHOOD=False)
    # pkl.dump(a[0],open("a.pkl",'w'))
    # pkl.dump(b,open("b.pkl",'w'))
    #a = runTileMLConstruction(objid,workerErrorfunc="GTLSA",Qjfunc=QjGTLSA,A_percentile=90,Niter=5,DEBUG=True,PLOT_LIKELIHOOD=False)
    # pkl.dump(a[0],open("a.pkl",'w'))
    #pkl.dump(likelihood_lst,open("likelihood_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    #pkl.dump(Tstar_lst,open("Tstar_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    #pkl.dump(Tstar_idx_lst,open("Tstar_idx_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    #pkl.dump(Qj_lst,open("Qj_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    #end2 = time.time()
    #print "Time Elapsed: ",end2-end
    # DATA_DIR="output_15"
    # exp_num=14
    # objid=26
    # end = time.time()
    # Tstar_idx_lst ,likelihood_lst,Qj_lst,Tstar_lst=runTileMLConstruction(objid,workerErrorfunc="GTLSA",Qjfunc=QjGTLSA,A_percentile=90,Niter=10,DEBUG=True,PLOT_LIKELIHOOD=False)
    # # a,b = runTileMLConstruction(objid,workerErrorfunc="GTLSA",Qjfunc=QjGTLSA,A_percentile=90,Niter=5,DEBUG=True,PLOT_LIKELIHOOD=False)
    # # pkl.dump(a[0],open("a.pkl",'w'))
    # # pkl.dump(b,open("b.pkl",'w'))
    # #a = runTileMLConstruction(objid,workerErrorfunc="GTLSA",Qjfunc=QjGTLSA,A_percentile=90,Niter=5,DEBUG=True,PLOT_LIKELIHOOD=False)
    # # pkl.dump(a[0],open("a.pkl",'w'))
    # pkl.dump(likelihood_lst,open("likelihood_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    # pkl.dump(Tstar_lst,open("Tstar_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    # pkl.dump(Tstar_idx_lst,open("Tstar_idx_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    # pkl.dump(Qj_lst,open("Qj_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    # end2 = time.time()
    # print "Time Elapsed: ",end2-end

    # DATA_DIR="output_15"
    # exp_num=0
    # objid=26
    # end = time.time()
    # Qjs = pd.read_csv("final_all_tiles/worker_obj_qualities_all.csv")
    # Tstar =runTileMLConstruction2(objid,workerErrorfunc="GTLSA",Qjfunc=QjGTLSA,A_percentile=90,Niter=100,DEBUG=True,PLOT_LIKELIHOOD=False)
    # pkl.dump(Tstar,open("Tstar_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    # end2 = time.time()
    # print "Time Elapsed: ",end2-end
    
    #QJINIT=0.5
    DATA_DIR="output_15"
    exp_num=14
    objid=3
    end = time.time()
    Tstar_idx_lst ,likelihood_lst,Qj_lst,Tstar_lst=runTileMLConstruction(objid,workerErrorfunc="GTLSA",Qjfunc=QjGTLSA,A_percentile=90,QJINIT=0.5, Niter=100,DEBUG=True,PLOT_LIKELIHOOD=False)
    pkl.dump(likelihood_lst,open("likelihood_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    pkl.dump(Tstar_lst,open("Tstar_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    pkl.dump(Tstar_idx_lst,open("Tstar_idx_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    pkl.dump(Qj_lst,open("Qj_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    end2 = time.time()
    print "Time Elapsed: ",end2-end

    #QJINIT=0.6
    DATA_DIR="output_15"
    exp_num=15
    print "Running experiment #",exp_num
    objid=3
    end = time.time()
    Tstar_idx_lst ,likelihood_lst,Qj_lst,Tstar_lst=runTileMLConstruction(objid,workerErrorfunc="GTLSA",Qjfunc=QjGTLSA,A_percentile=90,QJINIT=0.6, Niter=100,DEBUG=True,PLOT_LIKELIHOOD=False)
    pkl.dump(likelihood_lst,open("likelihood_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    pkl.dump(Tstar_lst,open("Tstar_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    pkl.dump(Tstar_idx_lst,open("Tstar_idx_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    pkl.dump(Qj_lst,open("Qj_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    end2 = time.time()
    print "Time Elapsed: ",end2-end

    #QJINIT=0.7
    DATA_DIR="output_15"
    exp_num=16
    print "Running experiment #",exp_num
    objid=3
    end = time.time()
    Tstar_idx_lst ,likelihood_lst,Qj_lst,Tstar_lst=runTileMLConstruction(objid,workerErrorfunc="GTLSA",Qjfunc=QjGTLSA,A_percentile=90,QJINIT=0.7, Niter=100,DEBUG=True,PLOT_LIKELIHOOD=False)
    pkl.dump(likelihood_lst,open("likelihood_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    pkl.dump(Tstar_lst,open("Tstar_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    pkl.dump(Tstar_idx_lst,open("Tstar_idx_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    pkl.dump(Qj_lst,open("Qj_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    end2 = time.time()
    print "Time Elapsed: ",end2-end

    # T init with high confidence snowball 
    DATA_DIR="output_15"
    exp_num=17
    print "Running experiment #",exp_num
    objid=3
    end = time.time()
    Tstar_idx_lst ,likelihood_lst,Qj_lst,Tstar_lst=runTileMLConstruction3(objid,workerErrorfunc="GTLSA",Qjfunc=QjGTLSA,A_percentile=90,Niter=100,DEBUG=True,PLOT_LIKELIHOOD=False)
    pkl.dump(likelihood_lst,open("likelihood_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    pkl.dump(Tstar_lst,open("Tstar_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    pkl.dump(Tstar_idx_lst,open("Tstar_idx_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    pkl.dump(Qj_lst,open("Qj_exp{0}_obj{1}.pkl".format(exp_num,objid),'w'))
    end2 = time.time()
    print "Time Elapsed: ",end2-end

