import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from TileEM_plot_toolbox import *
from TileEM_Models import *
gamma_properties=False
DATA_DIR="sampletopworst5"
os.chdir(DATA_DIR)
try: 
    topTilePickHeuristic=sys.argv[1]
except(IndexError):
    topTilePickHeuristic="area"

# topk = 40
training_tbl = []
my_BBG  = pd.read_csv("../../my_ground_truth.csv")
import itertools
topk = 10
selected_objids=[ 5,  6,  8,  9, 14, 15, 18, 19, 20, 21, 23, 24, 25, 27, 29, 30,33, 35, 37, 46]
for objid in tqdm(selected_objids):
    print "Working on obj:",objid
    #Get Tile information for that object
    #worker_ids,worker_precision_lst,worker_recall_lst = compute_worker_PR_obj(objid,return_worker_id=True)
    tiles = pkl.load(open("vtiles{}.pkl".format(objid),'r'))
    worker_ids = pkl.load(open("worker{}.pkl".format(objid),'r'))
    indicatorMat = pkl.load(open("indMat{}.pkl".format(objid),'r'))
    if gamma_properties : 
        gammas = pkl.load(open("gfile{}.pkl".format(objid),'r'))
        if  list(gammas[0])==[] and topTilePickHeuristic=='gamma' :
            print "No Gamma information for this object, going onto the next"
            continue

    #using the area information in the last row 
    tile_area = np.array(indicatorMat[-1])
    # Loop through all combinations of 20 randomly chosen tiles 
    tile_subset_idx = np.random.choice(np.arange(len(tiles)),topk)
    ground_truth_match = my_BBG[my_BBG.object_id==objid]
    x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
    BBG = shapely.geometry.Polygon(zip(x_locs,y_locs))
    tile_area_ratio = tile_area/BBG.area
    print "Picked top-k tiles"
    if topTilePickHeuristic=="area": 
        # Loop through all combinations of 40 top-area tiles 
        # print "area heuristic"
        tile_subset_idx = tile_area_ratio.argsort()[::-1][:topk]
    elif topTilePickHeuristic=="gamma":
        # print "gamma"
        tile_subset_idx = np.argsort(gammas[0])[::-1][:topk]
    elif topTilePickHeuristic=="majvote":
        # print "majvote"
        tile_votes = np.sum(indicatorMat[:-1],axis=0)
        tile_subset_idx = np.argsort(tile_votes)[::-1][:topk]
    print "Creating random subsets from topk tiles"
    rand_subset =[]
    for i in range(500):#3000
        NumTilesInCombo= np.random.randint(1,topk)#at least one tile must be selected
        tidxInCombo= np.random.choice(tile_subset_idx,NumTilesInCombo,replace=False)
        rand_subset.append(tidxInCombo)
    print "Compute feature properties for T prime "
    for Tprime in tqdm(rand_subset):
        p,r =compute_PR(objid,Tprime,tiles)
        gvals=[]
        experiment_idx=0
        # Majority Votes 
        region_votes=[]
        Tareas=[]
        for tidx in Tprime:
            #Number of votes for that tile
            region_votes.append(np.sum(indicatorMat[:-1][:,tidx]))

            if gamma_properties:
                if gammas!=[]:
                    gvals.append(gammas[experiment_idx][tidx])
                else:
                    gvals.append(0)

            Tareas.append(Polygon(tiles[tidx]).area)
        pTprime_val = pTprimeBasic(objid,Tprime,BBG)
        pTprimeLSA_val1 = pTprimeLSA(objid,Tprime,BBG,1)
        pTprimeLSA_val10 = pTprimeLSA(objid,Tprime,BBG,10)
        pTprimeLSA_val50 = pTprimeLSA(objid,Tprime,BBG,50)
        pTprimeLSA_val100 = pTprimeLSA(objid,Tprime,BBG,100)
        #if pTprime_val==0: break
        training_tbl.append([objid,Tprime,np.sum(region_votes), np.mean(region_votes),np.sum(gvals),np.mean(gvals),\
                             np.sum(Tareas),np.mean(Tareas),pTprime_val,pTprimeLSA_val1,pTprimeLSA_val10,pTprimeLSA_val50,\
                             pTprimeLSA_val100,p,r])


df = pd.DataFrame(training_tbl,columns=["objid","T prime","Total Votes","Average Votes","Total gamma value","Average gamma value",\
                                         "Total area","Average area","pTprime","pTprime[Athres>1]","pTprime[Athres>10]",\
                                         "pTprime[Athres>50]","pTprime[Athres>100]","Precision","Recall"])
df.to_csv("all_tile_combo_metric_{}.csv".format(topTilePickHeuristic))