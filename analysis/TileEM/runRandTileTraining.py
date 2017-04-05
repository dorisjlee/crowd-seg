import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from TileEM_plot_toolbox import *

DATA_DIR = "output"
os.chdir(DATA_DIR)
try: 
    topTilePickHeuristic=sys.argv[1]
except(IndexError):
    topTilePickHeuristic="area"

topk = 40
training_tbl = []
my_BBG  = pd.read_csv("../../my_ground_truth.csv")
import itertools
for objid in tqdm(object_lst):
    # print "working on obj",objid
    #Get Tile information for that object
    worker_ids,worker_precision_lst,worker_recall_lst = compute_worker_PR_obj(objid,return_worker_id=True)
    tiles = pkl.load(open("../{0}/tiles{1}.pkl".format(DATA_DIR,objid),'r'))
    worker_ids = pkl.load(open("../{0}/worker{1}.pkl".format(DATA_DIR,objid),'r'))
    indicatorMat = pkl.load(open("../{0}/indMat{1}.pkl".format(DATA_DIR,objid),'r'))
    gammas = pkl.load(open("gfile{}.pkl".format(objid),'r'))
    if  list(gammas[0])==[] and topTilePickHeuristic=='gamma' :
        print "No Gamma information for this object, going onto the next"
        continue

    #using the area information in the last row 
    tile_area = np.array(indicatorMat[-1])
    # Loop through all combinations of 20 randomly chosen tiles 
    # tile_subset_idx = np.random.choice(np.arange(len(tiles)),20)
    ground_truth_match = my_BBG[my_BBG.object_id==objid]
    x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
    BBG = shapely.geometry.Polygon(zip(x_locs,y_locs))
    tile_area_ratio = tile_area/BBG.area
     
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

    rand_subset =[]
    for i in range(3000):
        NumTilesInCombo= np.random.randint(0,topk)
        tidxInCombo= np.random.choice(tile_subset_idx,NumTilesInCombo,replace=False)
        rand_subset.append(tidxInCombo)
    # combs = []
    # for i in range(1, len(tile_subset_idx)+1):
    #     els = [list(x) for x in itertools.combinations(tile_subset_idx, i)]
    #     combs.extend(els)
    # # Compute metric values for 3000 of these tile combinations
    # rand_subset = np.random.choice(combs,3000)
    for Tprime in rand_subset:
        p,r =compute_PR(objid,Tprime,tiles)
        gvals=[]
        experiment_idx=0
        # Majority Votes 
        region_votes=[]
        Tareas=[]
        for tidx in Tprime:
            #Number of votes for that tile
            region_votes.append(np.sum(indicatorMat[:-1][:,tidx]))

            if gammas!=[]:
                gvals.append(gammas[experiment_idx][tidx])
            else:
                gvals.append(0)

            Tareas.append(Polygon(tiles[tidx]).area)
        training_tbl.append([objid,Tprime,np.sum(region_votes), np.mean(region_votes),np.sum(gvals),np.mean(gvals),np.sum(Tareas),np.mean(Tareas),p,r])

df = pd.DataFrame(training_tbl,columns=["objid","T prime","Total Votes","Average Votes","Total gamma value","Average gamma value",\
                                        "Total area","Average area","Precision","Recall"])
df.to_csv("area_based_tile_combo_metric_{}.csv".format(topTilePickHeuristic))