# Retreiving Tiles based on ground truth, compute P,R = upper limit to tile-based method 
from TileEM import *
from BB2TileExact import *
import os 
import pickle as pkl 
from  analysis_toolbox import * 
OVERLAP_THRESHOLD=0.6
base_dir = 'stored_ptk_run'
def overlap(a,b,mode='smaller'):
    if mode=="larger":
        if a.area>b.area:
            norm_area= a.area
        else:
            norm_area = b.area
    elif mode=="smaller":
        if a.area<b.area:
            norm_area= a.area
        else:
            norm_area = b.area
    try:    
        return a.intersection(b).area/norm_area
    except(shapely.geos.TopologicalError):
        return a.buffer(1e-10).intersection(b).area/norm_area
my_BBG  = pd.read_csv("my_ground_truth.csv")
worker_Nbatches={5:10,10:8,15:6,20:4,25:2,30:1}
sampleN_lst=worker_Nbatches.keys()
for Nworker in [10]:#sampleN_lst[2:]:
    for batch_num in [7]:#range(worker_Nbatches[Nworker]):
        DATA_DIR = "{0}/{1}worker_rand{2}/".format(base_dir,Nworker,batch_num)
	print "Working on ",DATA_DIR
        #Qj=[]
        best_P_lst = []
        best_R_lst = []
        A_thres_lst=[]
        df = pd.read_csv(DATA_DIR+"PR_tbl_all.csv",index_col=0)
        for objid in object_lst:
            ground_truth_match = my_BBG[my_BBG.object_id==objid]
            x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
            T = Polygon(zip(x_locs,y_locs))
            tiles = pkl.load(open(DATA_DIR+"vtiles{}.pkl".format(objid)))
            indMat = pkl.load(open(DATA_DIR+"indMat{}.pkl".format(objid)))
            workers = pkl.load(open(DATA_DIR+"/worker{}.pkl".format(objid)))
            bestTprime = []
            for i,tk in enumerate(tiles):
                if overlap(tk,T)>OVERLAP_THRESHOLD:
                    bestTprime.append(i)
            if bestTprime==[]:
                P=0
                R=0
            else:
                P,R = compute_PR(objid,np.array(bestTprime),tiles)
            tile_area = np.array(indMat[-1])
            A_thres = np.median(tile_area[bestTprime])
            best_P_lst.append(P)
            best_R_lst.append(R)
            A_thres_lst.append(A_thres)
#             print P,R,A_thres
            #for wid in workers:
            #    j = workers.index(wid)
            #    qn1,qn2,qp1,qp2 = QjGTLSA(tiles,indMat,T,j,A_thres)
            #    Qj.append([objid,wid,j,qn1,qn2,qp1,qp2])
        GTdf = pd.DataFrame(np.array([object_lst,best_P_lst,best_R_lst,A_thres_lst]).T,columns=["object_id","GT Tile-based Precision","GT Tile-based Recall","GT A_thres"])    
        GTdf.to_csv(DATA_DIR+"GT_PR.csv")
        #Qj_tbl = pd.DataFrame(Qj,columns=["object_id","worker_id","j","Qn1","Qn2","Qp1","Qp2"])
        #Qj_tbl.to_csv(DATA_DIR+"GT_Qj.csv")
#         break
#     break
