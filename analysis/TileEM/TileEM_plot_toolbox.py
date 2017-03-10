from dataset import Dataset
from BB2tile import *
from greedy import *
from data import *
from experiment import *
import pandas as pd
import pickle as pkl
import shapely
from shapely.ops import cascaded_union
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import warnings

DATA_DIR="output"
warnings.filterwarnings('ignore')

my_BBG  = pd.read_csv("../my_ground_truth.csv")
object_lst = list(object_tbl.id)
def compute_PR(objid,solnset,tiles):
    '''
    Compute precision recall against ground truth bounding box
    for a given solution set and tile coordinates.
    '''
    try:
        ML_regions = join_tiles(solnset,tiles)
    except(ValueError):
        return -1,-1
    ground_truth_match = my_BBG[my_BBG.object_id==objid]
    x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
    BBG = shapely.geometry.Polygon(zip(x_locs,y_locs))
    recall = ML_regions.intersection(BBG).area/float(BBG.area)
    if float(ML_regions.area)!=0:
        precision = ML_regions.intersection(BBG).area/float(ML_regions.area)
    else:
        # Empty solution set
        precision = -1
    return precision,recall

def getSolutionThreshold(gammas,threshold=0.5):
	'''
	Derive a solution set of tiles in BB for a given gamma tile values 
	Tiles for which gamma > threshold is included in the solution set
	'''
	if gammas is None:
	# In the case when the CVX solver can not find a solution, it returns gamma as None and l as inf. 
	# This happens for Median or Average case, where   your T value is just very off, so you can't really find a good ML region corresponding to the T constraints.
	# This means that our solution set should be empty.
		return []
	solutionList = []
	for i,gamma in enumerate(gammas):
		if gamma > threshold:
			solutionList.append(i+1)
	return solutionList

def getSolutionTopK(data,k=5):
	#Derive a solution set of tiles of top-K gamma tile values 
	return np.argsort(data)[::-1][:k]

def compute_worker_PR_obj(objid):
    # List of PR measures of all workers 
    precision_lst = []
    recall_lst = []
    objBBs = bb_info[bb_info.object_id==objid]
    ground_truth_match = my_BBG[my_BBG.object_id==objid]
    BBG_x_locs,BBG_y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
    for bb in objBBs.iterrows():
        oid = bb[1]["object_id"]
        bbx_path= bb[1]["x_locs"]
        bby_path= bb[1]["y_locs"]
        worker_x_locs,worker_y_locs= process_raw_locs([bbx_path,bby_path])
        precision_lst.append(precision([worker_x_locs,BBG_x_locs],[worker_y_locs,BBG_y_locs]))
        recall_lst.append(recall([worker_x_locs,BBG_x_locs],[worker_y_locs,BBG_y_locs]))
    return np.array(precision_lst),np.array(recall_lst)

def compute_PR_obj(objid,experiment_idx=0,threshold=-1,topk=-1,majority_topk=-1,PLOT_HEATMAP=False):
    '''
    Compute the precision recall for a object indexed by objid
    based on T-search strategy indexed by experiment_idx, where:
    0 = Avrg 
    1 = Median
    2 = Local Search
    3 = Exhaustive Search
    The post-processing method could either be gamma threshold, top-k gamma, or top-k majority vote.
    OPTIONAL: Plot Gamma Heatmap 
    '''
    
    precision_lst = []
    recall_lst = []
    try:
        tiles = pkl.load(open("tiles{}.pkl".format(objid),'r'))
        gammas = pkl.load(open("gfile{}.pkl".format(objid),'r'))#ga,gm,gl,ge
        # Deriving new solution set from different thresholding criteria
        if majority_topk==-1:
            if threshold!=-1:
                procstr = "Gamma>{}".format(threshold)
                solnset = getSolutionThreshold(gammas[experiment_idx],threshold=threshold)
            elif topk!=-1:
                procstr="Gamma k={}".format(topk)
                solnset = getSolutionTopK(gammas[experiment_idx],k=topk)
            if PLOT_HEATMAP: 
                mask = plot_tile_heatmap(objid,solnset ,tiles,gammas[experiment_idx],PLOT_BBG=True,PLOT_GSOLN=True)
        elif majority_topk!=-1:
            os.chdir("..")
            tiles, objIndicatorMat = createObjIndicatorMatrix(objid,PRINT=False)
            os.chdir(DATA_DIR)
            tile_votes = np.sum(objIndicatorMat[:-1],axis=0)
            procstr="Majority k={}".format(majority_topk)
            solnset = getSolutionTopK(tile_votes,k=majority_topk)
            if PLOT_HEATMAP: 
                mask = plot_tile_heatmap(objid,solnset ,tiles,tile_votes,PLOT_BBG=True,PLOT_GSOLN=True)
        precision,recall = compute_PR(objid,solnset,tiles)
        if PLOT_HEATMAP:
	        font0 = matplotlib.font_manager.FontProperties()
	        font1=font0.copy()
	        font1.set_size('xx-large')
	        font1.set_weight('heavy')
	        plt.figtext(0.45,0.7,'{0}\n P={1:.2f}, R={2:.2f}'.format(procstr,precision,recall),color='white',fontproperties=font1,ha='center')
    except(IOError):
        pass
    return precision,recall


def plot_all_postprocess_PR_curves(objid,legend=False):
    '''
	Plot PR curves for each object for all post-processing methods
	'''
    plt.figure()
    plt.title("Object #{}".format(objid))
    # Worker Individual Precision and Recall based on their BB drawn for this object
    worker_precision_lst,worker_recall_lst = compute_worker_PR_obj(objid)
    plt.plot(worker_recall_lst ,worker_precision_lst , '.',color="red",label="Worker")
    # Plotting PR from Top-k Majority vote 
    os.chdir("..")
    tiles, objIndicatorMat = createObjIndicatorMatrix(objid,PRINT=False)
    os.chdir(DATA_DIR)
    k_lst = np.arange(1,len(tiles))
    Maj_topk_precision_lst = []
    Maj_topk_recall_lst = []
    for  k in k_lst :
        Maj_topk_precision,Maj_topk_recall= compute_PR_obj(objid,2,topk=k)
        Maj_topk_precision_lst.append(Maj_topk_precision)
        Maj_topk_recall_lst.append(Maj_topk_recall)
    Maj_topk_recall_lst = np.array(Maj_topk_recall_lst)
    Maj_topk_precision_lst = np.array(Maj_topk_precision_lst)
    order = np.argsort(Maj_topk_recall_lst)
    plt.plot(Maj_topk_recall_lst[order],Maj_topk_precision_lst[order], linestyle='-', linewidth=1,marker='D',color="cyan",label="Majority Top-k")

    # Plotting PR from TileEM for different thresholds
    threshold_lst = np.linspace(0,0.95,20)
    TileEM_thres_precision_lst = []
    TileEM_thres_recall_lst = []
    for threshold in threshold_lst :
        TileEM_thres_precision, TileEM_thres_recall= compute_PR_obj(objid,2,threshold=threshold)
        TileEM_thres_precision_lst.append(TileEM_thres_precision)
        TileEM_thres_recall_lst.append(TileEM_thres_recall)
    TileEM_thres_recall_lst = np.array(TileEM_thres_recall_lst)
    TileEM_thres_precision_lst = np.array(TileEM_thres_precision_lst)
        # print "{0}:{1},{2}".format(threshold,TileEM_thres_precision,TileEM_thres_recall)
    order = np.argsort(TileEM_thres_recall_lst)
    plt.plot(TileEM_thres_recall_lst[order],TileEM_thres_precision_lst[order], linestyle='-',linewidth=2,ms=13, marker='x',color="blue",label="TileEM Thresh")
    # Plotting PR from TileEM for different Top-k
    TileEM_topk_precision_lst = []
    TileEM_topk_recall_lst = []
    for  k in k_lst :
        TileEM_topk_precision, TileEM_topk_recall= compute_PR_obj(objid,2,topk=k)
        TileEM_topk_precision_lst.append(TileEM_topk_precision)
        TileEM_topk_recall_lst.append(TileEM_topk_recall)
    TileEM_topk_recall_lst = np.array(TileEM_topk_recall_lst)
    TileEM_topk_precision_lst = np.array(TileEM_topk_precision_lst)
    order = np.argsort(TileEM_topk_recall_lst)
    plt.plot(TileEM_topk_recall_lst[order],TileEM_topk_precision_lst[order], linestyle='--', linewidth=3,marker='o',color="green",label="TileEM Top-k")

    plt.xlim(0,1.05)
    plt.ylim(0,1.05)
    plt.ylabel("Precision",fontsize=13)
    plt.xlabel("Recall",fontsize=13)
    if legend: plt.legend(loc="top left",numpoints=1)
    plt.savefig("PR_obj{}.pdf".format(objid))
def compute_joined_PR(objid):
    '''
	Compute the PR values for all post-processing method
	'''
    # Worker Individual Precision and Recall based on their BB drawn for this object
    worker_precision_lst,worker_recall_lst = compute_worker_PR_obj(objid)
    # Plotting PR from Top-k Majority vote 
    os.chdir("..")
    tiles, objIndicatorMat = createObjIndicatorMatrix(objid,PRINT=False)
    os.chdir(DATA_DIR)
    k_lst = np.arange(1,len(tiles))
    Maj_topk_precision_lst = []
    Maj_topk_recall_lst = []
    for  k in k_lst :
        Maj_topk_precision,Maj_topk_recall= compute_PR_obj(objid,2,topk=k)
        Maj_topk_precision_lst.append(Maj_topk_precision)
        Maj_topk_recall_lst.append(Maj_topk_recall)
    Maj_topk_recall_lst = np.array(Maj_topk_recall_lst)
    Maj_topk_precision_lst = np.array(Maj_topk_precision_lst)
    # Plotting PR from TileEM for different thresholds
    threshold_lst = np.linspace(0,0.95,20)
    TileEM_thres_precision_lst = []
    TileEM_thres_recall_lst = []
    for threshold in threshold_lst :
        TileEM_thres_precision, TileEM_thres_recall= compute_PR_obj(objid,2,threshold=threshold)
        TileEM_thres_precision_lst.append(TileEM_thres_precision)
        TileEM_thres_recall_lst.append(TileEM_thres_recall)
    TileEM_thres_recall_lst = np.array(TileEM_thres_recall_lst)
    TileEM_thres_precision_lst = np.array(TileEM_thres_precision_lst)
    # Plotting PR from TileEM for different Top-k
    TileEM_topk_precision_lst = []
    TileEM_topk_recall_lst = []
    for  k in k_lst :
        TileEM_topk_precision, TileEM_topk_recall= compute_PR_obj(objid,2,topk=k)
        TileEM_topk_precision_lst.append(TileEM_topk_precision)
        TileEM_topk_recall_lst.append(TileEM_topk_recall)
    TileEM_topk_recall_lst = np.array(TileEM_topk_recall_lst)
    TileEM_topk_precision_lst = np.array(TileEM_topk_precision_lst)
    return Maj_topk_recall_lst ,Maj_topk_precision_lst,\
            TileEM_thres_recall_lst ,TileEM_thres_precision_lst,\
            TileEM_topk_recall_lst ,TileEM_topk_precision_lst ,\
            worker_recall_lst ,worker_precision_lst 
def plot_joined_PR_curves():
    '''
    Plot Combined PR Curve for all object for all post-processing method
    '''
    #Compute Precision Recall values for each post-processing method
    Maj_topk_recall_lst =[]
    Maj_topk_precision_lst=[]
    TileEM_thres_recall_lst =[]
    TileEM_thres_precision_lst=[]
    TileEM_topk_recall_lst=[]
    TileEM_topk_precision_lst =[]
    worker_recall_lst =[]
    worker_precision_lst=[]
    for objid in tqdm(object_lst):   
        Maj_topk_recall, Maj_topk_precision ,TileEM_thres_recall,TileEM_thres_precision,\
        TileEM_topk_recall  ,TileEM_topk_precision ,worker_recall,worker_precision =compute_joined_PR(objid)
        Maj_topk_recall_lst.extend(Maj_topk_recall)
        Maj_topk_precision_lst.extend(Maj_topk_precision)
        TileEM_thres_recall_lst.extend(TileEM_thres_recall)
        TileEM_thres_precision_lst.extend(TileEM_thres_precision)
        TileEM_topk_recall_lst.extend(TileEM_topk_recall)
        TileEM_topk_precision_lst.extend(TileEM_topk_precision)
        worker_recall_lst.extend(worker_recall)
        worker_precision_lst.extend(worker_precision)
    # Plot Combined PR Curve
    plt.figure()
    plt.title("Combined PR Curve for all objects")

    plt.plot(worker_recall_lst ,worker_precision_lst , '.',color="red",label="Worker")

    Maj_topk_recall_lst = np.array(Maj_topk_recall_lst)
    Maj_topk_precision_lst = np.array(Maj_topk_precision_lst)
    order = np.argsort(Maj_topk_recall_lst)
    plt.plot(Maj_topk_recall_lst[order],Maj_topk_precision_lst[order], linestyle='-', linewidth=1,marker='D',color="cyan",label="Majority Top-k")

    TileEM_thres_recall_lst = np.array(TileEM_thres_recall_lst)
    TileEM_thres_precision_lst = np.array(TileEM_thres_precision_lst)
    order = np.argsort(TileEM_thres_recall_lst)
    plt.plot(TileEM_thres_recall_lst[order],TileEM_thres_precision_lst[order], linestyle='-',linewidth=2,ms=13, marker='x',color="blue",label="TileEM Thresh")


    TileEM_topk_recall_lst = np.array(TileEM_topk_recall_lst)
    TileEM_topk_precision_lst = np.array(TileEM_topk_precision_lst)
    order = np.argsort(TileEM_topk_recall_lst)
    plt.plot(TileEM_topk_recall_lst[order],TileEM_topk_precision_lst[order], linestyle='--', linewidth=3,marker='o',color="green",label="TileEM Top-k")
    plt.xlim(0,1.05)
    plt.ylim(0,1.05)
    plt.ylabel("Precision",fontsize=13)
    plt.xlabel("Recall",fontsize=13)
    plt.savefig("PR_obj_all.pdf")
def interpolate_gaps(values, limit=None):
    """
    Fill gaps using linear interpolation, optionally only fill gaps up to a
    size of `limit`.
    """
    values = np.asarray(values)
    i = np.arange(values.size)
    valid = np.isfinite(values)
    filled = np.interp(i, i[valid], values[valid])

    if limit is not None:
        invalid = ~valid
        for n in range(1, limit+1):
            invalid[:-n] &= invalid[n:]
        filled[invalid] = np.nan

    return filled

#####################################
## Tile Heatmap visualization Code ##
#####################################
def join_tiles(solutionList,tiles): 
    '''
    Given a solutionList of tile indicies, join the tiles together into a Polygon/MultiPolygon object.
    '''
    return cascaded_union([shapely.geometry.Polygon(zip(tiles[int(tidx)-1][:,1],tiles[int(tidx)-1][:,0])) for tidx in solutionList])

def plot_tile_heatmap(objid,solnset,tiles,gamma,PLOT_BBG=False,PLOT_GSOLN=False):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon
    '''
	Plot Tile Heatmap for the object 
	Optional: include ground truth (BBG) or Gamma tile solution (GSOLN)
	'''
    area_lst=[]

    for tile in tiles:
        area_lst.append(shapely.geometry.Polygon(tile).area)
    sorted_ascend_tile_by_size= list(np.array(tiles)[np.argsort(area_lst)])
    fig,ax = plt.subplots(1)
    if PLOT_GSOLN:
        # Cascade Union creates a Polygon or MultiPolygon Object
        try:
            ML_regions = join_tiles(solnset,tiles)
        except(ValueError):
            return 
        if type(ML_regions)==shapely.geometry.polygon.Polygon:
            x,y=ML_regions.exterior.xy
            plt.plot(x, y, '-', color='lime',linewidth=1)    
        else:
            for region in ML_regions:
                x,y=region.exterior.xy
                plt.plot(x, y, '-', color='lime',linewidth=1)    
    if PLOT_BBG:
        my_BBG  = pd.read_csv("../../my_ground_truth.csv")
        ground_truth_match = my_BBG[my_BBG.object_id==objid]
        x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
        plt.plot(x_locs,y_locs,'--',color='#0000ff',linewidth=1.5)

    # Gamma Tiles 
    patches = []
    for tile_idx in range(len(tiles)):
        tile =  sorted_ascend_tile_by_size[tile_idx]
        polygon = Polygon(zip(tile[:,1],tile[:,0]),closed=True,\
                          linewidth=1,edgecolor='black',fill=False)
        patches.append(polygon)

    collection = PatchCollection(patches,cmap=matplotlib.cm.autumn_r,alpha=0.25)
    pcollection = ax.add_collection(collection)
    
    collection.set_array(gamma)
    ax.add_collection(collection)
    plt.colorbar(collection)
    img_name = img_info[img_info.id==int(object_tbl[object_tbl.id==objid]["image_id"])]["filename"].iloc[0]
    fname = "../../../web-app/app/static/"+img_name+".png"
    plt.title("Object {0} [{1}]".format(objid,object_tbl[object_tbl.object_id==objid]["name"].iloc[0]))
    img =mpimg.imread(fname)
    width,height = get_size(fname)
    plt.xlim(0,width)
    plt.ylim(height,0)
    plt.imshow(img,alpha=0.8)
    ax.autoscale_view()
def plot_all_obj_tiles(experiment_idx,threshold=0.01,PLOT_BBG=True,PLOT_GSOLN=True):
    '''
    Plot Tile Heatmap for all objects for a given experiment index
    Experiment index 
    0 = Avrg 
    1 = Median
    2 = Local Search
    3 = Exhaustive Search
    '''
    for objid in object_lst:
        try:
            tiles = pkl.load(open("tiles{}.pkl".format(objid),'r'))
            gammas = pkl.load(open("gfile{}.pkl".format(objid),'r'))#ga,gm,gl,ge
            # Deriving new solution set from different thresholding criteria
            solnset =  getSolutionThreshold(gammas[experiment_idx],threshold=threshold)
            mask = plot_tile_heatmap(objid,solnset ,tiles,gammas[experiment_idx],PLOT_BBG=PLOT_BBG,PLOT_GSOLN=PLOT_GSOLN)
        except(IOError):
            pass
def plot_dual_PR_curves(objid,method="majority_top_k",PLOT_WORKER=False):
    '''
    Plot PR curves for each object for all post-processing methods
    '''
    plt.figure()
    plt.title("Object #{0} [{1}]".format(objid,method))
    # Worker Individual Precision and Recall based on their BB drawn for this object
    worker_precision_lst,worker_recall_lst = compute_worker_PR_obj(objid)
    # Plotting PR from Top-k Majority vote 
    os.chdir("..")
    tiles, objIndicatorMat = createObjIndicatorMatrix(objid,PRINT=False,PLOT=False)
    os.chdir(DATA_DIR)
    precision_lst = []
    recall_lst = []
    if method=='majority_top_k':
    	print int((len(tiles)-1)/30.)
    	print len(tiles)
        param_lst = np.arange(1,len(tiles),max(int((len(tiles)-1)/30.),1))
        for  k in tqdm(param_lst):
            precision,recall= compute_PR_obj(objid,2,majority_topk=k)
            precision_lst.append(precision)
            recall_lst.append(recall)
        plt.xlabel("k")
        plt.xlim(0,len(tiles))
    elif method=="gamma_threshold":
        param_lst = np.linspace(0,0.95,20)
        for  threshold in param_lst :
            precision,recall= compute_PR_obj(objid,2,threshold=threshold)
            precision_lst.append(precision)
            recall_lst.append(recall)
        plt.xlabel("Threshold")
        plt.xlim(0,1)
    elif method=="gamma_top_k":
        param_lst = np.arange(1,len(tiles),max(int((len(tiles)-1)/30.),1))
        for  k in param_lst :
            precision,recall= compute_PR_obj(objid,2,topk=k)
            precision_lst.append(precision)
            recall_lst.append(recall)
        plt.xlabel("k")
        plt.xlim(0,len(tiles))
    recall_lst = np.array(recall_lst)
    precision_lst = np.array(precision_lst)
    plt.plot(param_lst,recall_lst, linestyle='-', linewidth=1,marker='.',color="blue",label="Recall")
    plt.plot(param_lst,precision_lst, linestyle='-', linewidth=1,marker='.',color="red",label="Precision")
        
    if PLOT_WORKER:
        plt.plot(worker_recall_lst , '^',color="gray",label="Worker Recall")
        plt.plot(worker_precision_lst  , 'o',color="gray",label="Worker Precision")
    
    plt.ylim(0,1.05)
    plt.savefig("Dual_PR_{0}_{1}.pdf".format(objid,method))
#     plt.legend(loc="lower right",numpoints=1)