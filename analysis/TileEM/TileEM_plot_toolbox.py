# from dataset import Dataset
# from greedy import *
# from data import *
# from experiment import *
from BB2TileExact import *
import pandas as pd
import pickle as pkl
import shapely
from shapely.ops import cascaded_union
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import warnings
DEBUG=False
warnings.filterwarnings('ignore')
NParam=10.
my_BBG  = pd.read_csv("../my_ground_truth.csv")
object_lst = list(object_tbl.id)
def compute_PR(objid,solnset,tiles):
    '''
    Compute precision recall against ground truth bounding box
    for a given solution set and tile coordinates.
    '''
    if len(solnset)==1:
        joined_bb=tiles[solnset]
        problematic_tiles=[]
    else:
        try:
            joined_bb,problematic_tiles = join_tiles(solnset,tiles)
        except(ValueError):
            return -1,-1
    
    ground_truth_match = my_BBG[my_BBG.object_id==objid]
    x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
    BBG = shapely.geometry.Polygon(zip(x_locs,y_locs))
    if problematic_tiles!=[]:
        intersect_area =0
        joined_bb_area =0
        if type(joined_bb)==Polygon:
            joined_bb_iter=[joined_bb]
        else:
            joined_bb_iter=joined_bb
        for jbb in joined_bb_iter:
            ia = intersection_area(BBG,jbb)
            intersect_area += ia
            joined_bb_area += jbb.area
    else:
	try:
            intersect_area=intersection_area(BBG,joined_bb)
	except(shapely.geos.TopologicalError):
	    return -1,-1
        joined_bb_area =joined_bb.area
    if intersect_area ==-1:
	return -1,-1
    if float(joined_bb_area)!=0:
        precision = intersect_area/float(joined_bb_area)
    else:
        # Empty solution set
        precision = -1
    recall = intersect_area/BBG.area
    #This patches up PR>1 because intersect_area using the slow intersection method can sometimes be larger since overlap area may not be completely deducted, so the estimate is a bit larger than the actual area, but not by much 
    if recall>1:
        recall=1
    if precision>1:
        precision=1
    return precision,recall
def intersection_area(poly1,poly2):
    try:
        return poly1.intersection(poly2).area
    except(shapely.geos.TopologicalError):
	#return poly1.buffer(1e-5).intersection(poly2.buffer(1e-5)).area
	try: 
            return poly1.buffer(1e-5).intersection(poly2.buffer(1e-5)).area
	except(shapely.geos.TopologicalError):
	    try:
	    	return poly1.buffer(-1e-10).intersection(poly2.buffer(-1e-10)).area
	    except(shapely.geos.TopologicalError):
		return -1 

	
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
			solutionList.append(i)
	return solutionList

def getSolutionTopK(data,k=5):
	#Derive a solution set of tiles of top-K gamma tile values 
	return np.argsort(data)[::-1][:k]
def compute_worker_lst_jaccard_obj(objid,worker_lst,EXCLUDE_BBG=True):
    # List of Jaccard Index (IOU) of given list of workers
    jaccard_lst = []

    objBBs = bb_info[bb_info.object_id==objid]
    ground_truth_match = my_BBG[my_BBG.object_id==objid]
    BBG_x_locs,BBG_y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
    for worker_id in worker_lst:
        bb = objBBs[objBBs["worker_id"]==worker_id]
        oid = bb["object_id"]
        bbx_path= bb["x_locs"].values[0]
        bby_path= bb["y_locs"].values[0]
        worker_x_locs,worker_y_locs= process_raw_locs([bbx_path,bby_path])
        jaccard = intersection([worker_x_locs,BBG_x_locs],[worker_y_locs,BBG_y_locs])/union([worker_x_locs,BBG_x_locs],[worker_y_locs,BBG_y_locs])
        jaccard_lst.append(jaccard)
    return jaccard_lst

def compute_worker_lst_PR_obj(objid,worker_lst,EXCLUDE_BBG=True):
    # List of PR measures of given lsit of worker ids
    precision_lst = []
    recall_lst = []

    objBBs = bb_info[bb_info.object_id==objid]
    ground_truth_match = my_BBG[my_BBG.object_id==objid]
    BBG_x_locs,BBG_y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
    for worker_id in worker_lst:
        bb = objBBs[objBBs["worker_id"]==worker_id]
        oid = bb["object_id"]
        bbx_path= bb["x_locs"].values[0]
        bby_path= bb["y_locs"].values[0]
        worker_x_locs,worker_y_locs= process_raw_locs([bbx_path,bby_path])
        precision_lst.append(precision([worker_x_locs,BBG_x_locs],[worker_y_locs,BBG_y_locs]))
        recall_lst.append(recall([worker_x_locs,BBG_x_locs],[worker_y_locs,BBG_y_locs]))
    return np.array(precision_lst),np.array(recall_lst)
def compute_worker_PR_obj(objid,return_worker_id=False,EXCLUDE_BBG=True):
    # List of PR measures of all workers 
    precision_lst = []
    recall_lst = []
    worker_lst=[]

    objBBs = bb_info[bb_info.object_id==objid]
    if EXCLUDE_BBG: objBBs=  objBBs[objBBs.worker_id!=3]
    ground_truth_match = my_BBG[my_BBG.object_id==objid]
    BBG_x_locs,BBG_y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
    for bb in objBBs.iterrows():
        oid = bb[1]["object_id"]
        bbx_path= bb[1]["x_locs"]
        bby_path= bb[1]["y_locs"]
        worker_x_locs,worker_y_locs= process_raw_locs([bbx_path,bby_path])
        worker_lst.append(bb[1]["worker_id"])
        precision_lst.append(precision([worker_x_locs,BBG_x_locs],[worker_y_locs,BBG_y_locs]))
        recall_lst.append(recall([worker_x_locs,BBG_x_locs],[worker_y_locs,BBG_y_locs]))
    if return_worker_id:
        return np.array(worker_lst),np.array(precision_lst),np.array(recall_lst)
    else:
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
    If PLOT_HEATMAP="all-region", plot all tile heatmap region whether selected from solnset or not
    '''
    INCLUDE_ALL=False
    if PLOT_HEATMAP=="all-region": INCLUDE_ALL=True
    
    try:
        tiles = pkl.load(open("../{0}/tiles{1}.pkl".format(DATA_DIR,objid),'r'))
        gammas = pkl.load(open("gfile{}.pkl".format(objid),'r'))#ga,gm,gl,ge
        # Deriving new solution set from different thresholding criteria
        if threshold!=-1:
            procstr = "Gamma>{}".format(threshold)
            solnset = getSolutionThreshold(gammas[experiment_idx],threshold=threshold) 
            print "Nsolnset: ", len(solnset)
            if PLOT_HEATMAP: mask = plot_tile_heatmap(objid,solnset ,tiles,gammas[experiment_idx],PLOT_BBG=True,PLOT_GSOLN=True,INCLUDE_ALL=INCLUDE_ALL)
        elif topk!=-1:
            procstr="Gamma k={}".format(topk)
            solnset = getSolutionTopK(gammas[experiment_idx],k=topk)
            if PLOT_HEATMAP: mask = plot_tile_heatmap(objid,solnset ,tiles,gammas[experiment_idx],PLOT_BBG=True,PLOT_GSOLN=True,INCLUDE_ALL=INCLUDE_ALL)
            if DEBUG:
	        	print procstr
	        	print solnset
	        	print np.array(gammas[experiment_idx])[solnset]
        elif majority_topk!=-1:
            objIndicatorMat = pkl.load(open("../{0}/indMat{1}.pkl".format(DATA_DIR,objid),'r'))
            print objIndicatorMat
            tile_votes = np.sum(objIndicatorMat[:-1],axis=0)
            procstr="Majority k={}".format(majority_topk)
            solnset = getSolutionTopK(tile_votes,k=majority_topk)
            if PLOT_HEATMAP: 
                mask = plot_tile_heatmap(objid,solnset ,tiles,tile_votes,PLOT_BBG=True,PLOT_GSOLN=True,INCLUDE_ALL=INCLUDE_ALL)
        
        precision,recall = compute_PR(objid,solnset,tiles)
        if PLOT_HEATMAP:
	        font0 = matplotlib.font_manager.FontProperties()
	        font1=font0.copy()
	        font1.set_size('large')
	        font1.set_weight('heavy')
	        plt.figtext(0.60,0.75,'{0}\n P={1:.2f}, R={2:.2f}'.format(procstr,precision,recall),color='black',fontproperties=font1,ha='center')
	        # font1.set_size('xx-large')
	        # font1.set_weight('heavy')
	        # plt.figtext(0.45,0.7,'{0}\n P={1:.2f}, R={2:.2f}'.format(procstr,precision,recall),color='white',fontproperties=font1,ha='center')
    except(IOError):
        print "IOERROR"
        pass
    print precision,recall
    return precision,recall


def plot_all_postprocess_PR_curves(objid,experiment_idx=0,legend=False):
    '''
	Plot PR curves for each object for all post-processing methods
	'''
    print "Working on object: ",objid
    plt.figure()
    plt.title("Object #{}".format(objid))
    # Worker Individual Precision and Recall based on their BB drawn for this object
    worker_precision_lst,worker_recall_lst = compute_worker_PR_obj(objid)
    plt.plot(worker_recall_lst ,worker_precision_lst , '.',color="gray",label="Worker")
    # Plotting PR from Top-k Majority vote 
    print "Top k Majority Vote"
    tiles = pkl.load(open("../{0}/tiles{1}.pkl".format(DATA_DIR,objid),'r'))
    objIndicatorMat = pkl.load(open("../{0}/indMat{1}.pkl".format(DATA_DIR,objid),'r'))
    #k_lst = np.arange(1,len(tiles),max(int((len(tiles)-1)/NParam),1))
    k_lst = [1,5,10,15,20,25,30]
    Maj_topk_precision_lst = []
    Maj_topk_recall_lst = []
    for  k in k_lst :
        print "k = ",k
        Maj_topk_precision,Maj_topk_recall= compute_PR_obj(objid,experiment_idx,topk=k)
        Maj_topk_precision_lst.append(Maj_topk_precision)
        Maj_topk_recall_lst.append(Maj_topk_recall)
    Maj_topk_recall_lst = np.array(Maj_topk_recall_lst)
    Maj_topk_precision_lst = np.array(Maj_topk_precision_lst)
    order = np.argsort(Maj_topk_recall_lst)
    plt.plot(Maj_topk_recall_lst[order],Maj_topk_precision_lst[order], linestyle='-', linewidth=1,marker='D',color="cyan",label="Majority Top-k")

    # Plotting PR from TileEM for different thresholds
    #threshold_lst = np.linspace(0,1,NParam)
    print "Gamma threshold"
    threshold_lst = np.linspace(0.95,1,5)
    TileEM_thres_precision_lst = []
    TileEM_thres_recall_lst = []
    for threshold in threshold_lst :
        print "threshold:", threshold
        TileEM_thres_precision, TileEM_thres_recall= compute_PR_obj(objid,threshold=threshold)
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
    print "Gamma top-k"
    for  k in k_lst :
        print "k : ",k
        TileEM_topk_precision, TileEM_topk_recall= compute_PR_obj(objid,topk=k)
        TileEM_topk_precision_lst.append(TileEM_topk_precision)
        TileEM_topk_recall_lst.append(TileEM_topk_recall)
    TileEM_topk_recall_lst = np.array(TileEM_topk_recall_lst)
    TileEM_topk_precision_lst = np.array(TileEM_topk_precision_lst)
    order = np.argsort(TileEM_topk_recall_lst)
    plt.plot(TileEM_topk_recall_lst[order],TileEM_topk_precision_lst[order], linestyle='--', linewidth=3,marker='o',color="green",label="TileEM Top-k")

    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.ylabel("Precision",fontsize=13)
    plt.xlabel("Recall",fontsize=13)
    if legend: plt.legend(loc="bottom left",numpoints=1)
    plt.savefig("PR_obj{}.pdf".format(objid))

    # Save Data
    
    pkl.dump(Maj_topk_recall_lst,open("Maj_topk_recall_lst_obj{}.pkl".format(objid),'w'))
    pkl.dump(Maj_topk_precision_lst,open("Maj_topk_precision_lst_obj{}.pkl".format(objid),'w'))

    pkl.dump(TileEM_thres_recall_lst,open("TileEM_thres_recall_lst_obj{}.pkl".format(objid),'w'))
    pkl.dump(TileEM_thres_precision_lst,open("TileEM_thres_precision_lst_obj{}.pkl".format(objid),'w'))

    pkl.dump(TileEM_topk_recall_lst,open("TileEM_topk_recall_lst_obj{}.pkl".format(objid),'w'))
    pkl.dump(TileEM_topk_precision_lst,open("TileEM_topk_recall_lst_obj{}.pkl".format(objid),'w'))
def compute_joined_PR(objid):
    '''
	Compute the PR values for all post-processing method
	'''
    # Worker Individual Precision and Recall based on their BB drawn for this object
    worker_precision_lst,worker_recall_lst = compute_worker_PR_obj(objid)
    # Plotting PR from Top-k Majority vote 
    tiles = pkl.load(open("../{0}/tiles{1}.pkl".format(DATA_DIR,objid),'r'))
    objIndicatorMat = pkl.load(open("../{0}/indMat{1}.pkl".format(DATA_DIR,objid),'r'))
    k_lst = np.arange(1,len(tiles),max(int((len(tiles)-1)/NParam),1))
    Maj_topk_precision_lst = []
    Maj_topk_recall_lst = []
    for  k in k_lst :
        Maj_topk_precision,Maj_topk_recall= compute_PR_obj(objid,2,topk=k)
        Maj_topk_precision_lst.append(Maj_topk_precision)
        Maj_topk_recall_lst.append(Maj_topk_recall)
    Maj_topk_recall_lst = np.array(Maj_topk_recall_lst)
    Maj_topk_precision_lst = np.array(Maj_topk_precision_lst)
    # Plotting PR from TileEM for different thresholds
    threshold_lst = np.linspace(0,1,NParam)
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
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
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
    try:
        return cascaded_union([tiles[tidx] for tidx in solutionList]),[]
    except:
        #slow version, run through and exclude problematic solutionset item
        Utile=tiles[0]
        problematic_tiles =[]
        for soln in solutionList:
            try:
                Utile=Utile.union(tiles[soln])
            except(shapely.geos.TopologicalError):
                problematic_tiles.append(tiles[soln])
        problematic_tiles.append(Utile)
        return Utile,problematic_tiles
#mask = plot_tile_heatmap(objid,solnset ,tiles,tile_votes,PLOT_BBG=True,PLOT_GSOLN=True)
def plot_tile_heatmap(objid,solnset,tiles,z_values,PLOT_BBG=False,PLOT_GSOLN=False,INCLUDE_ALL=False):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon
    '''
	Plot Tile Heatmap for the object z_values (can either be gamma values or majority votes)
	Optional: include ground truth (BBG) or Gamma/Majority vote tile solution (GSOLN)
	INCLUDE_ALL: plot  heatmap values for all tiles, otherwise, plot only the tiles selected by solnset
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
            plot_coords(ML_regions,color="lime",reverse_xy=True)
        else:
            for region in ML_regions:
                plot_coords(region,color="lime",reverse_xy=True)
    if PLOT_BBG:
        my_BBG  = pd.read_csv("../../my_ground_truth.csv")
        ground_truth_match = my_BBG[my_BBG.object_id==objid]
        x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
        plt.plot(x_locs,y_locs,'--',color='#0000ff',linewidth=1.5)

    # Plot only z_values of tiles included in the solnset
    patches = []

    if INCLUDE_ALL:
    	solnset=range(len(tiles))	
    for tile_idx in  solnset:
        #tile =  sorted_ascend_tile_by_size[tile_idx]
        tile=tiles[tile_idx]
        from shapely.geometry import mapping
        polygon = Polygon(mapping(tile)["coordinates"][0],closed=True,\
                          linewidth=1,edgecolor='black',fill=False)
        patches.append(polygon)

    collection = PatchCollection(patches,cmap=matplotlib.cm.autumn_r,alpha=0.25)
    pcollection = ax.add_collection(collection)
    
    if max(z_values)>2 : #majority vote
    	cbar_range = np.arange(0,max(z_values))
    else: 
    	cbar_range = np.linspace(0,1,21)
    if INCLUDE_ALL:
    	selected_z_vals=np.array(z_values)
    else:
	    selected_z_vals=np.array(z_values[solnset])
    collection.set_array(selected_z_vals)
    ax.add_collection(collection)
    plt.colorbar(collection)#,boundaries=cbar_range)
    img_name = img_info[img_info.id==int(object_tbl[object_tbl.id==objid]["image_id"])]["filename"].iloc[0]
    fname = "../../../web-app/app/static/"+img_name+".png"
    plt.title("Object {0} [{1}]".format(objid,object_tbl[object_tbl.object_id==objid]["name"].iloc[0]))
    img =mpimg.imread(fname)
    width,height = get_size(fname)
    plt.xlim(0,width)
    plt.ylim(height,0) 
    collection.set_clim(0,max(z_values))
    plt.imshow(img,alpha=0.8)
    ax.autoscale_view()
def plot_all_T_search_PR_curves(objid,postprocess='majority-top-k'):
    '''
    Plot PR curves for each object for all T-search methods
    '''
    print "Plot all T search PR with postprocess = ",postprocess
    experiment_names = {0:'Avrg',1:'Median',2:'Local',3:'Exhaustive'}

    plt.figure()
    plt.title("Object #{0} [{1}]".format(objid,postprocess))
    # Worker Individual Precision and Recall based on their BB drawn for this object
    worker_precision_lst,worker_recall_lst = compute_worker_PR_obj(objid)
    plt.plot(worker_recall_lst ,worker_precision_lst , '.',color='gray',label="Worker")
    
    tiles = pkl.load(open("../{0}/tiles{1}.pkl".format(DATA_DIR,objid),'r'))
    objIndicatorMat = pkl.load(open("../{0}/indMat{1}.pkl".format(DATA_DIR,objid),'r'))
    
    k_lst = np.arange(1,len(tiles),max(int((len(tiles)-1)/NParam),1))
    linestyles = ['--','-','-.','--']
    markers=['^','D','o','>']
    for experiment_idx in [2,3,0,1]:
        if postprocess == 'majority-top-k':
            # Plotting PR from Top-k Majority vote 
            Maj_topk_precision_lst = []
            Maj_topk_recall_lst = []
            for  k in k_lst :
                Maj_topk_precision,Maj_topk_recall= compute_PR_obj(objid,experiment_idx,topk=k)
                Maj_topk_precision_lst.append(Maj_topk_precision)
                Maj_topk_recall_lst.append(Maj_topk_recall)
            Maj_topk_recall_lst = np.array(Maj_topk_recall_lst)
            Maj_topk_precision_lst = np.array(Maj_topk_precision_lst)
            order = np.argsort(Maj_topk_recall_lst)
            plt.plot(Maj_topk_recall_lst[order],Maj_topk_precision_lst[order], linestyle=linestyles[experiment_idx], linewidth=experiment_idx+1, marker=markers[experiment_idx], label=experiment_names[experiment_idx])

            pkl.dump(Maj_topk_recall_lst,open("exp{0}_Maj_topk_recall_lst_obj{1}.pkl".format(experiment_idx,objid),'w'))
            pkl.dump(Maj_topk_precision_lst,open("exp{0}_Maj_topk_precision_lst_obj{1}.pkl".format(experiment_idx,objid),'w'))
        elif postprocess == 'tile-threshold':
            # Plotting PR from TileEM for different thresholds
            threshold_lst = np.linspace(0,1,NParam)
            TileEM_thres_precision_lst = []
            TileEM_thres_recall_lst = []
            for threshold in threshold_lst :
                TileEM_thres_precision, TileEM_thres_recall= compute_PR_obj(objid,experiment_idx,threshold=threshold)
                TileEM_thres_precision_lst.append(TileEM_thres_precision)
                TileEM_thres_recall_lst.append(TileEM_thres_recall)
            TileEM_thres_recall_lst = np.array(TileEM_thres_recall_lst)
            TileEM_thres_precision_lst = np.array(TileEM_thres_precision_lst)
            #     print "{0}:{1},{2}".format(threshold,TileEM_thres_precision,TileEM_thres_recall)
            order = np.argsort(TileEM_thres_recall_lst)
            plt.plot(TileEM_thres_recall_lst[order],TileEM_thres_precision_lst[order], linestyle=linestyles[experiment_idx],linewidth=experiment_idx+1, marker=markers[experiment_idx], label=experiment_names[experiment_idx])
            pkl.dump(TileEM_thres_recall_lst,open("exp{0}_TileEM_thres_recall_lst_obj{1}.pkl".format(experiment_idx,objid),'w'))
            pkl.dump(TileEM_thres_precision_lst,open("exp{0}_TileEM_thres_precision_lst_obj{1}.pkl".format(experiment_idx,objid),'w'))
        elif postprocess=='tile-top-k':
            # Plotting PR from TileEM for different Top-k
            TileEM_topk_precision_lst = []
            TileEM_topk_recall_lst = []
            for  k in k_lst :
                TileEM_topk_precision, TileEM_topk_recall= compute_PR_obj(objid,experiment_idx,topk=k)
                TileEM_topk_precision_lst.append(TileEM_topk_precision)
                TileEM_topk_recall_lst.append(TileEM_topk_recall)
            TileEM_topk_recall_lst = np.array(TileEM_topk_recall_lst)
            TileEM_topk_precision_lst = np.array(TileEM_topk_precision_lst)
            order = np.argsort(TileEM_topk_recall_lst)
            plt.plot(TileEM_topk_recall_lst[order],TileEM_topk_precision_lst[order], linestyle=linestyles[experiment_idx], linewidth=experiment_idx+1,marker=markers[experiment_idx], label=experiment_names[experiment_idx])
            pkl.dump(TileEM_topk_recall_lst,open("exp{0}_TileEM_topk_recall_lst_obj{1}.pkl".format(experiment_idx,objid),'w'))
            pkl.dump(TileEM_topk_precision_lst,open("exp{0}_TileEM_topk_precision_lst_obj{1}.pkl".format(experiment_idx,objid),'w'))
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.ylabel("Precision",fontsize=13)
    plt.xlabel("Recall",fontsize=13)
    lgd = plt.legend( numpoints=1, loc="center right", bbox_to_anchor=(1.4, 0.5))
    plt.savefig("PR_obj{0}_{1}.pdf".format(objid,postprocess), bbox_extra_artists=(lgd,),bbox_inches='tight')


    
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
def plot_dual_PR_curves(objid,method="majority_top_k",PLOT_WORKER=False,legend=False):
    '''
    Plot PR curves for each object for all post-processing methods
    '''
    plt.figure()
    plt.title("Object #{0} [{1}]".format(objid,method))
    # Worker Individual Precision and Recall based on their BB drawn for this object
    worker_precision_lst,worker_recall_lst = compute_worker_PR_obj(objid)
    # Plotting PR from Top-k Majority vote 
    tiles = pkl.load(open("../{0}/tiles{1}.pkl".format(DATA_DIR,objid),'r'))
    objIndicatorMat = pkl.load(open("../{0}/indMat{1}.pkl".format(DATA_DIR,objid),'r'))
    precision_lst = []
    recall_lst = []
    if method=='majority_top_k':
        param_lst = np.arange(1,len(tiles),max(int((len(tiles)-1)/NParam),1))
        for  k in tqdm(param_lst):
            precision,recall= compute_PR_obj(objid,majority_topk=k)
            precision_lst.append(precision)
            recall_lst.append(recall)
        plt.xlabel("k")
        plt.xlim(-0.01,len(tiles))
    elif method=="gamma_threshold":
        param_lst = np.linspace(0,1,NParam)
        for  threshold in param_lst :
            precision,recall= compute_PR_obj(objid,threshold=threshold)
            precision_lst.append(precision)
            recall_lst.append(recall)
        plt.xlabel("Threshold")
        plt.xlim(-0.01,1)
    elif method=="gamma_top_k":
        param_lst = np.arange(1,len(tiles),max(int((len(tiles)-1)/NParam),1))
        for  k in param_lst :
            precision,recall= compute_PR_obj(objid,2,topk=k)
            precision_lst.append(precision)
            recall_lst.append(recall)
        plt.xlabel("k")
        plt.xlim(-0.01,len(tiles))
    recall_lst = np.array(recall_lst)
    precision_lst = np.array(precision_lst)
    plt.plot(param_lst,recall_lst, linestyle='-', linewidth=1,marker='.',color="blue",label="Recall")
    plt.plot(param_lst,precision_lst, linestyle='-', linewidth=1,marker='.',color="red",label="Precision")

    if PLOT_WORKER:
        plt.plot(np.zeros_like(worker_recall_lst),worker_recall_lst , '^',ms=5,color="blue",label="Worker Recall")
        plt.plot(np.zeros_like(worker_precision_lst), worker_precision_lst  , 'o',ms=5,color="red",label="Worker Precision")
    
    plt.ylim(0,1.05)
    if legend: plt.legend(loc="lower left",numpoints=1)
    plt.savefig("Dual_PR_{0}_{1}.pdf".format(objid,method))
def PR_compare(objid,sampleNworkers=40):    
    os.chdir("..")
    # worker_lst,tiles,indicatorMat= createObjIndicatorMatrix(objid,PRINT=True,sampleNworkers=sampleNworkers,tqdm_on=False,tile_only=False)
    worker_lst = pkl.load(open("{0}/worker{1}.pkl".format(DATA_DIR,objid),'r'))
    tiles = pkl.load(open("{0}/tiles{1}.pkl".format(DATA_DIR,objid),'r'))
    objIndicatorMat = pkl.load(open("{0}/indMat{1}.pkl".format(DATA_DIR,objid),'r'))
    os.chdir(DATA_DIR)
    
    worker_precision_lst,worker_recall_lst = compute_worker_lst_PR_obj(objid,worker_lst)
    best_worker =  np.argmax(worker_recall_lst)
    print "Best worker's PR against BBG: ", max(worker_precision_lst),max(worker_recall_lst)

    approved_tiles = np.where(indicatorMat[best_worker]==1)[0]
    for tidx in approved_tiles:
        plot_coords(tiles[tidx],color="lime")

    bb_objects = bb_info[bb_info["object_id"]==objid]
    bb_objects =  bb_objects[bb_objects.worker_id!=3]
    best_worker_id = worker_lst[best_worker]
    print best_worker_id
    worker_bb_info = bb_objects[bb_objects["worker_id"]==best_worker_id]
    worker_BB_polygon = Polygon(zip(*process_raw_locs([worker_bb_info["x_locs"].values[0],worker_bb_info["y_locs"].values[0]])))#.buffer(0)

    plot_coords(worker_BB_polygon,linestyle='--',color='#0000ff')
    
    joined_bb = join_tiles(approved_tiles,tiles)
    intersect_area = worker_BB_polygon.intersection(joined_bb).area
    precision = intersect_area/joined_bb.area
    recall = intersect_area/worker_BB_polygon.area
    print precision,recall
    return precision,recall
