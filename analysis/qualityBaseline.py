import numpy as np
from scipy import spatial
# Given all the x and y annotations for that object, which contains all responses from every worker
# If we want to compute ground truth comparison simply input 
# obj_x_locs = [[worker i response],[ground truth]]
# obj_y_locs = [[worker i response],[ground truth]]

def majority_vote(obj_x_locs,obj_y_locs): 
    '''
    Jaccard Simmilarity or Overlap Method
    used for PASCAL VOC challenge
    ''' 
    return intersection(obj_x_locs,obj_y_locs)/union(obj_x_locs,obj_y_locs)


from munkres import Munkres, print_matrix
def MunkresEuclidean(bb1,bb2):
    '''
    Given two worker's responses, 
    Compares Euclidean distances of all points in the polygon, 
    then find the best matching (min dist) config via Kuhn-Munkres
    '''
    matrix = spatial.distance.cdist(bb1,bb2,'euclidean')
    # print "Mat: " 
    # print np.ma.masked_equal(matrix,0)
    # print np.shape(np.ma.masked_equal(matrix,0))
    m = Munkres()
    try:
        indexes = m.compute(np.ma.masked_equal(matrix,0))
        total = 0
        for row, column in indexes:
            value = matrix[row][column]
            total += value
    #         print '(%d, %d) -> %d' % (row, column, value)
        return total         
    except(ValueError):
        print "bad"
        return 0
def DistAllWorkers(obj_x_locs,obj_y_locs,dist = MunkresEuclidean):
    '''
    Given all worker's responses,
    Perform pairwise distance comparison with all other workers
    returns quality for each worker
    #NOTE THIS NEEDS TO BE CHANGED TO INCORPORATE ALL PAIRWISE COMPARISONS
    '''
    minDistList=[]
    for i in np.arange(len(obj_x_locs)-1):
        # Compare worker with another worker
        bb1 = np.array([obj_x_locs[i],obj_y_locs[i]]).T
        bb2  = np.array([obj_x_locs[i+1],obj_y_locs[i+1]]).T
        # print bb1
        # print bb2
        # print dist(bb1,bb2)
        minDistList.append(dist(bb1,bb2))
    #worker's scores
    return np.array(minDistList)/max(minDistList)

import ast
def process_raw_locs(segmentation,COCO=False):
    '''
    Given a raw string of x and y coordinates, process it
    return a list of x_locs and y_locs
    '''
    x_locs=[]
    y_locs=[]
    if COCO:
        #COCO
#         print "Process COCO"
        poly = np.array(segmentation).reshape((len(segmentation)/2, 2))
        x_locs = list(poly[:,0])
        y_locs = list(poly[:,1])
    else: 
        bbx_path,bby_path = segmentation
        x_locs = [x for x in ast.literal_eval(bbx_path) if x is not None]
        y_locs = [y for y in ast.literal_eval(bby_path) if y is not None]
    # Append the starting point again in the end to close the BB
    x_locs.append(x_locs[0])
    y_locs.append(y_locs[0])
    return x_locs,y_locs
from shapely.geometry import box,Polygon
def intersection(obj_x_locs,obj_y_locs,debug=False):
    # Compute intersecting area
    polygon1 = Polygon(zip(obj_x_locs[0],obj_y_locs[0])).buffer(0)
    polygon2 = Polygon(zip(obj_x_locs[1],obj_y_locs[1])).buffer(0)
    if debug : plt.imshow(polygon1.intersection(polygon2),interpolation="None")
    return polygon1.intersection(polygon2).area
def union(obj_x_locs,obj_y_locs,debug=False):
    # Compute union area of two given polygon 
    polygon1 = Polygon(zip(obj_x_locs[0],obj_y_locs[0])).buffer(0)
    polygon2 = Polygon(zip(obj_x_locs[1],obj_y_locs[1])).buffer(0)
    if debug : plt.imshow(polygon1.union(polygon2),interpolation="None")
    return polygon1.union(polygon2).area
def precision(obj_x_locs,obj_y_locs):
    worker_bb = Polygon(zip(obj_x_locs[0],obj_y_locs[0]))
    worker_bb_area  = worker_bb.area
#     print "Intersection: ", intersection(obj_x_locs,obj_y_locs)
#     print "Worker BB area: ",worker_bb_area
    return intersection(obj_x_locs,obj_y_locs)/float(worker_bb_area)
def recall(obj_x_locs,obj_y_locs):
    truth_bb = Polygon(zip(obj_x_locs[1],obj_y_locs[1]))
    truth_bb_area  = truth_bb.area
    return intersection(obj_x_locs,obj_y_locs)/float(truth_bb_area)    
def simple_rectangle_test():
    # 2 simple overlapping rectangle test example 
    obj_x_locs=[[1,3,3,1],[2,5,5,2]]
    obj_y_locs=[[3,3,1,1],[2,2,0,0]]
    print "Check ``union``: ", union(obj_x_locs,obj_y_locs) == 9
    print "Check ``intersection``: ", intersection(obj_x_locs,obj_y_locs) == 1.
    print "Check ``majority_vote``: ", majority_vote(obj_x_locs,obj_y_locs) == 1./9
    print "Check ``precision``: ", precision(obj_x_locs,obj_y_locs) == 1./4
    print "Check ``recall``: ", recall(obj_x_locs,obj_y_locs) == 1./6

from pycocotools.coco import COCO
from analysis_toolbox import *
def compute_my_COCO_BBvals():
    save_db_as_csv(connect=False)
    img_info,object_tbl,bb_info,hit_info = load_info()
    #Load COCO annotations 
    dataDir='../../coco/'
    dataType='train2014'
    annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)
    # initialize COCO api for instance annotations
    coco=COCO(annFile)
    ground_truth = pd.read_csv("../../data/object_ground_truth.csv")
    worker_info = pd.read_csv("../../data/worker.csv",skipfooter=1)
    my_BBG  = pd.read_csv("my_ground_truth.csv")

    for i in np.arange(len(img_info)):
        img_name = img_info["filename"][i]
        if 'COCO' in img_name:
            img_id = int(img_name.split('_')[-1])
            filtered_object_tbl = object_tbl[object_tbl["image_id"]==i+1]
            annIds = coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anns = coco.loadAnns(annIds)
            #for oid,bbx_path,bby_path in zip(bb_info["object_id"],bb_info["x_locs"],bb_info["y_locs"]):
            for bb in bb_info.iterrows():
                oid = bb[1]["object_id"]
                bbx_path= bb[1]["x_locs"]
                bby_path= bb[1]["y_locs"]
                if int(object_tbl[object_tbl.object_id==oid].image_id) ==i+1:
                    worker_x_locs,worker_y_locs= process_raw_locs([bbx_path,bby_path])
                    ground_truth_match = ground_truth[ground_truth.id==str(oid)]
                    COCO_id = int(ground_truth_match["COCO_annIds"])

                    #COCO-Annotations
                    for ann in anns:
                        if COCO_id==-1:
                            #No BB for this object collected by MSCOCO
                            pass
                        elif ann['id'] == COCO_id: 
    #                         print COCO_id
                            for annBB in ann["segmentation"]:
                                coco_x_locs,coco_y_locs = process_raw_locs(annBB,COCO=True)
                                obj_x_locs = [worker_x_locs,coco_x_locs]
                                obj_y_locs = [worker_y_locs,coco_y_locs]
                                bb_info = bb_info.set_value(bb[0],"Jaccard [COCO]",majority_vote(obj_x_locs,obj_y_locs))
                                bb_info = bb_info.set_value(bb[0],"Precision [COCO]",precision(obj_x_locs,obj_y_locs))
                                bb_info = bb_info.set_value(bb[0],"Recall [COCO]",recall(obj_x_locs,obj_y_locs))                
                                #bb_info = bb_info.set_value(bb[0],"Munkres Euclidean [COCO]",DistAllWorkers(obj_x_locs,obj_y_locs))
                    my_ground_truth_match = my_BBG[my_BBG.object_id==oid]
                    my_x_locs,my_y_locs =  process_raw_locs([my_ground_truth_match["x_locs"].iloc[0],my_ground_truth_match["y_locs"].iloc[0]])
                    obj_x_locs = [worker_x_locs,my_x_locs]
                    obj_y_locs = [worker_y_locs,my_y_locs]
                    bb_info = bb_info.set_value(bb[0],"Jaccard [Self]",majority_vote(obj_x_locs,obj_y_locs))   
                    bb_info = bb_info.set_value(bb[0],"Precision [Self]",precision(obj_x_locs,obj_y_locs))
                    bb_info = bb_info.set_value(bb[0],"Recall [Self]",recall(obj_x_locs,obj_y_locs))
                    #bb_info = bb_info.set_value(bb[0],"Munkres Euclidean [Self]",DistAllWorkers(obj_x_locs,obj_y_locs))
    # replace all NAN values with -1, these are entries for which we don't have COCO ground truth
    bb_info = bb_info.fillna(-1)
    bb_info.to_csv("computed_my_COCO_BBvals.csv")
def visualize_metric_histograms():
    bb_info = pd.read_csv('computed_my_COCO_BBvals.csv')
    metrics_lst = ['Precision [COCO]','Recall [COCO]','Jaccard [COCO]',\
                   'Precision [Self]','Recall [Self]','Jaccard [Self]']

    NUM_COL = 3
    NUM_ROW = 2
    NUM_PLOTS = NUM_COL*NUM_ROW

    fig, axs = plt.subplots(NUM_ROW,NUM_COL, figsize=(NUM_ROW*3,NUM_COL*2), sharex='col')
    #fig.subplots_adjust(hspace = .5, wspace=.001)

    axs = axs.ravel()

    for i,metric in zip(range(len(metrics_lst)),metrics_lst):
        metric_value = np.array(bb_info[metric][bb_info[metric]>0][bb_info[metric]<=1]) 
        ax = axs[i]
        ax.set_title(metric)
        ax.hist(metric_value,bins=100)
        ax.set_xlim(0,1.03)
    fig.tight_layout()
    fig.savefig('metric_histogram.pdf')
if __name__ =="__main__":
    compute_my_COCO_BBvals()
