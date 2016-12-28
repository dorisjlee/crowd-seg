import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
# Given all the x and y annotations for that object, which contains all responses from every worker
# If we want to compute ground truth comparison simply input 
# obj_x_locs = [[worker i response],[ground truth]]
# obj_y_locs = [[worker i response],[ground truth]]

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

################################################
##                                            ##
##       SIMPLE AREA-BASED MEASURES           ##
##                                            ##
################################################
def majority_vote(obj_x_locs,obj_y_locs): 
    '''
    Jaccard Simmilarity or Overlap Method
    used for PASCAL VOC challenge
    ''' 
    return intersection(obj_x_locs,obj_y_locs)/union(obj_x_locs,obj_y_locs)
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
################################################
##                                            ##
##          BOUNDARY-BASED METHODS            ##
##                                            ##
################################################
from scipy.interpolate import splprep,splev
def parametric_interpolate(obj_x_locs,obj_y_locs,numPts=50,PLOT=False):
    '''
    Parametric interpolation of points between polygon boundaries
    Given obj_x_locs,obj_y_locs
    return a new set of interpolated obj_x_locs,obj_y_locs interpolated on numPts 
    '''
    interpolated_obj_x_locs = []
    interpolated_obj_y_locs = []
    for i in range(len(obj_x_locs)):
        x = obj_x_locs[i]
        y = obj_y_locs[i]
        # if numPts==0: numPts=len(x)*2
        if len(x)<numPts:
            tck, u =splprep(np.array([x,y]),s=0,per=1)
            u_new = np.linspace(u.min(),u.max(),numPts)
            new_points = splev(u_new, tck,der=0)
            if PLOT: 
                plt.figure()
                plt.plot(x, y, 'ro',label="N="+str(numPts))
                plt.legend(loc="lower right")
                plt.plot(new_points[0], new_points[1], 'r-')
            interpolated_obj_x_locs.append(new_points[0])
            interpolated_obj_y_locs.append(new_points[1])
        else:
            #interpolate only if number of points is less than numPts 
            interpolated_obj_x_locs.append(obj_x_locs[i])
            interpolated_obj_y_locs.append(obj_y_locs[i])
    return interpolated_obj_x_locs,interpolated_obj_y_locs

from scipy.spatial.distance import cdist,pdist
from munkres import Munkres, print_matrix
def MunkresEuclidean(obj_x_locs,obj_y_locs,numPts=50,PRINT=False):
    '''
    Given two worker's responses, 
    Compares Euclidean distances of all points in the polygon, 
    then find the best matching (min dist) config via Kuhn-Munkres
    '''
    if obj_x_locs[0]==obj_x_locs[1] and obj_y_locs[0]==obj_y_locs[1] :
        return 0
    interpolated_obj_x_locs,interpolated_obj_y_locs = parametric_interpolate(obj_x_locs,obj_y_locs,numPts=numPts,PLOT=PRINT)
    polygon1 = zip(interpolated_obj_x_locs[0],interpolated_obj_y_locs[0])
    polygon2 = zip(interpolated_obj_x_locs[1],interpolated_obj_y_locs[1])

    matrix = spatial.distance.cdist(polygon1,polygon2,'euclidean')
    
    m = Munkres()
    try:
        indexes = m.compute(np.ma.masked_equal(matrix,0))
        total = 0
        for row, column in indexes:
            value = matrix[row][column]
            total += value
            if PRINT: print '(%d, %d) -> %.2f' % (row, column, value)
        return total         
    except(ValueError):
        print "bad"
        return 0
def DistAllWorkers(obj_x_locs,obj_y_locs,dist = MunkresEuclidean,MAX_DIST=10000.,numPts = 50):
    '''
    Given all worker's responses,
    Perform pairwise distance comparison with all other workers
    returns quality for each worker
    '''
    # interpolated_obj_x_locs,interpolated_obj_y_locs = parametric_interpolate(obj_x_locs,obj_y_locs,numPts)
    # # Compare worker with ground truth worker
    # bb1 = zip(interpolated_obj_x_locs[i],interpolated_obj_y_locs[i])
    # bb2 = zip(interpolated_obj_x_locs[i+1],interpolated_obj_y_locs[i+1])

    #worker's scores
    return 1.-dist(obj_x_locs,obj_y_locs)/MAX_DIST


################################################
##                                            ##
##                 TEST EXAMPLES              ##
##                                            ##
################################################
def real_BB_test():
    print "-----------------"
    print "Real BB Test"

    bb_info = pd.read_csv('computed_my_COCO_BBvals.csv')
    bbg_info = pd.read_csv('my_ground_truth.csv')
    oid = 8
    worker =  bb_info[bb_info["object_id"]==oid].ix[73]
    bbg = bbg_info[bb_info["object_id"]==oid].ix[0]
    obj_x_locs = [ast.literal_eval(worker["x_locs"]),ast.literal_eval(bbg["x_locs"])]
    obj_y_locs = [ast.literal_eval(worker["y_locs"]),ast.literal_eval(bbg["y_locs"])]
    
    print "Check ``majority_vote``: ",np.isclose(majority_vote(obj_x_locs,obj_y_locs),0.621613376145)
    print "Check ``precision``: ",np.isclose(precision(obj_x_locs,obj_y_locs),0.954158717771)
    print "Check ``recall``: ",np.isclose(recall(obj_x_locs,obj_y_locs),0.640749081612)
    print "Check ``MunkresEuclidean``:",np.isclose(MunkresEuclidean(obj_x_locs,obj_y_locs,PRINT=True),946.390562322)

def simple_rectangle_test():
    print "-----------------"
    print "Simple Rectangle Test"
    # 2 simple overlapping rectangle test example 
    obj_x_locs=[[1,3,3,1],[2,5,5,2]]
    obj_y_locs=[[3,3,1,1],[2,2,0,0]]
    print "Check ``union``: ", union(obj_x_locs,obj_y_locs) == 9
    print "Check ``intersection``: ", intersection(obj_x_locs,obj_y_locs) == 1.
    print "Check ``majority_vote``: ", majority_vote(obj_x_locs,obj_y_locs) == 1./9
    print "Check ``precision``: ", precision(obj_x_locs,obj_y_locs) == 1./4
    print "Check ``recall``: ", recall(obj_x_locs,obj_y_locs) == 1./6
    print "Check ``MunkresEuclidean [No Interpolation]``:",np.isclose(MunkresEuclidean(obj_x_locs,obj_y_locs,numPts=4,PRINT=True),2*(np.sqrt(2)+np.sqrt(5)))
    print "Check ``MunkresEuclidean [N=50]``:",np.isclose(MunkresEuclidean(obj_x_locs,obj_y_locs,PRINT=True),87.9127740436)

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
                    worker_x_locs,worker_y_locs = zip(*list(OrderedDict.fromkeys(zip(worker_x_locs,worker_y_locs))))
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
                                #Remove duplicates
                                coco_x_locs,coco_y_locs = zip(*list(OrderedDict.fromkeys(zip(coco_x_locs,coco_y_locs))))
                                obj_x_locs = [list(worker_x_locs),list(coco_x_locs)]
                                obj_y_locs = [list(worker_y_locs),list(coco_y_locs)]
                                bb_info = bb_info.set_value(bb[0],"Jaccard [COCO]",majority_vote(obj_x_locs,obj_y_locs))
                                bb_info = bb_info.set_value(bb[0],"Precision [COCO]",precision(obj_x_locs,obj_y_locs))
                                bb_info = bb_info.set_value(bb[0],"Recall [COCO]",recall(obj_x_locs,obj_y_locs))    
                                bb_info = bb_info.set_value(bb[0],"Munkres Euclidean [COCO]",MunkresEuclidean(obj_x_locs,obj_y_locs))
                    my_ground_truth_match = my_BBG[my_BBG.object_id==oid]
                    my_x_locs,my_y_locs =  process_raw_locs([my_ground_truth_match["x_locs"].iloc[0],my_ground_truth_match["y_locs"].iloc[0]])
                    my_x_locs,my_y_locs = zip(*list(OrderedDict.fromkeys(zip(my_x_locs,my_y_locs))))
                    obj_x_locs = [list(worker_x_locs),list(my_x_locs)]
                    obj_y_locs = [list(worker_y_locs),list(my_y_locs)]
                    bb_info = bb_info.set_value(bb[0],"Jaccard [Self]",majority_vote(obj_x_locs,obj_y_locs))   
                    bb_info = bb_info.set_value(bb[0],"Precision [Self]",precision(obj_x_locs,obj_y_locs))
                    bb_info = bb_info.set_value(bb[0],"Recall [Self]",recall(obj_x_locs,obj_y_locs))
                    bb_info = bb_info.set_value(bb[0],"Munkres Euclidean [Self]",MunkresEuclidean(obj_x_locs,obj_y_locs))
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
    import sys
    if len(sys.argv)>1:
        if sys.argv[1]=='test':
            simple_rectangle_test()
            real_BB_test()
        elif sys.argv[1]=='compute':
            compute_my_COCO_BBvals()
    else:    
        print "Usage: python qualityBaseline.py test/compute"
        compute_my_COCO_BBvals()
