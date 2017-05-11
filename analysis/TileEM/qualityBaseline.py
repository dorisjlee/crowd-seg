import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import time
from tqdm import tqdm 
import ast
import pickle as pkl
# Given all the x and y annotations for that object, which contains all responses from every worker
# If we want to compute ground truth comparison simply input 
# obj_x_locs = [[worker i response],[ground truth]]
# obj_y_locs = [[worker i response],[ground truth]]


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
def compute_img_to_bb_area_ratio(img_name,worker_x_locs,worker_y_locs):
    '''
    Percentage of image area occupied by user's BB, proposed by Vittyakorn et al as a baseline
    '''
    fname = "../web-app/app/static/"+img_name+".png"
    width,height = get_size(fname)
    img_area = width*height
    bb_poly = Polygon(zip(worker_x_locs,worker_y_locs))
    return bb_poly.area/float(img_area)
#def majority_vote(obj_x_locs,obj_y_locs): 
#    '''
#    Jaccard Simmilarity or Overlap Method
#    used for PASCAL VOC challenge
#    ''' 
#    return intersection(obj_x_locs,obj_y_locs)/union(obj_x_locs,obj_y_locs)
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

#from scipy.spatial.distance import cdist,pdist
#from munkres import Munkres, print_matrix
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
        # When there are only two points
        print "ValueError"
        return -1
    except(IndexError):
        print "IndexError"
        return -1
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
##       CANNY-EDGE BASED METRIC              ##
##                                            ##
################################################
from matplotlib import cm
def plotContour(img_name,contour_lst,title="",ctype='skimage'):
    plt.title(title)
    img = plt.imread(img_name)
    plt.gca().invert_yaxis()
    plt.imshow(img,cmap= cm.Greys)
    for c in contour_lst:
        if ctype=='skimage':
            x,y = zip(*c)
        elif ctype =='opencv' :
            x,y = zip(*c[:,0])
        plt.plot(x,y,color='cyan',linewidth=0.5)
    dim = np.shape(img)
    plt.xlim(0,dim[1])
    plt.ylim(dim[0],0)
    plt.savefig(title+".pdf")
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

def get_size(fname):
    from PIL import Image
    #Open image for computing width and height of image 
    im = Image.open(fname)
    width = im.size[0]
    height = im.size[1]
    return width, height

def compute_my_COCO_BBvals(compute_metrics=['simple','area','dist']):
    compute_metrics=['Precision [COCO]','Recall [COCO]','Jaccard [COCO]',"NME [COCO]","Num Points", 'Precision [Self]','Recall [Self]','Jaccard [Self]',"NME [Self]","Area Ratio"]
    from analysis_toolbox import *
    from pycocotools.coco import COCO
    from collections import OrderedDict
    '''
    Selectively compute metrics and store into computed_my_COCO_BBvals.csv
    'simple': simple baselines [Point,Size]
    'area': Area based metrics [Precision, Recall, Jaccard index]
    'distance': Distance based metrics [MunkresEuclidean]
    '''
    if len(compute_metrics)==3: print "Note: It will take about 2 hours to compute all metrics for all workers"
    save_db_as_csv(connect=False)
    #If we are recomputing everything, then load brand new bb_info table
    img_info,object_tbl,bb_info,hit_info = load_info()
    if len(compute_metrics)!=3:
        #Reuse the table information already stored in computed_my_COCO_BBvals.csv
        bb_info = pd.read_csv('computed_my_COCO_BBvals.csv')
    else:
        #Load COCO annotations 
        dataDir='../../coco/'
        dataType='train2014'
        annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)
        # initialize COCO api for instance annotations
        coco=COCO(annFile)
        ground_truth = pd.read_csv("../../data/object_ground_truth.csv")
        my_BBG  = pd.read_csv("my_ground_truth.csv")

    for bb in tqdm(list(bb_info.iterrows())):

        oid = bb[1]["object_id"]
        #Image information 
        image_id = int(object_tbl[object_tbl.object_id==oid].image_id)
        img_name = img_info["filename"][image_id-1]

        bbx_path= bb[1]["x_locs"]
        bby_path= bb[1]["y_locs"]
        worker_x_locs,worker_y_locs= process_raw_locs([bbx_path,bby_path])
        worker_x_locs,worker_y_locs = zip(*list(OrderedDict.fromkeys(zip(worker_x_locs,worker_y_locs))))
        if ('area' in compute_metrics) or ('dist' in compute_metrics):
            cocoimg_id = int(img_name.split('_')[-1])
            annIds = coco.getAnnIds(imgIds=cocoimg_id, iscrowd=None)
            anns = coco.loadAnns(annIds)
            ground_truth_match = ground_truth[ground_truth.id==str(oid)]
            COCO_id = int(ground_truth_match["COCO_annIds"])
        #Simple Baseline Measures
        if 'simple' in compute_metrics:
            bb_info = bb_info.set_value(bb[0],"Num Points",len(worker_x_locs))
            bb_info = bb_info.set_value(bb[0],"Area Ratio",compute_img_to_bb_area_ratio(img_name,worker_x_locs,worker_y_locs))
        if ('area' in compute_metrics) or ('dist' in compute_metrics):
            # Comparing with COCO-Annotations
            for ann in anns:
                if COCO_id==-1:
                    #No BB for this object collected by MSCOCO
                    pass
                elif ann['id'] == COCO_id: 
                    print ann["segmentation"]
                    annBB =ann["segmentation"][0]
                    coco_x_locs,coco_y_locs = process_raw_locs(annBB,COCO=True)
                    #Remove duplicates
                    coco_x_locs,coco_y_locs = zip(*list(OrderedDict.fromkeys(zip(coco_x_locs,coco_y_locs))))
                    obj_x_locs = [list(worker_x_locs),list(coco_x_locs)]
                    obj_y_locs = [list(worker_y_locs),list(coco_y_locs)]
                    if ('area' in compute_metrics):
                        bb_info = bb_info.set_value(bb[0],"Jaccard [COCO]",majority_vote(obj_x_locs,obj_y_locs))
                        bb_info = bb_info.set_value(bb[0],"Precision [COCO]",precision(obj_x_locs,obj_y_locs))
                        bb_info = bb_info.set_value(bb[0],"Recall [COCO]",recall(obj_x_locs,obj_y_locs))    
                    if ('dist' in compute_metrics): bb_info = bb_info.set_value(bb[0],"Munkres Euclidean [COCO]",MunkresEuclidean(obj_x_locs,obj_y_locs))
                    break
            # Comparing with SELF ground truth
            my_ground_truth_match = my_BBG[my_BBG.object_id==oid]
            my_x_locs,my_y_locs =  process_raw_locs([my_ground_truth_match["x_locs"].iloc[0],my_ground_truth_match["y_locs"].iloc[0]])
            my_x_locs,my_y_locs = zip(*list(OrderedDict.fromkeys(zip(my_x_locs,my_y_locs))))
            obj_x_locs = [list(worker_x_locs),list(my_x_locs)]
            obj_y_locs = [list(worker_y_locs),list(my_y_locs)]
            if ('area' in compute_metrics):
                bb_info = bb_info.set_value(bb[0],"Jaccard [Self]",majority_vote(obj_x_locs,obj_y_locs))   
                bb_info = bb_info.set_value(bb[0],"Precision [Self]",precision(obj_x_locs,obj_y_locs))
                bb_info = bb_info.set_value(bb[0],"Recall [Self]",recall(obj_x_locs,obj_y_locs))
            if ('dist' in compute_metrics): bb_info = bb_info.set_value(bb[0],"Munkres Euclidean [Self]",MunkresEuclidean(obj_x_locs,obj_y_locs))
    if ('dist' in compute_metrics):
        #Normalized Munkres Euclidean = NME
        coco_dist = bb_info[bb_info['Munkres Euclidean [COCO]']!=-1]["Munkres Euclidean [COCO]"]
        self_dist = bb_info[bb_info['Munkres Euclidean [Self]']!=-1]["Munkres Euclidean [Self]"]
        bb_info["NME [COCO]"]= 1-coco_dist/coco_dist.max()
        bb_info["NME [Self]"]= 1-self_dist/self_dist.max()
    #Drop Unnamed columns (index from rewriting same file)
    bb_info = bb_info[bb_info.columns[~bb_info.columns.str.contains('Unnamed:')]]
    # replace all NAN values with -1, these are entries for which we don't have COCO ground truth
    bb_info = bb_info.fillna(-1)
    bb_info.to_csv("computed_my_COCO_BBvals.csv")
    return bb_info


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
def precision_from_list(test_list, base_poly):
    int_area = 0.0
    test_poly_area = 0.0
    for test_poly in test_list:
	#print test_poly
	#print base_poly
        int_area += intersection_area(test_poly, base_poly)
        test_poly_area += test_poly.area
    return (int_area / test_poly_area) if (test_poly_area != 0) else 0

def recall_from_list(test_list, base_poly):
    int_area = 0.0
    for test_poly in test_list:
        int_area += intersection_area(test_poly, base_poly)
    return (int_area / base_poly.area) if (base_poly.area != 0) else 0
def prj_from_list(test_list, base_poly):
    int_area = 0.0
    test_poly_area = 0.0
    for test_poly in test_list:
        #print test_poly
        #print base_poly
        int_area += intersection_area(test_poly, base_poly)
        test_poly_area += test_poly.area
    precision = (int_area / test_poly_area) if (test_poly_area != 0) else 0
    recall = (int_area / base_poly.area) if (base_poly.area != 0) else 0
    jaccard = (int_area / (base_poly.area+test_poly_area-int_area)) if (test_poly.area != 0) else 0
    return precision,recall,jaccard
def intersection_area(poly1, poly2):
    intersection_poly = None
    try:
        try:
            intersection_poly = poly1.intersection(poly2)
        except:
            try:
                intersection_poly = poly1.buffer(0).intersection(poly2)
            except:
                intersection_poly = poly1.buffer(1e-10).intersection(poly2)
        return intersection_poly.area
    except:
        print 'intersection failed'
        return 0

if __name__ =="__main__":
    import sys
    start_time = time.time()
    if len(sys.argv)>1:
        if sys.argv[1]=='test':
            simple_rectangle_test()
            real_BB_test()
        elif sys.argv[1]=='compute':
            compute_metrics=sys.argv[2].split(',')
            print compute_metrics
            compute_my_COCO_BBvals(compute_metrics)
    else:    
        print "Usage: python qualityBaseline.py test/compute"
        compute_my_COCO_BBvals()
    print "---%f seconds---" % (time.time() - start_time)
