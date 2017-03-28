import os
import matplotlib.pyplot as plt
import numpy as np
from TileEM_plot_toolbox import *
os.chdir("output")

def PR_compare(objid,overlap_threshold):    
    worker_ids,worker_precision_lst,worker_recall_lst = compute_worker_PR_obj(objid,return_worker_id=True)
    best_worker =  np.argmax(worker_recall_lst)
    # print "Best worker's PR against BBG: ", max(worker_precision_lst),max(worker_recall_lst)
    os.chdir("..")
    tiles,indicatorMat= createObjIndicatorMatrix(objid,PRINT=False,overlap_threshold=overlap_threshold)
    os.chdir(DATA_DIR)
    approved_tiles = np.where(indicatorMat[best_worker]==1)[0]
    for tidx in approved_tiles:
        plot_coords(Polygon(tiles[tidx]),color="lime")

    bb_objects = bb_info[bb_info["object_id"]==objid]
    bb_objects =  bb_objects[bb_objects.worker_id!=3]
    best_worker_id = worker_ids[best_worker]
    
    worker_bb_info = bb_objects[bb_objects["worker_id"]==best_worker_id]
    worker_BB_polygon = Polygon(zip(*process_raw_locs([worker_bb_info["x_locs"].values[0],worker_bb_info["y_locs"].values[0]])))#.buffer(0)

    plot_coords(worker_BB_polygon,linestyle='--',color='#0000ff',reverse_xy=True)

    joined_bb = join_tiles(approved_tiles,tiles)

    joined_p, joined_r = compute_PR(objid,approved_tiles,tiles)
    # print "Joined PR:",joined_p, joined_r
    # PR against each other 
    precision = worker_BB_polygon.intersection(joined_bb).area/joined_bb.area
    recall = worker_BB_polygon.intersection(joined_bb).area/worker_BB_polygon.area
    return precision,recall
os.chdir("..")
img_info,object_tbl,BB_info,hit_info = load_info()
os.chdir(DATA_DIR)
mega_plst=[]
mega_rlst=[]
for overlap_threshold in tqdm(np.linspace(0.1,0.9,9)):
    print "Working on threshold >",overlap_threshold
    p_lst = []
    r_lst = []
    for objid in object_lst:
        plt.figure()
        plt.title("Obj{0} [thres>{1}]".format(objid,overlap_threshold))
        p,r=PR_compare(objid,overlap_threshold)
        plt.suptitle("p={0};r={1}".format(p,r))
        plt.savefig("Obj{0}_thres{1}.pdf".format(objid,overlap_threshold))
        p_lst.append(p)
        r_lst.append(r)
    mega_plst.append(p_lst)
    mega_rlst.append(r_lst)
print mega_plst
print mega_rlst
np.savetxt("plst.txt",mega_plst)
np.savetxt("rlst.txt",mega_rlst)