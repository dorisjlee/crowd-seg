import os
import pandas as pd 
from greedy import *
from data import *
from experiment import *

from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import sys
sys.path.append('../')
from analysis_toolbox import *
from qualityBaseline import *
################################################
##                                            ##
##        Preprocessing                       ##
##                                            ##
################################################
# Loading bounding box drawn by workers
img_info,object_tbl,bb_info,hit_info = load_info()

def createObjIndicatorMatrix(objid,PLOT=False,sampleNworkers=-1,PRINT=False,EXCLUDE_BBG=True,overlap_threshold=0.5):
    # Ji_tbl (bb_info) is the set of all workers that annotated object i 
    bb_objects = bb_info[bb_info["object_id"]==objid]
    if EXCLUDE_BBG: bb_objects =  bb_objects[bb_objects.worker_id!=3]
    # Sampling Data from Ji table 
    if sampleNworkers>0 and sampleNworkers<len(bb_objects):
        bb_objects = bb_objects.sample(n=sampleNworkers)#,random_state=111)
    # Create a masked image for the object
    # where each of the worker BB is considered a mask and overlaid on top of each other 
    img_name = img_info[img_info.id==int(object_tbl[object_tbl.id==objid]["image_id"])]["filename"].iloc[0]
    fname = "../../web-app/app/static/"+img_name+".png"
    img=mpimg.imread(fname)
    width,height = get_size(fname)
    mega_mask = np.zeros((height,width))
    
    obj_x_locs = [process_raw_locs([x,y])[0] for x,y in zip(bb_objects["x_locs"],bb_objects["y_locs"])]
    obj_y_locs = [process_raw_locs([x,y])[1] for x,y in zip(bb_objects["x_locs"],bb_objects["y_locs"])]
    for x_locs, y_locs in zip(obj_x_locs,obj_y_locs):
        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(zip(x_locs,y_locs), outline=1, fill=1)
        mask = np.array(img)==1
        # plt.imshow(mask)
        mega_mask+=mask
    if PLOT: 
        # Visualize mega_mask
        plt.figure()
        plt.imshow(mega_mask,interpolation="none")#,cmap="rainbow")
        plt.colorbar()

    # Create masks for single valued tiles (so that they are more disconnected)
    from matplotlib import _cntr as cntr
    tiles = [] # list of coordinates of all the tiles extracted
    unique_tile_values = np.unique(mega_mask)
    # print unique_tile_values
    for tile_value in unique_tile_values[1:]: #exclude 0
        singly_masked_img = np.zeros_like(mega_mask)
        for x,y in zip(*np.where(mega_mask==tile_value)):
            singly_masked_img[x][y]=1
        #Extract a set of contours from these masks
        x, y = np.mgrid[:singly_masked_img.shape[0], :singly_masked_img.shape[1]]
        c = cntr.Cntr(x, y, singly_masked_img)
        # trace a contour at z ~= 1
        res = c.trace(0.9)
        #if PLOT: plot_trace_contours(singly_masked_img,res)
        for segment in res:
            if segment.dtype!=np.uint8 and len(segment)>2:
                #Take the transpose of the tile graph polygon because during the tile creation process the xy was flipped
                tile= Polygon(zip(segment[:,1],segment[:,0]))
                # print tile.area
                # if tile.area>=1: #FOR DEBUGGING PURPOSES
                tiles.append(segment)

    # Convert set of tiles to indicator matrix for all workers and tiles
    # by checking if the worker's BB contains the tile pieces
    # The indicator matrix is a (N + 1) X M matrix, 
    # with first N rows indicator vectors for each annotator and
    # the last row being region sizes
    M = len(tiles)
    worker_lst  = list(bb_objects.worker_id)
    N = len(worker_lst)
    if PRINT: 
        print "Number of non-overlapping tile regions (M) : ",M
        print "Number of workers (N) : ",N
    indicator_matrix = np.zeros((N+1,M),dtype=int)

    for  wi in range(N):
        worker_id = worker_lst[wi]
        worker_bb_info = bb_objects[bb_objects["worker_id"]==worker_id]
        worker_BB_polygon = Polygon(zip(*process_raw_locs([worker_bb_info["x_locs"].values[0],worker_bb_info["y_locs"].values[0]]))).buffer(0)

        # Check if worker's polygon contains this tile
        for tile_i in range(M):
            # tile = Polygon(tiles[tile_i])
            tile= Polygon(zip(tiles[tile_i][:,1],tiles[tile_i][:,0]))
            # Check that tiles are indeed close to BB (no mis-alignment issue)
            # if PLOT and tile_i==0:
            #     plt.figure()
            #     plot_coords(tile)
            #     plot_coords(worker_BB_polygon,color="blue")
            # if worker_BB_polygon.contains(tile): #or tile.contains(worker_BB_polygon): 

            # Tried worker_BB_polygon expansion method but this led to too many votes among workers in the indicator matrix
            # worker_BB_polygon= worker_BB_polygon.buffer(1.0)
            tileBB_overlap = tile.intersection(worker_BB_polygon).area/float(tile.area)
            
            #If either centroid is not contained in the polygon or overlap is too low, then its prob not a containment tile
            
            # if tileBB_overlap>=0.8:
            if  worker_BB_polygon.contains(tile.centroid) or tileBB_overlap>=overlap_threshold:
            #if worker_BB_polygon.contains(tile.centroid): #or tile.contains(worker_BB_polygon): 
                # plt.figure()
                # plot_coords(worker_BB_polygon,color="green")
                # plot_coords(tile,color="blue")
                # y,x =tile.centroid.xy
                # plt.plot(x[0],y[0],'x',color='red')
                indicator_matrix[wi][tile_i]=1
    # The last row of the indicator matrix is the tile area
    for tile_i in range(M):
        tile= Polygon(zip(tiles[tile_i][:,1],tiles[tile_i][:,0]))
        indicator_matrix[-1][tile_i]=tile.area
    # Debug plotting all tiles that have not been voted by workers 
    all_unvoted_tiles=np.where(np.sum(indicator_matrix[:-1],axis=0)==0)[0]
    if PRINT:
        print "all unvoted tiles:",all_unvoted_tiles
        print "all unvoted workers:",np.where(np.sum(indicator_matrix,axis=1)==0)[0]
    # delete_tile_idx = np.where(np.sum(indicator_matrix[:-1],axis=0)==0)[0]
    # if PRINT: print "Deleting ", len(delete_tile_idx),"tiles: ",delete_tile_idx
    # indicator_matrix = np.delete(indicator_matrix,delete_tile_idx,axis=1)
    # for tile_i in delete_tile_idx: 
    #     tile= Polygon(tiles[tile_i])
    #     plot_coords(tile)
    #     # print "Tile",tile_i
    #     # print tile.intersection(worker_BB_polygon).area
    #     # print worker_BB_polygon.intersection(tile).area
    #     # print float(tile.area)
    #     print tile.area
    #     tiles.pop(tile_i) #remove corresponding tile information
    # colors=cm.rainbow(np.linspace(0,1,len(np.where(np.sum(indicator_matrix[:-1],axis=0)==0)[0])))
    
    # Debug Visualizing what the bad bounding boxes look like
    # os.chdir("..")
    # visualize_bb_objects(objid,img_bkgrnd=False,gtypes=['worker'])
    # os.chdir("TileEM/")
    # for tile_idx,c in zip(all_unvoted_tiles,colors):
    #     plt.plot(tiles[tile_idx][:,1],tiles[tile_idx][:,0],color=c,linewidth=3,linestyle='--')
        # bad_tile = Polygon(tiles[tile_idx])
        # shrunk_bad_tile=bad_tile.buffer(-0.5)
        # plot_coords(shrunk_bad_tile)
    
    # if len(all_unvoted_tiles)!=0:
    #     for tile_idx  in  all_unvoted_tiles :
    #         tile = Polygon(zip(tiles[tile_idx][:,1],tiles[tile_idx][:,0]))
    #         overlap_lst=[]
    #         max_overlap=True
    #         for wi in range(len(worker_lst)):
    #             worker_id = worker_lst[wi]
    #             worker_bb_info = bb_objects[bb_objects["worker_id"]==worker_id]
    #             worker_BB_polygon = Polygon(zip(*process_raw_locs([worker_bb_info["x_locs"].values[0],worker_bb_info["y_locs"].values[0]]))).buffer(0)

    #             tileBB_overlap = tile.intersection(worker_BB_polygon).area/float(tile.area)
    #             overlap_lst.append(tileBB_overlap)
    #             if tileBB_overlap>0.9:
    #                 indicator_matrix[wi][tile_idx]=1
    #                 max_overlap=False
    #         if max_overlap:
    #             most_overlapping_workerBB = np.argmax(overlap_lst)
    #             indicator_matrix[most_overlapping_workerBB][tile_idx]=1
    #             # #visually checking that tiles that don't pass the threshold and we pick from max overlap is decent
    #             worker_bb_info = bb_objects[bb_objects["worker_id"]==worker_lst[most_overlapping_workerBB]]
    #             worker_BB_polygon = Polygon(zip(*process_raw_locs([worker_bb_info["x_locs"].values[0],worker_bb_info["y_locs"].values[0]]))).buffer(0)
    #             # plt.figure()
    #             # plt.title(str(overlap_lst[most_overlapping_workerBB]))
    #             # plot_coords(tile)
    #             # plot_coords(worker_BB_polygon,color="blue")
    #     for wi in np.where(np.sum(indicator_matrix,axis=1)==0)[0]:
    #         worker_id = worker_lst[wi]
    #         worker_bb_info = bb_objects[bb_objects["worker_id"]==worker_id]
    #         worker_BB_polygon = Polygon(zip(*process_raw_locs([worker_bb_info["x_locs"].values[0],worker_bb_info["y_locs"].values[0]]))).buffer(0)
    #         for tile_idx  in  range(len(tiles)):
    #             tile = Polygon(zip(tiles[tile_idx][:,1],tiles[tile_idx][:,0]))
    #             tileBB_overlap = tile.intersection(worker_BB_polygon).area/float(tile.area)
    #             if tileBB_overlap>0.9:
    #                 indicator_matrix[wi][tile_idx]=1
    #                 # plt.figure()
    #                 # # plt.title(str(overlap_lst[most_overlapping_workerBB]))
    #                 # plt.title(str(tileBB_overlap))
    #                 # plot_coords(tile)
    #                 # plot_coords(worker_BB_polygon,color="blue")
    #     if PRINT:
    #         print "After overlap adding"
    #         print "all unvoted tiles:",np.where(np.sum(indicator_matrix[:-1],axis=0)==0)[0]
    #         print "all unvoted workers:",np.where(np.sum(indicator_matrix,axis=1)==0)[0]
        
        
    # #for all the workers with all-zero rows
    # for wi in np.where(np.sum(indicator_matrix,axis=1)==0)[0]:
    #     worker_id = worker_lst[wi]
    #     worker_bb_info = bb_objects[bb_objects["worker_id"]==worker_id]
    #     worker_BB_polygon = Polygon(zip(*process_raw_locs([worker_bb_info["x_locs"].values[0],worker_bb_info["y_locs"].values[0]])))
    #     dist_lst = []

    #     if len(np.where(np.sum(indicator_matrix[:-1],axis=0)==0)[0])==0:
    #         tile_candidates= range(len(tiles))
    #     else:
    #         tile_candidates=np.where(np.sum(indicator_matrix[:-1],axis=0)==0)[0]
    #     # for tile_i in range(len(tiles)): 
    #     # Pick from the tiles that have not yet been voted by any worker yet
    #     for tile_i in tile_candidates:
    #         #Take the transpose of the tile graph polygon because during the tile creation process the xy was flipped
    #         tile = Polygon(zip(tiles[tile_i][:,1],tiles[tile_i][:,0]))
    #         #Find the closest tile that corresponds to that worker
    #         dist_lst.append(worker_BB_polygon.distance(tile))
    #     argmin_dist_idx = np.where(dist_lst==min(np.array(dist_lst)))[0]
    #     for min_dist_idx in argmin_dist_idx:
    #         indicator_matrix[wi][tile_i]=1
    # # Throw out all tiles that have not yet been voted by any worker
    # try:
    #     delete_tile_idx = np.where(np.sum(indicator_matrix[:-1],axis=0)==0)[0]
    #     if PRINT: print "Deleting ", len(delete_tile_idx),"tiles: ",delete_tile_idx
    #     indicator_matrix = np.delete(indicator_matrix,delete_tile_idx,axis=1)
    #     for _i in delete_tile_idx: tiles.pop(_i) #remove corresponding tile information
    # except(IndexError):
    #     print "IndexError"
    #     pass
    if PLOT or PRINT:
        print "Object ",objid
        sanity_check(indicator_matrix,PLOT)
        
    return tiles,indicator_matrix

def sanity_check(indicator_matrix,PLOT=False): 
    print "Check that there are no all-zero rows in indicator matrix:" , len(np.where(np.sum(indicator_matrix,axis=1)==0)[0])==0
    print "Check that there are no all-zero columns in indicator matrix:" , len(np.where(np.sum(indicator_matrix[:-1],axis=0)==0)[0])==0
    if PLOT:
        plt.figure()
        plt.title("Tile Area")
        sorted_indicator_matrix = indicator_matrix[:,indicator_matrix[-1].argsort()]
        plt.semilogy(sorted_indicator_matrix[-1])
        plt.plot(sorted_indicator_matrix[-1])
        plt.figure()
        plt.title("Indicator Matrix")
        #Plot all excluding last row (area)
        #plt.imshow(sorted_indicator_matrix[:-1],cmap="cool",interpolation='none', aspect='auto')
        plt.ylabel("Worker #")
        plt.xlabel("Tile #")
        plt.imshow(indicator_matrix[:-1],cmap="cool",interpolation='none', aspect='auto')
        plt.colorbar()

def plot_coords(ob,color='red',reverse_xy=False,linestyle='-',fill_color=""):
    #Plot shapely polygon coord 
    if reverse_xy:
        x,y = ob.exterior.xy
    else:
        y,x = ob.exterior.xy
    plt.plot(x, y, linestyle, color=color, zorder=1)
    if fill_color!="": plt.fill_between(x, y , facecolor=fill_color,color='none', alpha=0.5)

def plot_all_tiles(tiles):
    plt.figure()
    colors=cm.rainbow(np.linspace(0,1,len(tiles)))
    for tile,c in zip(tiles,colors):
        plot_coords(Polygon(tile),color=c)
    plt.gca().invert_yaxis()

def plot_individual_tiles(tiles):
    for tile in tiles :
        plt.figure(figsize=(1,1))
        plot_coords(Polygon(tile),color='red')
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
def plot_trace_contours(singly_masked_img,res):
    # result is a list of arrays of vertices and path codes
    # (see docs for matplotlib.path.Path)
    nseg = len(res) // 2
    segments, codes = res[:nseg], res[nseg:]

    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(singly_masked_img.T, origin='lower',interpolation='none')
    plt.colorbar(img)
    ax.hold(True)
    for segment in segments:
        p = plt.Polygon(segment, fill=False, color='w')
        ax.add_artist(p)
    x= res[0][:,0]
    plt.xlim(min(x)-10,max(x)+20)
    y= res[0][:,1]
    plt.ylim(min(y)-10,max(y)+20)
