import os
import pandas as pd 
import pickle as pkl
import json
from PIL import Image, ImageDraw
import shapely
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import sys
sys.path.append('../')
from analysis_toolbox import *
from qualityBaseline import *
import copy
#DATA_DIR="sample/"+os.getcwd().split('/')[-1]
DATA_DIR="."
#DATA_DIR="sampletopworst5"
img_info,object_tbl,bb_info,hit_info = load_info()

def createObjIndicatorMatrix(objid,tiles="",load_existing_tiles_from_file=False, PLOT=False,sampleNworkers=-1,random_state=111,PRINT=False,SAVE=False,EXCLUDE_BBG=True,overlap_threshold=0.8,tile_only=False,tqdm_on=False):

    # Ji_tbl (bb_info) is the set of all workers that annotated object i 
    bb_objects = bb_info[bb_info["object_id"]==objid]
    if EXCLUDE_BBG: bb_objects =  bb_objects[bb_objects.worker_id!=3]
    # Sampling Data from Ji table 
    if sampleNworkers>0 and sampleNworkers<len(bb_objects):
        bb_objects = bb_objects.sample(n=sampleNworkers,random_state=random_state)
    # Create a list of polygons based on worker BBs 
    xylocs = [list(zip(*process_raw_locs([x,y]))) for x,y in zip(bb_objects["x_locs"],bb_objects["y_locs"])]
    BB = []
    for xyloc in xylocs:
        BB.append(Polygon(xyloc).buffer(0))

    #Compute Tiles 
    if load_existing_tiles_from_file:
        tiles = pkl.load(open("{0}/tiles{1}.pkl".format(DATA_DIR,objid),'r'))
        #worker_lst= pkl.load(open("{0}/worker{1}.pkl".format(DATA_DIR,objid),'r'))
    elif tiles=="":
        tiles = BB2TileExact(objid,BB,tqdm_on=tqdm_on,save_tiles=SAVE)
        tiles = compute_unique_tileset(objid,tiles,SAVE=SAVE)
    if tile_only:
    	if PLOT: visualizeTiles(tiles)
    	return tiles,0
    # Convert set of tiles to indicator matrix for all workers and tiles
    # by checking if the worker's BB contains the tile pieces
    # The indicator matrix is a (N + 1) X M matrix, 
    # with first N rows indicator vectors for each annotator and
    # the last row being region sizes
    worker_lst = list(bb_objects.worker_id)
    M = len(tiles)
    N = len(worker_lst)
    if PRINT: 
        print "Number of non-overlapping tile regions (M) : ",M
        print "Number of workers (N) : ",N
    indicator_matrix = np.zeros((N+1,M))

    for  wi in range(N):
        worker_id = worker_lst[wi]
        worker_bb_info = bb_objects[bb_objects["worker_id"]==worker_id]
        worker_BB_polygon = Polygon(zip(*process_raw_locs([worker_bb_info["x_locs"].values[0],worker_bb_info["y_locs"].values[0]]))).buffer(0)

        # Check if worker's polygon contains this tile
        for tile_i in range(M):
            tile = tiles[tile_i]
            if worker_BB_polygon.contains(tile.centroid):
                indicator_matrix[wi][tile_i]=1
            else:
            	try:
            		tileBB_overlap = tile.intersection(worker_BB_polygon).area/float(tile.area)
            		if tileBB_overlap>=overlap_threshold:
            			indicator_matrix[wi][tile_i]=1
            	except(shapely.geos.TopologicalError):
            		pass

    # The last row of the indicator matrix is the tile area
    for tile_i in range(M):
        tile=tiles[tile_i]
        indicator_matrix[-1][tile_i]=tile.area
    # Debug plotting all tiles that have not been voted by workers 
    all_unvoted_tiles=np.where(np.sum(indicator_matrix[:-1],axis=0)==0)[0]
    if PRINT:
        print "all unvoted tiles:",all_unvoted_tiles
        print "all unvoted workers:",np.where(np.sum(indicator_matrix,axis=1)==0)[0]
    if PLOT or PRINT:
        print "Object ",objid
        sanity_check(indicator_matrix,PLOT)
    if SAVE:
    	pkl.dump(worker_lst,open('{0}/worker{1}.pkl'.format(DATA_DIR,objid),'w'))
    	pkl.dump(indicator_matrix,open('{0}/indMat{1}.pkl'.format(DATA_DIR,objid),'w'))
    return worker_lst,tiles,indicator_matrix
def add_object_to_tiles(tiles,obj):
    if obj==[]:
        return
    if  obj.is_valid:
        if type(obj)==shapely.geometry.polygon.Polygon and obj.area>1e-8:
            tiles.append(obj)
        elif type(obj)==shapely.geometry.MultiPolygon or type(obj)==shapely.geometry.collection:
            for region in obj:
                if type(region)!=shapely.geometry.LineString and region.area>1e-8:
                    tiles.append(region)  
def overlap(a,b):
    if a.area>b.area:
        larger_area = a.area
    else:
        larger_area = b.area
    return a.intersection(b).area/larger_area
def compute_unique_tileset(objid,tiles,SAVE=False,PLOT=False):
    verified_tiles = []
    for tidx in tqdm(range(len(tiles))): 
        t=tiles[tidx]
        duplicated=False
        verified_tiles_new= verified_tiles[:]
        for vtidx in range(len(verified_tiles)):
            try:
                vt = verified_tiles[vtidx]
            except(IndexError):
                print "last element removed"
            try:
                overlap_score=overlap(vt,t)
                if overlap_score>0.2:
                    print "Duplicate tiles: ",tidx,vtidx, overlap_score, vt.area, t.area
                    duplicated=True
                    if overlap_score<0.99:
                        verified_tiles_new.remove(vt)
                        overlap_region = vt.intersection(t)
                        add_object_to_tiles(verified_tiles_new,overlap_region)
                        add_object_to_tiles(verified_tiles_new,vt.difference(overlap_region))
                        add_object_to_tiles(verified_tiles_new,t.difference(overlap_region))

                    if PLOT: 
                        plt.figure()
                        plt.title("[{0},{1}]{2}".format(tidx,vtidx, overlap_score))

                        try:
                            plot_coords(vt)
                            plot_coords(t,linestyle='--',color="blue")
                            plot_coords(overlap_region,fill_color="lime")
                        except(AttributeError):
                            print "problem with plotting"
            except(shapely.geos.TopologicalError):
                print "Topological Error",tidx,vtidx
        if not duplicated:
            verified_tiles_new.append(t)
        verified_tiles=verified_tiles_new[:]
    if SAVE: pkl.dump(verified_tiles,open("{0}/vtiles{1}.pkl".format(DATA_DIR,objid),'w'))
    return verified_tiles
def slow_cascaded_union(tiles):
    all_tiles  = copy.deepcopy(tiles)
    Utile=tiles[0]
    all_tiles.remove(tiles[0])
    i=1
    loss_tiles = []
    while(len(all_tiles)>0):
        tile = tiles[i]
        try:
            Utile = Utile.union(tile)
        except(shapely.geos.TopologicalError):
            try:
                Utile =Utile.buffer(0).union(tile.buffer(0))
            except(shapely.geos.TopologicalError):
#                 return tile,Utile 
                #print "Throwing away:", tile.area 
                loss_tiles.append(tile)
        all_tiles.remove(tile)
        i+=1
    try:
        Utile=Utile.union(cascaded_union(loss_tiles))
    except(shapely.geos.TopologicalError):
        try:
            Utile=Utile.buffer(9e-13).union(cascaded_union(loss_tiles).buffer(0))
        except(shapely.geos.TopologicalError):
            loss_area = sum([t.area for t in loss_tiles])
            print "Throwing away:", loss_area
    return loss_tiles,Utile
def visualizeTilesSeparate(tiles,colorful=True):
    plt.figure()
    colors=cm.rainbow(np.linspace(0,1,len(tiles)))
    for t,i in zip(tiles,range(len(tiles))): 
#         plt.figure()
        if colorful: 
            c = colors[i]
        else: 
            c="lime"
        if type(t)==shapely.geometry.polygon.Polygon:
            plot_coords(t,color=c,reverse_xy=True,fill_color=c)
        elif type(t)==shapely.geometry.MultiPolygon or type(t)==shapely.geometry.collection:
            for region in t:
                
                if type(t)!=shapely.geometry.LineString:
                    plot_coords(region,color=c,reverse_xy=True,fill_color=c)
    #xylocs of the largest tile for estimating the obj size
    xlocs,ylocs = tiles[np.argmax([t.area for t in tiles])].exterior.coords.xy
    plt.ylim(np.min(ylocs)-50,np.max(ylocs)+50)
    plt.gca().invert_yaxis()
def BB2TileExact(objid,BB,tqdm_on=False,save_tiles=True):
    '''
    Given a list of worker polygons BB (potentially sampled) and the objectID 
    return a list of non-overlapping tiles (shapely Polygon objects)
    # BB is a list of polygons based on worker BBs 
    ''' 
    tiles=[]
    if tqdm_on: 
        BB_lst = tqdm(range(len(BB)))
    else:
        BB_lst=range(len(BB))
    for i in BB_lst:
        bi = BB[i]
        # base case, when i=0, only 2 polygon intersecting
        if i==0:
            tiles.append(bi)
        else: 
            xj_lst = []
            for tj in tiles:
                try:
                    xj=tj.intersection(bi)
                    if xj.area>1e-10: #eliminating spurious LineString-looking Polygons 
                        tiles.remove(tj)
                        add_object_to_tiles(tiles,xj)
                        diff_region = (tj.symmetric_difference(xj)).difference(xj)#.buffer(0)
                        add_object_to_tiles(tiles,diff_region)       
                        xj_lst.append(xj)
                except(shapely.geos.TopologicalError):
                    xj_lst=xj_lst[:-1]
                    pass
            try:
                leftovers = bi.difference(cascaded_union(xj_lst))
            except(shapely.geos.TopologicalError):
                try: 
                    leftovers = bi.difference(cascaded_union(xj_lst).buffer(1e-10))
                except(shapely.geos.TopologicalError):
                    leftovers=[]    
            except(ValueError):
                #ERROR:shapely.geos:TopologyException: no outgoing dirEdge 
                leftovers=[]
                
            add_object_to_tiles(tiles,leftovers)
    if save_tiles: pkl.dump(tiles,open('tiles{0}.pkl'.format(objid),'w'))
    return tiles

def visualizeTiles(tiles,colorful=True):
    plt.figure()
    colormap = plt.cm.rainbow
    for t in tiles: 
        if colorful: 
            c = colormap(i)
        else: 
            c="lime"
        if type(t)==shapely.geometry.polygon.Polygon:
            plot_coords(t,color=c,reverse_xy=True)
        elif type(t)==shapely.geometry.MultiPolygon or type(t)==shapely.geometry.collection:
            for region in t:
                if type(t)!=shapely.geometry.LineString:
                    plot_coords(region,color=c,reverse_xy=True)

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

def plot_coords(obj,color='red',reverse_xy=False,linewidth=2,linestyle='-',fill_color=""):
    #Plot shapely polygon coord 
    if type(obj)!=shapely.geometry.MultiPolygon:
        obj=[obj]
    for ob in obj: 
        if reverse_xy:
            x,y = ob.exterior.xy
        else:
            y,x = ob.exterior.xy
        plt.plot(x, y, linestyle, color=color,linewidth=linewidth, zorder=1)
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
