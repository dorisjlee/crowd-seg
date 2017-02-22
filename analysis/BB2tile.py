import pandas as pd 
from analysis_toolbox import *
from qualityBaseline import *
os.chdir("../../BoundingBoxArchive/Code/")
from greedy import *
from data import *
from experiment import *
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
################################################
##                                            ##
##        Preprocessing                       ##
##                                            ##
################################################
# Loading bounding box drawn by workers
bb_info = pd.read_csv('../../crowd-seg/analysis/computed_my_COCO_BBvals.csv')
obj_sorted_tbl =  bb_info[bb_info['Jaccard [COCO]']!=-1][bb_info['Jaccard [COCO]']!=0][bb_info['Jaccard [Self]']!=0].sort('object_id')
object_id_lst  = list(set(obj_sorted_tbl.object_id))
img_info,object_tbl,bb_info,hit_info = load_info()

def createObjIndicatorMatrix(objid,PLOT=False):
    # Ji_tbl (bb_info) is the set of all workers that annotated object i 
    bb_objects  = obj_sorted_tbl[obj_sorted_tbl["object_id"]==objid]
    # Create a masked image for the object
    # where each of the worker BB is considered a mask and overlaid on top of each other 
    img_name = img_info[img_info.id==int(object_tbl[object_tbl.id==objid]["image_id"])]["filename"].iloc[0]
    fname = "../../crowd-seg/web-app/app/static/"+img_name+".png"
    img=mpimg.imread(fname)
    width,height = get_size(fname)
    mega_mask = np.zeros((height,width))
    img = Image.new('L', (width, height), 0)
    obj_x_locs = [process_raw_locs([x,y])[0] for x,y in zip(bb_objects["x_locs"],bb_objects["y_locs"])]
    obj_y_locs = [process_raw_locs([x,y])[1] for x,y in zip(bb_objects["x_locs"],bb_objects["y_locs"])]
    for x_locs, y_locs in zip(obj_x_locs,obj_y_locs):
        ImageDraw.Draw(img).polygon(zip(x_locs,y_locs), outline=1, fill=1)
        mask = np.array(img)==1
        mega_mask+=mask
    if PLOT: 
        # Visualize mega_mask
        plt.figure(figsize=(5,10))
        plt.imshow(mega_mask)
        plt.colorbar()

    # Create masks for single valued tiles (so that they are more disconnected)
    from matplotlib import _cntr as cntr
    tiles = [] # list of coordinates of all the tiles extracted
    unique_tile_values = np.unique(mega_mask)
    for tile_value in unique_tile_values:
        singly_masked_img = np.zeros_like(mega_mask)
        for x,y in zip(*np.where(mega_mask==tile_value)):
            singly_masked_img[x][y]=1
        #Extract a set of contours from these masks
        x, y = np.mgrid[:singly_masked_img.shape[0], :singly_masked_img.shape[1]]
        c = cntr.Cntr(x, y, singly_masked_img)
        # trace a contour at z ~= 1
        res = c.trace(0.9)
        if PLOT: plot_trace_contours(res)
        for segment in res:
            if segment.dtype!=np.uint8:
                tiles.append(segment)
    # Convert set of tiles to indicator matrix for all workers and tiles
    # by checking if the worker's BB contains the tile pieces
    # The indicator matrix is a (N + 1) X M matrix, 
    # with first N rows indicator vectors for each annotator and
    # the last row being region sizes
    M = len(tiles)
    worker_lst  = np.unique(bb_objects.worker_id)
    N = len(worker_lst)
    indicator_matrix = np.zeros((N+1,M))
    for  wi in range(N):
        worker_id = worker_lst[wi]
        worker_bb_info = bb_objects[bb_objects["worker_id"]==worker_id]
        worker_BB_polygon = Polygon(zip(*process_raw_locs([worker_bb_info["x_locs"].values[0],worker_bb_info["y_locs"].values[0]])))
        # Check if worker's polygon contains this tile
        for tile_i in range(M):
            #tile = Polygon(tiles[tile_i])
            #Take the transpose of the tile graph polygon because during the tile creation process the xy was flipped
            tile= Polygon(zip(tiles[tile_i][:,1],tiles[tile_i][:,0]))
            # Check that tiles are indeed close to BB (no mis-alignment issue)
            if PLOT and tile_i==0:
                plt.figure()
                plot_coords(tile)
                plot_coords(worker_BB_polygon,color="blue")
            if worker_BB_polygon.contains(tile): 
                indicator_matrix[wi][tile_i]=1
    # The last row of the indicator matrix is the tile area
    for tile_i in range(M):
        tile= Polygon(zip(tiles[tile_i][:,1],tiles[tile_i][:,0]))
        indicator_matrix[-1][tile_i]=tile.area
    if PLOT: sanity_check(indicator_matrix)
    return indicator_matrix
def sanity_check(indicator_matrix): 
    plt.title("Tile Area")
    plt.semilogy(indicator_matrix[-1])
    plt.figure()
    plt.title("Indicator Matrix")
    #Plot all excluding last row (area)
    plt.imshow(indicator_matrix[:-1],cmap="cool",interpolation='none', aspect='auto')
    plt.colorbar()
def plot_coords(ob,color='red'):
    #Plot shapely polygon coord 
    x, y = ob.exterior.xy
    plt.plot(x, y, '-', color=color, zorder=1)
def plot_trace_contours(res):
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
