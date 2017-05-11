# visualizing vision tiles and worker / ground truth boxes
# constructing vision baseline using vision tiles and worker / ground truth boxes
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import shapely
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from collections import defaultdict
import matplotlib.cm as cm
import numpy as np
import pickle
import glob
import os
import time
BASE_DIR = '/home/jlee782/crowd-seg/analysis/TileEM/'
ALL_SAMPLES_DIR = BASE_DIR + 'uniqueTiles'


def get_obj_to_img_id():
    import csv
    obj_to_img_id = {}
    with open('{}object.csv'.format(BASE_DIR), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            obj_to_img_id[int(row['id'])] = row['image_id']
    return obj_to_img_id


def get_size(fname):
    from PIL import Image
    #Open image for computing width and height of image
    im = Image.open(fname)
    width = im.size[0]
    height = im.size[1]
    return width, height


def plot_coords(obj, color='red', reverse_xy=False, linestyle='-', fill_color="", show=False, invert_y=False):
    #Plot shapely polygon coord
    if type(obj) != shapely.geometry.MultiPolygon:
        obj = [obj]

    for ob in obj:
        if ob.exterior is None:
            print 'Plotting bug: exterior is None (potentially a 0 area tile). Ignoring and continuing...'
            continue
        if reverse_xy:
            x, y = ob.exterior.xy
        else:
            y, x = ob.exterior.xy
        plt.plot(x, y, linestyle, color=color, zorder=1)
        if fill_color != "":
            plt.fill_between(x, y, facecolor=fill_color, color='none', alpha=0.5)
    if invert_y:
        plt.gca().invert_yaxis()
    # if show: plt.show()


def visualizeTilesSeparate(tiles, reverse_xy=False, colorful=True, savename=None, invert_y=True, default_color='lime'):
    
    colors = cm.rainbow(np.linspace(0, 1, len(tiles)))
    for t, i in zip(tiles, range(len(tiles))):
        # plt.figure()
        if colorful:
            c = colors[i]
        else:
            c = default_color
        if type(t) == shapely.geometry.polygon.Polygon:
            plot_coords(t, color=c, reverse_xy=reverse_xy, fill_color=c)
        elif type(t) == shapely.geometry.MultiPolygon or type(t) == shapely.geometry.collection:
            for region in t:
                if type(t) != shapely.geometry.LineString:
                    plot_coords(region, color=c, reverse_xy=reverse_xy, fill_color=c)
    if reverse_xy:
        #xylocs of the largest tile for estimating the obj size
        xlocs, ylocs = tiles[np.argmax([t.area for t in tiles])].exterior.coords.xy
        # plt.ylim(np.min(ylocs)-50,np.max(ylocs)+50)

    if invert_y:
        plt.gca().invert_yaxis()
    # if savename:
    #     plt.savefig(savename)
    # else:
    #     plt.show()


def get_gt(objid, reverse=False, rescale_factor=1):
    # from qualityBaseline import process_raw_locs
    # import pandas as pd
    # objid = 2  # 1756 bag
    # objid = 18  # 480 computer
    # my_BBG = pd.read_csv('{}my_ground_truth.csv'.format(BASE_DIR))
    # ground_truth_match = my_BBG[my_BBG.object_id == objid]
    # x_locs, y_locs = process_raw_locs([ground_truth_match['y_locs'].iloc[0], ground_truth_match['x_locs'].iloc[0]])

    # x_locs = [x / rescale_factor for x in x_locs]
    # y_locs = [y / rescale_factor for y in y_locs]
    # BBG = shapely.geometry.Polygon(zip(x_locs, y_locs))
    # return BBG

    return get_polygon_from_csv('{}my_ground_truth.csv'.format(BASE_DIR), objid, reverse=reverse, rescale_factor=rescale_factor)


def get_polygon_from_csv(filepath, objid, reverse=False, rescale_factor=1):
    from qualityBaseline import process_raw_locs
    import pandas as pd
    # objid = 2  # 1756 bag
    # objid = 18  # 480 computer
    my_BB = pd.read_csv(filepath)
    bb_match = my_BB[my_BB.object_id == objid]
    if reverse:
        x_locs, y_locs = process_raw_locs([bb_match['y_locs'].iloc[0], bb_match['x_locs'].iloc[0]])
    else:
        x_locs, y_locs = process_raw_locs([bb_match['x_locs'].iloc[0], bb_match['y_locs'].iloc[0]])

    x_locs = [x / rescale_factor for x in x_locs]
    y_locs = [y / rescale_factor for y in y_locs]
    BB = shapely.geometry.Polygon(zip(x_locs, y_locs))
    return BB


def core(tiles, indMat):
    # In the initial step, we pick T to be the top 5 area-vote score
    # where we combine the area and vote in a 1:5 ratio
    topk = 1
    area = np.array(indMat[-1])
    votes = indMat[:-1].sum(axis=0)
    norm_area_vote = area/max(area)+5*votes/max(votes)
    tidx = np.argsort(norm_area_vote)[::-1][:topk]
    # return join_tiles(tidx, tiles)[0], list(tidx)
    return tidx


def adjacent(tileA, tileB):
    return tileA.buffer(0.1).overlaps(tileB.buffer(0.1))

def find_all_tk_in_shell(tiles, current_shell_idx, memoized_adjacency=defaultdict(dict), exclude_idx=[]):
    # Find all tiles at the shell d=d+1
    # add all tiles adjacent to currentShell front
    # memoized_adjacency[tid1][tid2] = True if tiles[tid1] is adcent to tiles[id2] else False
    filtered_tidxs = list(np.delete(np.arange(len(tiles)), exclude_idx))
    new_filtered_tidxs = filtered_tidxs[:]
    adjacent_tkidxs = []
    for ctidx in current_shell_idx:
        for tkidx in filtered_tidxs:
	    #print ctidx
	    #print memoized_adjacency.keys()
            if (ctidx not in memoized_adjacency) or (tkidx not in memoized_adjacency[ctidx]):
                # if this pair not seen before, memoize it
                ck = tiles[ctidx]
                tk = tiles[tkidx]
                memoized_adjacency[ctidx][tkidx] = adjacent(ck, tk)
                memoized_adjacency[tkidx][ctidx] = memoized_adjacency[ctidx][tkidx]
            if memoized_adjacency[ctidx][tkidx]:
                # get adjacency from memoized adjacency matrix
                adjacent_tkidxs.append(tkidx)
                new_filtered_tidxs.remove(tkidx)  # add a tile only once
        filtered_tidxs = new_filtered_tidxs
    return list(set(adjacent_tkidxs)), memoized_adjacency

def calc_Tstar(tiles, pInT, pNotInT, seed_tile, thresh_param):
    tile_ids_processed = [seed_tile]
    output_tile_ids = [seed_tile]
    #Tstar = Polygon()

    current_frontier = [seed_tile]
    memoized_adjacency = defaultdict(dict)
    tiles_to_explore_next, memoized_adjacency = find_all_tk_in_shell(tiles, current_frontier, memoized_adjacency, tile_ids_processed)

    while (len(tiles_to_explore_next) > 0):
        # print 'len(tiles_to_explore_next) = ', len(tiles_to_explore_next)
        current_frontier = []
        for tid in tiles_to_explore_next:
            tile_ids_processed.append(tid)
            #current_frontier.append(tid)
            if pInT[tid] >= thresh_param + pNotInT[tid]:
                # include tile
		current_frontier.append(tid)
                output_tile_ids.append(tid)
                # Tstar.union(tiles[tid])

        tiles_to_explore_next, memoized_adjacency= find_all_tk_in_shell(tiles, current_frontier, memoized_adjacency, tile_ids_processed) 

    return output_tile_ids #, Tstar


def calc_all_Tstars(indir,object_lst=range(1, 48),thres_lst=[1]):
    # objects = get_obj_to_img_id().keys()
    sample_batch_lst = glob.glob('{}/*/'.format(indir))
    #print sample_batch_lst
    #sample_batch_lst = ['20workers_rand0','20workers_rand1','20workers_rand2','20workers_rand3']
    #sample_batch_lst = ['5workers_rand8','10workers_rand7','25workers_rand1','30workers_rand0']
    #thres_lst = [0,4,2,-4,-2]
    for sample_path in sample_batch_lst:#sample_batch_lst[10:]:
        #sample_path = '/home/jlee782/crowd-seg/analysis/TileEM/uniqueTiles/'+sample_path+'/' 
	start = time.time() 
        print '======================================================'
        sample_name = sample_path.split('/')[-2]
        print 'Processing sample ', sample_name
        # if sample_name != '5worker_rand0':
        #     continue
        # for objid in objects:
        for objid in object_lst:
            # if objid != 18:
            #     continue
            print 'Doing object ', objid
            print '-----------------------------------------------------'
            tiles = pickle.load(open('{}vtiles{}.pkl'.format(sample_path, objid)))
            indMat = pickle.load(open('{}indMat{}.pkl'.format(sample_path, objid)))
            seed_tile = core(tiles, indMat)[0]
	    
            for pInT_iter_path in glob.glob('{}pInT_lst_obj{}_iter*.pkl'.format(sample_path, objid)): 
                # load all data
                iter_num = int(pInT_iter_path.split('.')[-2][-1])
		if iter_num != 4: continue
                print 'Doing iter num ', iter_num
                pInT = pickle.load(open(pInT_iter_path))
                pNotInT = pickle.load(open('{}pNotInT_lst_obj{}_iter{}.pkl'.format(sample_path, objid, iter_num)))

                # continue
                for thresh_param in thres_lst:
		    print "Threshold: ",thresh_param
                    #output_tile_ids = calc_Tstar(tiles, pInT, pNotInT, seed_tile, thresh_param)
                    # print 'output tile ids: ', output_tile_ids
                    outdir = sample_path + 'obj{}/'.format(objid) + 'thresh{}/'.format(int(thresh_param * 10)) + 'iter_{}/'.format(iter_num)
                    if not os.path.isdir(outdir):
                        os.makedirs(outdir)
 			output_tile_ids = calc_Tstar(tiles, pInT, pNotInT, seed_tile, thresh_param)
			with open('{}/tid_list.pkl'.format(outdir), 'w') as fp:
                            fp.write(pickle.dumps(output_tile_ids))
		    else:
			print "Already ran: ",outdir
                    #with open('{}/tid_list.pkl'.format(outdir), 'w') as fp:
                    #    fp.write(pickle.dumps(output_tile_ids))
                    #with open('{}/Tstar.pkl'.format(outdir), 'w') as fp:
                    #    fp.write(pickle.dumps(Tstar))

                    # sanity check plots
                    #plt.figure()
                    #tids = pickle.load(open('{}/tid_list.pkl'.format(outdir)))
                    #final_tiles = [tiles[tid] for tid in tids]
                    #plot_coords(get_gt(objid), color='red', fill_color='red')
                    #visualizeTilesSeparate(final_tiles, colorful=False)
                    #plot_coords(tiles[seed_tile], color='blue', fill_color='blue')
                    #plt.savefig('{}/out.png'.format(outdir))
                    #plt.close()

            print '-----------------------------------------------------'
	end=time.time()
	print "Time:", str(end-start)
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
    except:
        print 'intersection failed'

    int_area = intersection_poly.area if intersection_poly else 0
    return int_area
def precision(test_poly, base_poly):
    # returns area-based precision score of two polygons
    int_area = intersection_area(test_poly, base_poly)
    return (int_area / test_poly.area) if (test_poly.area != 0) else 0
def recall(test_poly, base_poly):
    # returns area-based recall score of two polygons
    int_area = intersection_area(test_poly, base_poly)
    return (int_area / base_poly.area) if (base_poly.area != 0) else 0
def testing_cores(indir=ALL_SAMPLES_DIR,object_lst=range(1,48)):
    precisions = []
    recalls = []
    for sample_path in glob.glob('{}/*/'.format(indir)):
        # print '======================================================'
        sample_name = sample_path.split('/')[-2]
        print 'Processing sample ', sample_name
        # if sample_name != '5worker_rand0':
        #     continue
        # for objid in objects:
        for objid in object_lst:
            # if objid != 18:
            #     continue
            # print 'Doing object ', objid
            # print '-----------------------------------------------------'
            tiles = pickle.load(open('{}vtiles{}.pkl'.format(sample_path, objid)))
            indMat = pickle.load(open('{}indMat{}.pkl'.format(sample_path, objid)))
            seed_tile = core(tiles, indMat)[0]

            GTBB = get_gt(objid)
            CORE = tiles[seed_tile]
            plt.figure()
            plot_coords(GTBB, color='red', fill_color='red')
            plot_coords(CORE, color='blue', fill_color='blue')
            outdir = sample_path + 'core_comparison/'
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
            plt.savefig('{}{}.png'.format(outdir, objid))
            plt.close()

            precisions.append(precision(CORE, GTBB))
            recalls.append(recall(CORE, GTBB))
    print 'Precision stats:'
    print 'Mean: {}, median: {}, std: {}, min: {}, max: {}'.format(
        np.mean(precisions), np.median(precisions), np.std(precisions), min(precisions), max(precisions))
    print 'Recall stats:'
    print 'Mean: {}, median: {}, std: {}, min: {}, max: {}'.format(
        np.mean(recalls), np.median(recalls), np.std(recalls), min(recalls), max(recalls))
if __name__ == '__main__':
    mode='production'
    if mode=='testing-core':
	testing_cores(ALL_SAMPLES_DIR)
    elif mode=="production":
        import sys
        object_min = int(sys.argv[1])	
        object_max = int(sys.argv[2])+1
        thres=int(sys.argv[3])
        print "Working on obj{0} to {1}, with threshold={2}".format(object_min, object_max,thres)
        calc_all_Tstars(ALL_SAMPLES_DIR,object_lst=range(object_min, object_max),thres_lst=[thres])
