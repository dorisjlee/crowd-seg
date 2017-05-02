# visualizing vision tiles and worker / ground truth boxes
# constructing vision baseline using vision tiles and worker / ground truth boxes

import shapely
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pickle
import glob
import os

BASE_DIR = '/Users/akashds1/Dropbox/CrowdSourcing/Image-Segmentation/'
ALL_SAMPLES_DIR = BASE_DIR + 'all_samples'


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
    # plt.figure()
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


def find_all_tk_in_shell(tiles, current_shell_idx, exclude_idx=[]):
    # Find all tiles at the shell d=d+1
    # add all tiles adjacent to currentShell front
    filtered_tidxs = np.delete(np.arange(len(tiles)), exclude_idx)

    adjacent_tkidxs = []
    for ctidx in current_shell_idx:
        ck = tiles[ctidx]
        for tkidx in filtered_tidxs:
            tk = tiles[tkidx]
            if adjacent(tk, ck):
                adjacent_tkidxs.append(tkidx)
    # There might be a lot of duplicate tiles that is adjacent to more than one tile on the current shell front
    return list(set(adjacent_tkidxs))


def calc_Tstar(tiles, pInT, pNotInT, seed_tile, thresh_param):
    tile_ids_processed = []
    output_tile_ids = [seed_tile]
    Tstar = Polygon()

    current_frontier = [seed_tile]
    tiles_to_explore_next = find_all_tk_in_shell(tiles, current_frontier, tile_ids_processed)

    # print 'Num tiles: ', len(tiles)
    # print 'len(pInT) = ', len(pInT)
    # print 'len(pNotInT) = ', len(pNotInT)

    # print 'pInT = ', pInT
    # print 'pNotInT = ', pNotInT

    while (len(tiles_to_explore_next) > 0):
        # print 'len(tiles_to_explore_next) = ', len(tiles_to_explore_next)
        current_frontier = []
        for tid in tiles_to_explore_next:
            tile_ids_processed.append(tid)
            current_frontier.append(tid)
            if pInT[tid] >= thresh_param * pNotInT[tid]:
                # include tile
                output_tile_ids.append(tid)
                # Tstar.union(tiles[tid])

        tiles_to_explore_next = find_all_tk_in_shell(tiles, current_frontier, tile_ids_processed)

    return output_tile_ids, Tstar


def calc_all_Tstars(indir):
    # objects = get_obj_to_img_id().keys()
    for sample_path in glob.glob('{}/*/'.format(indir)):
        print '======================================================'
        sample_name = sample_path.split('/')[-2]
        print 'Processing sample ', sample_name
        # if sample_name != '5worker_rand0':
        #     continue
        # for objid in objects:
        for objid in range(1, 48):
            # if objid != 18:
            #     continue
            print 'Doing object ', objid
            print '-----------------------------------------------------'
            tiles = pickle.load(open('{}vtiles{}.pkl'.format(sample_path, objid)))
            indMat = pickle.load(open('{}indMat{}.pkl'.format(sample_path, objid)))
            seed_tile = core(tiles, indMat)[0]
            for pInT_iter_path in glob.glob('{}pInT_lst_obj{}_iter*.pkl'.format(sample_path, objid)):
                # load all data
                iter_num = pInT_iter_path.split('.')[-2][-1]
                print 'Doing iter num ', iter_num
                pInT = pickle.load(open(pInT_iter_path))
                pNotInT = pickle.load(open('{}pInT_lst_obj{}_iter{}.pkl'.format(sample_path, objid, iter_num)))
                # pInT = pickle.load(open('{}indMat{}.pkl'.format(sample_path, objid)))

                # continue
                for thresh_param in [1.0]:
                    output_tile_ids, Tstar = calc_Tstar(tiles, pInT, pNotInT, seed_tile, thresh_param)
                    # print 'output tile ids: ', output_tile_ids
                    outdir = sample_path + 'obj{}/'.format(objid) + 'thresh{}/'.format(int(thresh_param * 10)) + 'iter_{}/'.format(iter_num)
                    if not os.path.isdir(outdir):
                        os.makedirs(outdir)
                    with open('{}/tid_list.pkl'.format(outdir), 'w') as fp:
                        fp.write(pickle.dumps(output_tile_ids))
                    with open('{}/Tstar.pkl'.format(outdir), 'w') as fp:
                        fp.write(pickle.dumps(Tstar))

                    # sanity check plots
                    plt.figure()
                    tids = pickle.load(open('{}/tid_list.pkl'.format(outdir)))
                    final_tiles = [tiles[tid] for tid in tids]
                    plot_coords(get_gt(objid), color='red', fill_color='red')
                    visualizeTilesSeparate(final_tiles, colorful=False)
                    plot_coords(tiles[seed_tile], color='blue', fill_color='blue')
                    plt.savefig('{}/out.png'.format(outdir))
                    plt.close()

            print '-----------------------------------------------------'


if __name__ == '__main__':
    calc_all_Tstars(ALL_SAMPLES_DIR)
