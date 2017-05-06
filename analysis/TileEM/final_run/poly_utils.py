import shapely
from shapely.validation import explain_validity
from matplotlib import pyplot as plt
from collections import defaultdict
import matplotlib.cm as cm
import numpy as np
import csv
import pickle

BASE_DIR = '/home/jlee782/crowd-seg/analysis/TileEM/' 
OUR_DIR = BASE_DIR + 'Vision-stuff/'
COLOR_SEGMENTED_IMG_DIR = OUR_DIR + 'color-segmented-images/'
VISION_TILE_DIR = OUR_DIR + 'vision-tiles/'
ORIGINAL_IMG_DIR = OUR_DIR + 'COCO/'
VISION_BASELINE_DIR = OUR_DIR + 'vision-baseline/'
ALL_SAMPLES_DIR = BASE_DIR + 'final_run'


def reverse_poly(poly):
    if type(poly) == shapely.geometry.collection.GeometryCollection:
        try:
            poly = shapely.geometry.Polygon(poly)
        except:
            print 'failed to convert to polygon...'
    y, x = poly.exterior.xy
    poly = shapely.geometry.Polygon(zip(x, y))
    return poly


def rescale_polygon(poly, rescale_factor):
    # try:
    x_locs, y_locs = poly.exterior.xy
    # except AttributeError:
    # if poly.geometry.type == 'Polygon':
    #     x_locs, y_locs = poly.geometry.exterior.xy
    # elif poly.geometry.type == 'MultiPolygon':
    #     allparts = [p.buffer(0) for p in poly.geometry]
    #     poly.geometry = shapely.ops.cascaded_union(allparts)
        # x_locs, y_locs = poly.geometry.exterior.xy  # here happens the error
    x_locs = [x / rescale_factor for x in x_locs]
    y_locs = [y / rescale_factor for y in y_locs]
    scaled_poly = shapely.geometry.Polygon(zip(x_locs, y_locs))
    return scaled_poly


def intersection(poly1, poly2):
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
    return intersection_poly


def poly_diff(poly1, poly2):
    diff_poly = None
    try:
        try:
            diff_poly = poly1.difference(poly2)
        except:
            try:
                diff_poly = poly1.buffer(0).difference(poly2)
            except:
                diff_poly = poly1.buffer(1e-10).difference(poly2)
    except:
        print 'diff failed'
    return diff_poly


def intersection_area(poly1, poly2):
    intersection_poly = intersection(poly1, poly2)
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


def precision_from_list(test_list, base_poly):
    int_area = 0.0
    test_poly_area = 0.0
    for test_poly in test_list:
        int_area += intersection_area(test_poly, base_poly)
        test_poly_area += test_poly.area
    return (int_area / test_poly_area) if (test_poly_area != 0) else 0


def recall_from_list(test_list, base_poly):
    int_area = 0.0
    for test_poly in test_list:
        int_area += intersection_area(test_poly, base_poly)
    return (int_area / base_poly.area) if (base_poly.area != 0) else 0


def precision_and_recall_from_list(test_list, base_poly):
    int_area = 0.0
    test_poly_area = 0.0
    for test_poly in test_list:
        int_area += intersection_area(test_poly, base_poly)
        test_poly_area += test_poly.area
    p = (int_area / test_poly_area) if (test_poly_area != 0) else 0
    r = (int_area / base_poly.area) if (base_poly.area != 0) else 0
    return [p, r]


def get_img_id_to_name():
    img_id_to_name = {}
    with open('{}image.csv'.format(OUR_DIR), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_id_to_name[row['id']] = row['filename']
    return img_id_to_name


def get_obj_to_img_id():
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
        plt.plot(x, y, linestyle, linewidth=1, color=color, zorder=1)
        if fill_color != "":
            plt.fill_between(x, y, facecolor=fill_color,  linewidth=1, alpha=0.5)
    if invert_y:
        plt.gca().invert_yaxis()
def plot_coords_tmp(obj, color='red', reverse_xy=False, linestyle='-', fill_color="", show=False, invert_y=False):
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
        plt.plot(x, y, linestyle, linewidth=0, color=color, zorder=1)
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
    import sys
    sys.path.append("..")
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


def rescale_factor(objid):
    obj_to_img_id = get_obj_to_img_id()
    img_id_to_name = get_img_id_to_name()
    img_id = obj_to_img_id[objid]
    img_name = img_id_to_name[img_id]
    original_width, original_height = get_size('{}{}.png'.format(
        ORIGINAL_IMG_DIR, img_name))
    new_width, new_height = get_size('{}{}.png'.format(
        COLOR_SEGMENTED_IMG_DIR, img_name))
    return float(original_width / new_width)


def get_unconstrained_vision_tiles(objid, reverse=True, rescale=True):
    obj_to_img_id = get_obj_to_img_id()
    img_id = obj_to_img_id[objid]
    raw_v_tiles = pickle.load(open('{}{}.pkl'.format(VISION_TILE_DIR, img_id)))
    v_tiles = []
    for multi_tile in raw_v_tiles:
        if explain_validity(multi_tile).split("[")[0] == 'Self-intersection':
            multi_tile = multi_tile.buffer(-1e-10)
        if type(multi_tile) != shapely.geometry.MultiPolygon:
            multi_tile = [multi_tile]
        for vtile in multi_tile:
            v_tiles.append(vtile)
    if rescale:
        # rescale vision tiles to match gt and worker bb scale
        v_tiles = [rescale_polygon(vtile, rescale_factor=(1.0 / rescale_factor(objid))) for vtile in v_tiles]
    if reverse:
        # reverse vision tiles to match gt and worker bb scale
        v_tiles = [reverse_poly(vtile) for vtile in v_tiles]
    return v_tiles


def get_baseline_for_obj(
    objid, prec_thresh, w_box_or_tiles=None,
    granularity='box', vtile_constrained=True,
    expand_thresh=0.8, delete_thresh=0.2
):
    v_tiles = get_unconstrained_vision_tiles(objid)
    gt_box = get_gt(objid)
    if w_box_or_tiles is None:
        print 'No reference box, using gt instead'
        reference_box = gt_box
    else:
        reference_box = w_box_or_tiles

    if vtile_constrained:
        return prec_thresh_vision_baseline(v_tiles, reference_box, prec_thresh, granularity=granularity)
    else:
        return worker_vision_hybrid_baseline(
            v_tiles, w_box_or_tiles,
            expand_thresh=expand_thresh, delete_thresh=delete_thresh
        )


def clean_tile(tile):
    tile = tile.buffer(0)
    if explain_validity(tile).split("[")[0] == 'Self-intersection':
        tile = tile.buffer(-1e-10)
    x_locs, y_locs = tile.exterior.xy
    x_locs = [round(x, 3) for x in x_locs]
    y_locs = [round(y, 3) for y in y_locs]
    return tile


def worker_vision_hybrid_baseline(v_tiles, w_tiles, expand_thresh=0.8, delete_thresh=0.2):
    # iterate over worker tiles and vision tiles
    # if a SMALL vision tile is largely covered, then expand to include the rest of it
    # if a LARGE vision tile is only slightly covered, then contract to exclude the overlap
    final_tiles = []

    # index worker tiles
    indexed_w_tiles = []
    for wtile in w_tiles:
        indexed_w_tiles.append(wtile)

    # index vision tiles
    indexed_v_tiles = []
    for multi_tile in v_tiles:
        if explain_validity(multi_tile).split("[")[0] == 'Self-intersection':
            multi_tile = multi_tile.buffer(-1e-10)
        if type(multi_tile) != shapely.geometry.MultiPolygon:
            multi_tile = [multi_tile]
        for vtile in multi_tile:
            indexed_v_tiles.append(vtile)

    intersection_tiles = []  # intersection tiles
    worker_parent = dict()  # worker_parents[i] = w_tile_id that gave intersection_tiles[i]
    worker_children = defaultdict(list)  # worker_children[w] = intersection_tile_ids that came from indexed_w_tile[w]
    vision_parent = dict()  # vision_parents[i] = v_tile_id that gave intersection_tiles[i]
    vision_children = defaultdict(list)  # vision_children[v] = intersection_tile_ids that came from indexed_v_tile[v]
    workers_intersecting_v = defaultdict(list)  # w_tile_ids that intersect
    w_v_intersection_id = defaultdict(dict)  # w_v_intersection[w][v] = intersection_tile_id
    for v in range(len(indexed_v_tiles)):
        for w in range(len(indexed_w_tiles)):
            vtile = indexed_v_tiles[v]
            wtile = indexed_w_tiles[w]
            intersection_poly = intersection(vtile, wtile)
            if intersection_poly is not None and intersection_poly.area > 0:
                intersection_tiles.append(intersection_poly)
                int_tile_id = len(intersection_tiles) - 1
                worker_parent[int_tile_id] = w
                vision_parent[int_tile_id] = v
                worker_children[w].append(int_tile_id)
                vision_children[v].append(int_tile_id)
                workers_intersecting_v[v].append(w)
                w_v_intersection_id[w][v] = int_tile_id

    for v in range(len(indexed_v_tiles)):
        # either add in portions vision pieces not currently overlapping with any worker
        # or modify worker tiles to delete pieces of vision tile that we want to exclude
        total_int_area = 0.0
        vtile = indexed_v_tiles[v]
        for w in workers_intersecting_v[v]:
            int_tile = intersection_tiles[w_v_intersection_id[w][v]]
            # decide whether to include w or w + delta or w - delta
            total_int_area += int_tile.area

        if (total_int_area / indexed_v_tiles[v].area) > expand_thresh:
            # add in the rest of the v_tile
            remaining_vtile = vtile  # TODO: check that this is a copy by value
            for w in workers_intersecting_v[v]:
                int_tile = intersection_tiles[w_v_intersection_id[w][v]]
                remaining_vtile = poly_diff(remaining_vtile, int_tile)
            final_tiles.append(remaining_vtile)
        elif (total_int_area / indexed_v_tiles[v].area) < delete_thresh:
            # want to exclude this entire vision tile
            # modify worker tiles to remove the intersection pieces
            for w in workers_intersecting_v[v]:
                wtile = indexed_w_tiles[w]
                diff_tile = poly_diff(wtile, vtile)
                # int_tile = intersection_tiles[w_v_intersection_id[w][v]]
                # final_tiles.append(int_tile)
                indexed_w_tiles[w] = diff_tile

    for w in range(len(indexed_w_tiles)):
        # add in all modified worker tiles as is
        final_tiles.append(indexed_w_tiles[w])
    return final_tiles


def prec_thresh_vision_baseline(v_tiles, w_box_or_tiles, prec_thresh, granularity='box'):
    final_box = shapely.geometry.Polygon()
    if granularity == 'box':
        w_box = w_box_or_tiles
        if w_box is not None:
            # match v_tiles against given w_box
            for multi_tile in v_tiles:
                # if prec_thresh == 0 and (int(intersection_area(w_box, tile)) > 0):
                if explain_validity(multi_tile).split("[")[0] == 'Self-intersection':
                    multi_tile = multi_tile.buffer(-1e-10)
                if type(multi_tile) != shapely.geometry.MultiPolygon:
                    multi_tile = [multi_tile]
                for tile in multi_tile:
                    if prec_thresh == 0 and (int(w_box.intersection(tile).area) > 0):
                        # special case prec >= 0 to run faster
                        # TODO: returning 0.02 area for many tiles when it should be 0?
                        if False:
                            # debugging why everything is intersecting
                            print intersection_area(w_box, tile)
                            plt.figure()
                            plot_coords(w_box, color='blue', fill_color='blue', linestyle='--', reverse_xy=False, show=False, invert=False)
                            visualizeTilesSeparate([tile], reverse_xy=False)
                            plt.show()
                            plt.close()
                        final_box = final_box.union(tile)
                    elif prec_thresh == 1 and w_box.contains(tile):
                        # special case prec >= 0 to run faster
                        final_box = final_box.union(tile)
                    elif (precision(tile, w_box) >= prec_thresh):
                        final_box = final_box.union(tile)
    elif granularity == 'tile':
        num_tiles_picked = 0
        # print 'prec_thresh: ', prec_thresh
        w_tiles = w_box_or_tiles
        for multi_tile in v_tiles:
            if explain_validity(multi_tile).split("[")[0] == 'Self-intersection':
                multi_tile = multi_tile.buffer(-1e-10)
            if type(multi_tile) != shapely.geometry.MultiPolygon:
                multi_tile = [multi_tile]
            for vtile in multi_tile:
                int_area = 0.0  # add up total of intersection of this piece against wtiles
                for wtile in w_tiles:
                    # match v_tiles against given wtile
                    int_area += intersection_area(wtile, vtile)
                # print '(', int_area, (int_area / vtile.area), ') ',
                if (vtile.area > 0 and (int_area / vtile.area) > prec_thresh):
                    final_box = final_box.union(vtile)
                    num_tiles_picked += 1
        # print '\nnum_tiles_picked: ', num_tiles_picked

    return final_box


def overlap(a, b):
    if a.area > b.area:
        larger_area = a.area
    else:
        larger_area = b.area
    return a.intersection(b).area / larger_area


def add_object_to_tiles(tiles, obj):
    if obj == []:
        return
    if obj.is_valid:
        if type(obj) == shapely.geometry.polygon.Polygon and obj.area > 1e-8:
            tiles.append(obj)
        elif type(obj) == shapely.geometry.MultiPolygon or type(obj) == shapely.geometry.collection:
            for region in obj:
                if type(region) != shapely.geometry.LineString and region.area > 1e-8:
                    tiles.append(region)


def uniqify(tiles, overlap_threshold=0.2, SAVE=False, SAVEPATH=None, PLOT=False):
    # TODO: implement
    verified_tiles = []
    overlap_area = 0.0  # rough number
    for tidx in range(len(tiles)):
        t = tiles[tidx]
        t_to_add = t
        # duplicated = False
        verified_tiles_new = verified_tiles[:]
        for vtidx in range(len(verified_tiles)):
            try:
                vt = verified_tiles[vtidx]
            except(IndexError):
                print "last element removed"
            try:
                overlap_score = overlap(vt, t)
                if overlap_score > overlap_threshold:
                # if True:
                    # print "Duplicate tiles: ", tidx, vtidx, overlap_score, vt.area, t.area
                    # duplicated = True
                    # if overlap_score < 0.99:
                    if True:
                        verified_tiles_new.remove(vt)
                        overlap_region = vt.intersection(t)
                        diff_region = vt.difference(overlap_region)
                        add_object_to_tiles(verified_tiles_new, overlap_region)
                        add_object_to_tiles(verified_tiles_new, diff_region)
                        # add_object_to_tiles(verified_tiles_new, t.difference(overlap_region))
                        t_to_add = t_to_add.difference(overlap_region)

                    if PLOT:
                        plt.figure()
                        plt.title("[{0},{1}]{2}".format(tidx, vtidx, overlap_score))
 			if True:
                        #try:
                            plot_coords(vt)
                            plot_coords(t, linestyle='--', color="blue")
                            plot_coords(overlap_region, fill_color="lime")
                        #except(AttributeError):
                        #    print "problem with plotting"
                else:
                    overlap_area += intersection_area(vt, t)
            except(shapely.geos.TopologicalError):
                print "Topological Error", tidx, vtidx
        # if not duplicated:
        #     verified_tiles_new.append(t)
        verified_tiles_new.append(t_to_add)
        verified_tiles = verified_tiles_new[:]
    if SAVE:
        pickle.dump(verified_tiles, open(SAVEPATH, 'w'))

    total_area = sum([v_t.area for v_t in verified_tiles])
    return verified_tiles, overlap_area, total_area


def precision_and_recall_for_potentially_overlapping_list(tiles, GT):
    # DP solution
    # add in tiles one by one
    # maintain intersection_area[i], total_tile_area[i]
    # also maintain all disjoint intersections of existing tiles against GT (potentially more than i-1)
    # intersection_area[i] = int_area(i, GT) - \sum_j (int_area(i, GT_int_j))
    # update disjoint intersections GT_int_j
    # do DP for total_tile_area as well
    # TODO:
    # probably too expensive, abandoning for now
    raise NotImplementedError
