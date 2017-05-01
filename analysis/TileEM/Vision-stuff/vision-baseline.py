# visualizing vision tiles and worker / ground truth boxes
# constructing vision baseline using vision tiles and worker / ground truth boxes

import shapely
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pickle
# import glob
import csv
import os

BASE_DIR = '/Users/akashds1/Dropbox/CrowdSourcing/Image-Segmentation/'
OUR_DIR = BASE_DIR + 'Vision-stuff/'
COLOR_SEGMENTED_IMG_DIR = OUR_DIR + 'color-segmented-images/'
VISION_TILE_DIR = OUR_DIR + 'vision-tiles/'
ORIGINAL_IMG_DIR = OUR_DIR + 'COCO/'
VISION_BASELINE_DIR = OUR_DIR + 'vision-baseline/'


def get_size(fname):
    from PIL import Image
    #Open image for computing width and height of image
    im = Image.open(fname)
    width = im.size[0]
    height = im.size[1]
    return width, height


def plot_coords(obj, color='red', reverse_xy=False, linestyle='-', fill_color="", show=False, invert=False):
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
    if invert:
        plt.gca().invert_yaxis()
    # if show: plt.show()


def visualizeTilesSeparate(tiles, reverse_xy=False, colorful=True, savename=None, invert_y=True):
    # plt.figure()
    colors = cm.rainbow(np.linspace(0, 1, len(tiles)))
    for t, i in zip(tiles, range(len(tiles))):
        # plt.figure()
        if colorful:
            c = colors[i]
        else:
            c = "lime"
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


def get_gt(objid, rescale_factor=1):
    from qualityBaseline import process_raw_locs
    import pandas as pd
    # objid = 2  # 1756 bag
    # objid = 18  # 480 computer
    # my_BBG = pd.read_csv('{}my_ground_truth.csv'.format(OUR_DIR))
    # ground_truth_match = my_BBG[my_BBG.object_id == objid]
    # x_locs, y_locs = process_raw_locs([ground_truth_match['y_locs'].iloc[0], ground_truth_match['x_locs'].iloc[0]])

    # x_locs = [x / rescale_factor for x in x_locs]
    # y_locs = [y / rescale_factor for y in y_locs]
    # BBG = shapely.geometry.Polygon(zip(x_locs, y_locs))
    # return BBG

    return get_polygon_from_csv('{}my_ground_truth.csv'.format(OUR_DIR), objid, reverse=True, rescale_factor=rescale_factor)


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


def prec_thresh_vision_baseline(v_tiles, w_box, prec_thresh):
    from shapely.validation import explain_validity
    final_box = Polygon()
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
    return final_box


def get_obj_to_img_id():
    obj_to_img_id = {}
    with open('{}object.csv'.format(OUR_DIR), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            obj_to_img_id[int(row['id'])] = row['image_id']
    return obj_to_img_id


def get_img_id_to_name():
    img_id_to_name = {}
    with open('{}image.csv'.format(OUR_DIR), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_id_to_name[row['id']] = row['filename']
    return img_id_to_name


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


def get_unconstrained_vision_tiles(objid):
    obj_to_img_id = get_obj_to_img_id()
    img_id = obj_to_img_id[objid]
    v_tiles = pickle.load(open('{}{}.pkl'.format(VISION_TILE_DIR, img_id)))
    return v_tiles


def get_baseline_for_obj_from_box(objid, prec_thresh, w_box=None):
    obj_to_img_id = get_obj_to_img_id()
    img_id = obj_to_img_id[objid]
    v_tiles = pickle.load(open('{}{}.pkl'.format(VISION_TILE_DIR, img_id)))
    gt_box = get_gt(objid, rescale_factor(objid))
    return prec_thresh_vision_baseline(v_tiles, (w_box or gt_box), prec_thresh)


def get_baseline_for_obj_from_tiles(objid, prec_thresh, w_tiles):
    obj_to_img_id = get_obj_to_img_id()
    img_id = obj_to_img_id[objid]
    v_tiles = pickle.load(open('{}{}.pkl'.format(VISION_TILE_DIR, img_id)))
    raise NotImplementedError
    # gt_box = get_gt(objid, rescale_factor(objid))
    # return prec_thresh_vision_baseline(v_tiles, (w_box or gt_box), prec_thresh)


def visualize_and_compare_baseline_with_gt(objid, prec_thresh, outdir):
    baseline_box = get_baseline_for_obj_from_box(objid, prec_thresh)
    plt.figure()
    GTBB = get_gt(objid, rescale_factor(objid))
    plot_coords(GTBB, color='blue', fill_color='blue', linestyle='--', reverse_xy=False, show=False, invert=False)
    visualizeTilesSeparate([baseline_box], reverse_xy=False)
    # plt.gca().invert_yaxis()
    # plt.show()
    plt.savefig("{}{}.png".format(outdir, objid))
    plt.close()
    return baseline_box


def visualize_and_compare_baseline_with_box(objid, prec_thresh, outdir, compare_box_or_tiles, granularity='box', plot_stuff=True):
    if granularity == 'box':
        baseline_box = get_baseline_for_obj_from_box(objid, prec_thresh, compare_box_or_tiles)
    compare_box_or_tiles = rescale_polygon(compare_box_or_tiles, rescale_factor(objid))
    GTBB = get_gt(objid, rescale_factor(objid))

    if plot_stuff:
        plt.figure()
        plot_coords(GTBB, color='red', fill_color='red', linestyle='--', reverse_xy=False, show=False, invert=False)
        if granularity == 'box':
            plot_coords(compare_box_or_tiles, color='blue', linestyle='--', reverse_xy=False, show=False, invert=False)
        if granularity == 'tiles':
            visualizeTilesSeparate(compare_box_or_tiles, colorful=False, reverse_xy=False)
        visualizeTilesSeparate([baseline_box], reverse_xy=False)
        # plt.gca().invert_yaxis()
        # plt.show()
        plt.savefig("{}{}.png".format(outdir, objid))
        plt.close()

    return baseline_box


def vision_against_gt_box():
    obj_to_img_id = get_obj_to_img_id()
    for prec_thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        print '--------------------------------------------------------'
        print 'Precision threshold: ', prec_thresh
        print '--------------------------------------------------------'
        outdir = VISION_BASELINE_DIR + 'prec>=' + str(int(100*prec_thresh)) + '/'
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        with open('{}PR.csv'.format(outdir), 'w') as csvfile:
            fieldnames = ['objid', 'precision', 'recall']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for objid in obj_to_img_id:
                # if objid != 1:
                #     continue
                print 'Doing object ', objid
                baseline_box = visualize_and_compare_baseline_with_gt(objid, prec_thresh, outdir)
                GTBB = get_gt(objid, rescale_factor(objid))
                with open('{}vision_baseline_{}.pkl'.format(outdir, objid), 'w') as fp:
                    fp.write(pickle.dumps(baseline_box))
                p = precision(baseline_box, GTBB)
                r = recall(baseline_box, GTBB)
                writer.writerow(
                    {
                        'objid': objid,
                        'precision': p,
                        'recall': r
                    }
                )
        print '--------------------------------------------------------'


def vision_against_other(indir, final_out_file_prefix, granularity='box'):
    import glob
    obj_to_img_id = get_obj_to_img_id()
    for prec_thresh in [0.5]:
        print '--------------------------------------------------------'
        print 'Precision threshold: ', prec_thresh
        print '--------------------------------------------------------'
        for sample_data_dir in glob.glob('{}/*/'.format(indir)):
            # if (indir == 'VisionGTComparisons'):
            #     final_out_file_prefix = 'MVT'
            print 'Running sample data dir: ', sample_data_dir
            outdir = sample_data_dir + final_out_file_prefix + '/' + granularity + '/' + 'prec>=' + str(int(100*prec_thresh)) + '/'
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
            with open('{}PR.csv'.format(outdir), 'w') as csvfile:
                fieldnames = ['objid', 'precision', 'recall']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for objid in obj_to_img_id:
                    print 'Doing object ', objid
                    try:
                        if final_out_file_prefix in ['best_Area_Ratio_BBs', 'best_Num_Points_BBs']:
                            filepath = '{}{}.csv'.format(OUR_DIR+sample_data_dir, final_out_file_prefix)
                            print 'Getting data from: ', filepath
                            final_polygon = get_polygon_from_csv(filepath, objid, reverse=True, rescale_factor=rescale_factor(objid))
                        else:
                            final_box_filename = '{}{}{}.pkl'.format(OUR_DIR+sample_data_dir, final_out_file_prefix, objid)
                            print 'Getting data from: ', final_box_filename
                            PolyData = pickle.load(open(final_box_filename))
                            if final_out_file_prefix == 'Tstar_obj':
                                final_polygon = PolyData[-1][0]
                            elif final_out_file_prefix == 'MVT':
                                final_polygon = PolyData
                                if len(final_polygon[1]) > 0:
                                    print 'Problematic MVT. Skipping...'
                                    continue
                                else:
                                    final_polygon = final_polygon[0]
                    except:
                        print 'Failed to load ', final_out_file_prefix, ' ', objid
                        continue

                    # print type(final_polygon)
                    # print len(final_polygon)
                    # continue

                    # if (indir == 'VisionGTComparisons'):
                    #     final_polygon = PolyData
                    #     print len(final_polygon)
                    #     continue
                    baseline_box = visualize_and_compare_baseline_with_box(objid, prec_thresh, outdir, final_polygon)
                    GTBB = get_gt(objid, rescale_factor(objid))
                    with open('{}vision_baseline_{}.pkl'.format(outdir, objid), 'w') as fp:
                        fp.write(pickle.dumps(baseline_box))
                    p = precision(baseline_box, GTBB)
                    r = recall(baseline_box, GTBB)
                    writer.writerow(
                        {
                            'objid': objid,
                            'precision': p,
                            'recall': r
                        }
                    )
            print '--------------------------------------------------------'


def visualize_hybrid_potential(indir, objid):
    # plot overlay of TileEM tiles + GT + vision tiles
    indir += ('' if indir[-1] == '/' else '/')
    try:
        tile_from_id = pickle.load(open('{}vtiles{}.pkl'.format(indir, objid)))
        T_idx = pickle.load(open('{}Tstar_idx_obj{}.pkl'.format(indir, objid)))
    except:
        print 'Faied to load data. Skipping.'
        return
    tile_id_list = T_idx[-1]

    rf = rescale_factor(objid)
    w_tiles = [rescale_polygon(tile_from_id[tidx], rf) for tidx in tile_id_list]
    vision_tiles = get_unconstrained_vision_tiles(objid)
    GTBB = get_gt(objid, rf)

    plt.figure()
    plot_coords(GTBB, color='red', fill_color='red', linestyle='--', reverse_xy=False, show=False, invert=False)
    for wtile in w_tiles:
        plot_coords(wtile, color='blue', fill_color='blue', linestyle='--', reverse_xy=True, show=False, invert=False)
    # visualizeTilesSeparate(w_tiles, colorful=True, reverse_xy=True, invert_y=False)
    for vtile in vision_tiles:
        plot_coords(vtile, color='black', reverse_xy=False, show=False, invert=False)
    plt.gca().invert_yaxis()
    # plt.show()

    outdir = indir + 'hybrid_potential/'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    plt.savefig("{}{}.png".format(outdir, objid))
    plt.close()


def visualize_all_hybrid_potentials(indir):
    obj_to_img_id = get_obj_to_img_id()
    for objid in obj_to_img_id:
        print 'Doing obj ', objid
        visualize_hybrid_potential(indir, objid)


def hybrid_soln(worker_tiles, v_tiles, prec_thresh):
    # takes vision tiles, final solution worker tiles
    # (1) for each w_tile find v_tiles that are either
    #   (a) high overlap (jaccard similarity), (b) or (mostly) contained in the w_tile
    # (2) union those v_tiles, and return as output
    from shapely.validation import explain_validity
    final_box = Polygon()
    for w_tile in worker_tiles:
        # match v_tiles against given w_tile
        for multi_tile in v_tiles:
            # if prec_thresh == 0 and (int(intersection_area(w_box, tile)) > 0):
            if explain_validity(multi_tile).split("[")[0] == 'Self-intersection':
                multi_tile = multi_tile.buffer(-1e-10)
            if type(multi_tile) != shapely.geometry.MultiPolygon:
                multi_tile = [multi_tile]
            for tile in multi_tile:
                if prec_thresh == 0 and (int(w_tile.intersection(tile).area) > 0):
                    # special case prec >= 0 to run faster
                    # TODO: returning 0.02 area for many tiles when it should be 0?
                    final_box = final_box.union(tile)
                elif prec_thresh == 1 and w_tile.contains(tile):
                    # special case prec >= 0 to run faster
                    final_box = final_box.union(tile)
                elif (precision(tile, w_tile) >= prec_thresh):
                    final_box = final_box.union(tile)
    return final_box


if __name__ == '__main__':
    if not os.path.isdir(VISION_BASELINE_DIR):
        os.makedirs(VISION_BASELINE_DIR)

    # vision_against_gt_box()

    # vision_against_other('VisionGTComparisons', 'Tstar_obj', granularity='box')
    # vision_against_other('VisionGTComparisons', 'MVT', granularity='box')
    # vision_against_other('VisionGTComparisons', 'best_Area_Ratio_BBs', granularity='box')
    vision_against_other('VisionGTComparisons', 'best_Num_Points_BBs', granularity='box')

    # visualize_all_hybrid_potentials('tile-em-output/output_15/')
