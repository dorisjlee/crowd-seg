
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import shapely.geometry
import matplotlib.image as mpimg
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import numpy as np
import pickle


def get_gt(objid):
    from qualityBaseline import process_raw_locs
    import pandas as pd
    # objid = 2  # 1756 bag
    # objid = 18  # 480 computer
    my_BBG  = pd.read_csv('my_ground_truth.csv')
    ground_truth_match = my_BBG[my_BBG.object_id==objid]
    x_locs,y_locs =  process_raw_locs([ground_truth_match['y_locs'].iloc[0],ground_truth_match['x_locs'].iloc[0]])

    if objid == 18:
        x_locs = [x / 1.04 for x in x_locs]
        y_locs = [y / 1.04 for y in y_locs]
    BBG = shapely.geometry.Polygon(zip(x_locs,y_locs))
    return BBG


def get_worker_bbs():
    from shapely.validation import explain_validity
    import pandas as pd
    eliminate_self_intersection_bb=True
    bb_info = pd.read_csv('bounding_box.csv',skipfooter=1)
    if eliminate_self_intersection_bb:
        for bb in bb_info.iterrows():
            bb=bb[1]
            xloc,yloc =  process_raw_locs([bb['x_locs'],bb['y_locs']]) 
            worker_BB_polygon=Polygon(zip(xloc,yloc))
            if explain_validity(worker_BB_polygon).split('[')[0]=='Self-intersection':
                bb_info.drop(bb.name, inplace=True)

    bb_objects = bb_info[bb_info["object_id"]==objid]
    # Create a list of polygons based on worker BBs 
    xylocs = [list(zip(*process_raw_locs([x,y]))) for x,y in zip(bb_objects["x_locs"],bb_objects["y_locs"])]
    BB = []
    for xyloc in xylocs:
        BB.append(Polygon(xyloc).buffer(0))


def plot_coords(obj,color='red',reverse_xy=False,linestyle='-',fill_color="", show=False, invert=False):
    #Plot shapely polygon coord 
    if type(obj)!=shapely.geometry.MultiPolygon:
        obj=[obj]
    for ob in obj: 
        if reverse_xy:
            x,y = ob.exterior.xy
        else:
            y,x = ob.exterior.xy
        plt.plot(x, y, linestyle, color=color, zorder=1)
        if fill_color!="": plt.fill_between(x, y , facecolor=fill_color,color='none', alpha=0.5)
    if invert: plt.gca().invert_yaxis()
    # if show: plt.show()


def visualizeTilesSeparate(tiles, reverse_xy=True, colorful=True, savename=None):
    # plt.figure()
    colors=cm.rainbow(np.linspace(0,1,len(tiles)))
    for t,i in zip(tiles,range(len(tiles))): 
#         plt.figure()
        if colorful: 
            c = colors[i]
        else: 
            c="lime"
        if type(t)==shapely.geometry.polygon.Polygon:
            plot_coords(t,color=c,reverse_xy=reverse_xy,fill_color=c)
        elif type(t)==shapely.geometry.MultiPolygon or type(t)==shapely.geometry.collection:
            for region in t:
                
                if type(t)!=shapely.geometry.LineString:
                    plot_coords(region,color=c,reverse_xy=reverse_xy,fill_color=c)
    if reverse_xy:
        #xylocs of the largest tile for estimating the obj size
        xlocs,ylocs = tiles[np.argmax([t.area for t in tiles])].exterior.coords.xy
        # plt.ylim(np.min(ylocs)-50,np.max(ylocs)+50)
    plt.gca().invert_yaxis()
    # if savename:
    #     plt.savefig(savename)
    # else:
    #     plt.show()


def get_size(fname):
    from PIL import Image
    #Open image for computing width and height of image
    im = Image.open(fname)
    width = im.size[0]
    height = im.size[1]
    return width, height


def rescale_img(basewidth, img_path):
    import PIL
    img = Image.open(img_path)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    img.save(img_path)


def tiles_from_masked_img(fname):
    # Create masks for single valued tiles (so that they are more disconnected)
    from matplotlib import _cntr as cntr
    import Image
    # img=mpimg.imread(fname)
    img=Image.open(fname)
    width,height = get_size(fname)
    # mega_mask = np.zeros((height,width))
    # mega_mask = list(img.getdata())
    mega_mask = np.array(img)

    print len(mega_mask), len(mega_mask[0]), len(mega_mask[0][0])

    # exit()
    # print mega_mask

    # plt.figure()
    # plt.imshow(mega_mask,interpolation="none")#,cmap="rainbow")
    # plt.colorbar()
    # plt.show()

    tiles = [] # list of coordinates of all the tiles extracted
    unique_tile_values = set()
    for x in range(len(mega_mask)):
        for y in range(len(mega_mask[x])):
            unique_tile_values.add(tuple(mega_mask[x][y]))
    unique_tile_values = list(unique_tile_values)
    # print unique_tile_values
    # print np.where(mega_mask==1)
    for tile_value in unique_tile_values[1:]: #exclude 0
        # print tile_value
        singly_masked_img = np.zeros((len(mega_mask), len(mega_mask[0])))
        # print len(singly_masked_img), len(singly_masked_img[0])
        num_pixels_in_tile = 0
        # for x,y in zip(*np.where(mega_mask==tile_value)):
        #     singly_masked_img[x][y]=1
        #     num_pixels_in_tile+=1
        for x in range(len(mega_mask)):
            for y in range(len(mega_mask[x])):
                if tuple(mega_mask[x][y]) == tuple(tile_value):
                        singly_masked_img[x][y]=1
                        num_pixels_in_tile+=1
        print 'Num pixels in tile: ', num_pixels_in_tile
        if num_pixels_in_tile < 100:
            continue
        #Extract a set of contours from these masks


        # TODO: currently will put everything, even disjoint pieces with completely different colors into the same polygon
        # FIGURE OUT HOW TO SEPARATE THEM

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

                # segment = Polygon(segment).buffer(1.04)

                segment = Polygon(segment).buffer(0.04)

                # buff_segments = segment.buffer(0)
                # for seg in buff_segments:
                #     tiles.append(seg)
                tiles.append(segment)

    return tiles


def process_img(input_img_path, output_img_path, output_tiles_path):
    tiles = tiles_from_masked_img(input_img_path)
    # TODO: currently different segments with same color marked as same tile? fix?
    with open(output_tiles_path, 'w') as fp:
        fp.write(pickle.dumps(tiles))
    visualizeTilesSeparate(tiles, reverse_xy=False, colorful=True, savename=output_img_path)


def jaccard(polygon1, polygon2):
    # returns area-based jaccard similarity score of two polygons
    intersection_poly = None
    union_poly = None
    try:
        try:
            intersection_poly = polygon1.intersection(polygon2)
        except:
            try:
                intersection_poly = polygon1.buffer(0).intersection(polygon2)
            except:
                intersection_poly = polygon1.buffer(1e-10).intersection(polygon2)
    except:
        print 'intersection failed, tile area: ', polygon1.area
        # visualizeTilesSeparate([polygon1])
        # exit()
    try:
        union_poly = polygon1.union(polygon2)
    except:
        print 'union failed'
    int_area = intersection_poly.area if intersection_poly else 0
    union_area = union_poly.area if union_poly else 0
    # if union_area > 0:
    #     print (int_area / union_area)
    return (int_area / union_area) if (union_area != 0) else 0


def precision(test_poly, base_poly):
    # returns area-based precision score of two polygons
    intersection_poly = None
    try:
        try:
            intersection_poly = test_poly.intersection(base_poly)
        except:
            try:
                intersection_poly = test_poly.buffer(0).intersection(base_poly)
            except:
                intersection_poly = test_poly.buffer(1e-10).intersection(base_poly)
    except:
        print 'intersection failed, tile area: ', test_poly.area

    int_area = intersection_poly.area if intersection_poly else 0
    return (int_area / test_poly.area) if (test_poly.area != 0) else 0


def vision_baseline1(v_tiles, w_box=None, w_tiles=None, ind_matrix=None):
    # takes vision tiles, worker tiles and indicator matrix
    # (1a) pick highest voted / highest area worker tiles  OR
    # (1b) given one worker bounding box
    # (2) for each w_tile / for the w_box, find v_tiles that are either
    #   (a) high overlap (jaccard similarity), (b) or (mostly) contained in the w_tile
    # (3) union those v_tiles, and return as output
    # raise NotImplementedError
    final_box = Polygon()
    if w_box is not None:
        # match v_tiles against given w_box
        for tile in v_tiles:
            # if (jaccard(tile, w_box) > 0.5 or w_box.contains(tile)):
            # if jaccard(tile, w_box) > 0.5:
            # if w_box.contains(tile):
            if (precision(tile, w_box) > 0.5 or w_box.contains(tile)):
            # if precision(tile, w_box) > 0.9:
                # try:
                #     print '----------------------------------------------------'
                #     print "Contained tile area: ", tile.area
                #     print "Intersection area: ", tile.intersection(w_box).area
                #     print 'Union area: ', tile.union(w_box).area
                #     print 'precision(tile, w_box): ', precision(tile, w_box)
                #     print '----------------------------------------------------'
                # except:
                #     print "Failed w_box.intersection(tile)"
                #     pass
                final_box = final_box.union(tile)
    elif w_tiles is not None and ind_matrix is not None:
        area_thresh = 10
        vote_threshold = 5
        for i, w_tile in enumerate(w_tiles):
            if (w_tile.area < area_thresh or sum(ind_matrix[i]) < vote_threshold):
                continue
            for v_tile in v_tiles:
                if (jaccard(v_tile, w_tiles) > 0.8 or w_tiles.contains(v_tile)):
                    final_box = final_box.union(v_tile)

    return final_box


def combine_worker_and_vision_tiles(img_dir, img_name, out_dir):
    # worker_tiles_file = img_dir + img_name + "_worker_tiles.pkl"
    vision_tiles_file = img_dir + img_name + "_vision_tiles.pkl"
    # w_tiles = pickle.load(open(worker_tiles_file))
    v_tiles = pickle.load(open(vision_tiles_file))

    # just re-draw worker and vision tiles
    # visualizeTilesSeparate(w_tiles, reverse_xy=True, colorful=True, savename=(out_dir + img_name + "_worker.png"))
    # visualizeTilesSeparate(v_tiles, reverse_xy=False, colorful=True, savename=(out_dir + img_name + "_vision.png"))
    # visualizeTilesSeparate(v_tiles, reverse_xy=False, colorful=True)
    # plt.show()

    # identify related tiles by centroid distance and overlap
    # merge / split / fix tiles
    # TODO

    # compute some vision baseline BBs
    # worker_box_file = img_dir + img_name + "_worker_boxes.pkl"
    # w_box = pickle.load(open(worker_box_file))[0]
    # ind_matrix_file = img_dir + img_name + "_indicator_matrix.pkl"
    # ind_matrix = pickle.load(open(ind_matrix_file))

    objid = 2 if '1756' in img_name else 18 if '480' in img_name else ''
    GTBB = get_gt(objid)

    #############################################
    ####### visualize vision_tiles and gt #######
    #############################################
    plt.figure()
    plot_coords(GTBB, color='blue', linestyle='--', reverse_xy=False, show=False, invert=False)
    visualizeTilesSeparate(v_tiles, reverse_xy=False)
    # plt.gca().invert_yaxis()
    # plt.show()
    plt.savefig("{}111.png".format(out_dir))
    plt.close()

    ##############################################
    # compute + visualize vision_baseline and gt #
    ##############################################
    baseline_box1 = vision_baseline1(v_tiles, w_box=GTBB)

    print 'GT area: ', GTBB.area, ' Vision area: ', baseline_box1.area
    plt.figure()
    plot_coords(GTBB, color='blue', fill_color='blue', linestyle='--', reverse_xy=False, show=False, invert=False)
    visualizeTilesSeparate([baseline_box1], reverse_xy=False)
    # plt.gca().invert_yaxis()
    # plt.show()
    plt.savefig("{}222.png".format(out_dir))
    plt.close()

    with open(out_dir + "_base1.pkl", 'w') as fp:
        fp.write(pickle.dumps(baseline_box1))


if __name__ == '__main__':
    img_dir = '/Users/akashds1/Dropbox/CrowdSourcing/Image Segmentation/images/'
    # img_name = "1756"
    img_name = "480"
    input_img_path = img_dir + img_name + ".png"
    output_img_path = img_dir + img_name + "_vision.png"
    output_tiles_path = img_dir + img_name + "_vision_tiles.pkl"

    # rescale_img(500, img_dir + "480.png")

    # process_img(input_img_path, output_img_path, output_tiles_path)

    out_dir = '/Users/akashds1/Dropbox/CrowdSourcing/Image Segmentation/vision/'
    combine_worker_and_vision_tiles(img_dir, img_name, out_dir)

    # get_worker_bbs()
    # visualizeTilesSeparate([get_gt(), get_gt()])
