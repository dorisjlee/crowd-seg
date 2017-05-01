# constructs iso-color vision tiles from color-segmented-image

import glob
import csv
import os

BASE_DIR = '/Users/akashds1/Dropbox/CrowdSourcing/Image-Segmentation/'
OUR_DIR = BASE_DIR + 'Vision-stuff/'
INPUT_IMG_DIR = OUR_DIR + 'color-segmented-images/'
OUTPUT_DIR = OUR_DIR + 'vision-tiles/'


def get_size(fname):
    from PIL import Image
    #Open image for computing width and height of image
    im = Image.open(fname)
    width = im.size[0]
    height = im.size[1]
    return width, height


def tiles_from_masked_img(fname):
    # Create masks for single valued tiles (so that they are more disconnected)
    from matplotlib import _cntr as cntr
    from shapely.geometry import Polygon
    import numpy as np
    import Image
    # img=mpimg.imread(fname)
    img = Image.open(fname)
    width, height = get_size(fname)
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

    tiles = []  # list of coordinates of all the tiles extracted
    unique_tile_values = set()
    for x in range(len(mega_mask)):
        for y in range(len(mega_mask[x])):
            unique_tile_values.add(tuple(mega_mask[x][y]))
    unique_tile_values = list(unique_tile_values)
    # print unique_tile_values
    # print np.where(mega_mask==1)
    for tile_value in unique_tile_values[1:]:  # exclude 0
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
                        singly_masked_img[x][y] = 1
                        num_pixels_in_tile += 1
        print 'Num pixels in tile: ', num_pixels_in_tile
        if num_pixels_in_tile < 100:
            continue

        # Extract a set of contours from these masks
        # TODO: will disjoint pieces with the same color go into the same polygon?
        x, y = np.mgrid[:singly_masked_img.shape[0], :singly_masked_img.shape[1]]
        c = cntr.Cntr(x, y, singly_masked_img)
        # trace a contour at z ~= 1
        res = c.trace(0.9)
        #if PLOT: plot_trace_contours(singly_masked_img,res)
        for segment in res:
            if segment.dtype != np.uint8 and len(segment) > 2:
                #Take the transpose of the tile graph polygon because during the tile creation process the xy was flipped
                tile = Polygon(zip(segment[:, 1], segment[:, 0]))
                # print tile.area
                # if tile.area>=1: #FOR DEBUGGING PURPOSES

                # segment = Polygon(segment).buffer(1.04)

                segment = Polygon(segment).buffer(0)

                # buff_segments = segment.buffer(0)
                # for seg in buff_segments:
                #     tiles.append(seg)
                tiles.append(segment)

    return tiles


def process_img(input_img_path, output_tiles_path):
    import pickle
    tiles = tiles_from_masked_img(input_img_path)
    # TODO: currently different segments with same color marked as same tile? fix?
    with open(output_tiles_path, 'w') as fp:
        fp.write(pickle.dumps(tiles))


def get_img_name_to_id():
    img_name_to_id = {}
    with open('{}image.csv'.format(OUR_DIR), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_name_to_id[row['filename']] = row['id']
    return img_name_to_id


def get_tiles_from_all_imgs():
    img_name_to_id = get_img_name_to_id()
    for input_img_path in glob.glob('{}*'.format(INPUT_IMG_DIR)):
        img_name = input_img_path.split('/')[-1].split('.')[0]
        img_id = img_name_to_id[img_name]
        # print img_name, ' ---> ', img_id
        output_tiles_path = OUTPUT_DIR + '{}.pkl'.format(img_id)
        process_img(input_img_path, output_tiles_path)
    # return 0


if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # segment_all()
    get_tiles_from_all_imgs()
