import pickle as pkl
import numpy as np
from numpy import mean
from matplotlib import pyplot as plt
from time import time

object_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47]
batch_lst = ['5workers_rand0', '10workers_rand0', '15workers_rand0', '20workers_rand0', '25workers_rand0', '30workers_rand0']

def edge_neighbor_widx(wmap,source):
    x=source[0]
    y=source[1]
    valid_neighbors = []
    w,h = np.shape(wmap)
    if x+1<w:
        valid_neighbors.append((x+1,y))
    if y+1<h:
        valid_neighbors.append((x,y+1))
    if x-1>=0:
        valid_neighbors.append((x-1,y))
    if y-1>=0:
        valid_neighbors.append((x,y-1))
    #return (x+1,y),(x,y+1),(x-1,y),(x,y-1)
    return valid_neighbors

def create_adj_list(sample, objid):
    tiles = pkl.load(open("pixel_em/{}/obj{}/tiles.pkl".format(sample, objid)))
    megamask = pkl.load(open("pixel_em/{}/obj{}/mega_mask.pkl".format(sample, objid)))
    # tile = np.array(list(tiles[0]))

    tile_containing = dict()  # tile_containing[pixel] = tidx
    for tidx in range(len(tiles)):
        # create inverse map
        for pix in tiles[tidx]:
            tile_containing[pix] = tidx

    adjacent_lst = [[] for _t in tiles]
    exclude_lst = [[] for _t in tiles]

    start = time()
    for i in range(1, len(tiles)):  # exclude the big outside tile, we don't really need to know who is connected to the outside tile
        for pix in tiles[i]:
            if pix not in exclude_lst[i]:
                neighbors = edge_neighbor_widx(megamask, pix)
                # get the adjacent tile of from the neighboring pixel
                # adjacent_tile = -1
                for nix in neighbors:
                    old_method = False
                    if old_method:
                        # without using pixel to tile mapping
                        # tiles_exclude_self = [_t for _t in tiles if _t != tiles[i]]  # not checking adjacency against self
                        # for tidx, t in enumerate(tiles_exclude_self):
                        tiles_exclude_self = [j for j in range(len(tiles)) if j != i]
                        for tidx in tiles_exclude_self:
                            if nix in tiles[tidx]:
                                adjacent_lst[i].append(tidx)
                                # adjacent_tile = tidx
                                # store boundary information
                                break
                    else:
                        if tile_containing[nix] != i:
                            adjacent_lst[i].append(tile_containing[nix])
                # exclude_lst[i].append(list(tiles[adjacent_tile])[0])
        adjacent_lst[i] = list(set(adjacent_lst[i]))
    # hard_lst = [1, len(tiles) - 1]
    # for hard_tidx in hard_lst:
    hard_tidx = 0
    for atidx, at in enumerate(adjacent_lst):
        if hard_tidx in at:
            adjacent_lst[hard_tidx].append(atidx)

    end = time()
    print 'Time taken =', end - start

    pkl.dump(adjacent_lst, open("pixel_em/{}/obj{}/adj_list.pkl".format(sample, objid), 'w'))


def test_created_adj_list(sample, objid):
    # TESTING
    adjacent_lst = pkl.load(open("pixel_em/{}/obj{}/adj_list.pkl".format(sample, objid)))

    tiles = pkl.load(open("pixel_em/{}/obj{}/tiles.pkl".format(sample, objid)))
    megamask = pkl.load(open("pixel_em/{}/obj{}/mega_mask.pkl".format(sample, objid)))

    if True:
        # checking the last tile
        pix = next(iter(tiles[-1]))
        neighbors = edge_neighbor_widx(megamask, pix)
        mask = np.zeros_like(megamask)
        for n in neighbors:
            mask[n] = 1
        mask[pix] = 2

        zoom_radius = 5
        x = int(mean(np.where(mask == 1)[1]))
        y = int(mean(np.where(mask == 1)[0]))
        plt.figure()
        plt.imshow(mask)
        plt.xlim(x - zoom_radius, x + zoom_radius)
        plt.ylim(y - zoom_radius, y + zoom_radius)
        plt.colorbar()
        plt.show()
        plt.close()

    if True:
        # checking outside tile
        test_tidx = 0
        mask = np.zeros_like(megamask)

        for tidx in tiles[test_tidx]:
            mask[tidx] = 1
        for neighbor_tile in set(adjacent_lst[test_tidx]):
            for tidx in tiles[neighbor_tile]:
                mask[tidx] = 2
        plt.figure()
        plt.imshow(mask)
        plt.colorbar()
        plt.show()
        plt.close()

    if True:
        for test_tidx in np.random.choice(range(len(adjacent_lst)), 10):
            mask = np.zeros_like(megamask)

            for pidx in tiles[test_tidx]:
                mask[pidx] = 1

            # print tiles[test_tidx]

            for neighbor_tile in set(adjacent_lst[test_tidx]):
                for pidx in tiles[neighbor_tile]:
                    mask[pidx] = 2
                    # print tidx
                # print neighbor_tile

            # from numpy import shape
            # print shape(megamask)
            # print test_tidx, tiles[test_tidx]
            # print np.where(mask == 1)
            x = int(mean(np.where(mask == 1)[1]))
            y = int(mean(np.where(mask == 1)[0]))
            zoom_radius = len(tiles[test_tidx])
            # zoom_radius = 0
            plt.figure()
            plt.imshow(mask)
            plt.xlim(x - zoom_radius, x + zoom_radius)
            plt.ylim(y - zoom_radius, y + zoom_radius)
            plt.colorbar()
            plt.show()
            plt.close()


if __name__ == '__main__':
    #create_adj_list('5workers_rand0', 10)
    #test_created_adj_list('5workers_rand0', 10)

    for batch in batch_lst:
        for objid in object_lst:
            print 'Creating adj list for {} x obj{}'.format(batch, objid)
            create_adj_list(batch, objid)

