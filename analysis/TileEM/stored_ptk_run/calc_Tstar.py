# visualizing vision tiles and worker / ground truth boxes
# constructing vision baseline using vision tiles and worker / ground truth boxes

from shapely.ops import cascaded_union
from matplotlib import pyplot as plt
from collections import defaultdict
import numpy as np
import pickle
import glob
import os

from poly_utils import *


def core(tiles, indMat):
    # In the initial step, we pick T to be the top 5 area-vote score
    # where we combine the area and vote in a 1:5 ratio
    topk = 1
    area = np.array(indMat[-1])
    votes = indMat[:-1].sum(axis=0)
    # norm_area_vote = area / max(area) + (5 * votes) / max(votes)

    # # print area/max(area)
    # # print 5*votes/max(votes)

    # tidx = np.argsort(norm_area_vote)[::-1][:topk]
    # # return join_tiles(tidx, tiles)[0], list(tidx)
    # return tidx

    max_votes = max(votes)
    final_idx = -1
    final_area = -1
    for i in range(len(area)):
        if votes[i] == max_votes and area[i] > final_area:
            final_idx = i
            final_area = area[i]
    return [final_idx]


def adjacent(tileA, tileB):
    return tileA.buffer(0.1).overlaps(tileB.buffer(0.1))


def find_all_tk_in_shell(tiles, current_shell_idx, memoized_adjacency=defaultdict(dict), exclude_idx=[], adjacency_constraint=True):
    # Find all tiles at the shell d=d+1
    # add all tiles adjacent to currentShell front
    # memoized_adjacency[tid1][tid2] = True if tiles[tid1] is adcent to tiles[id2] else False

    filtered_tidxs = list(np.delete(np.arange(len(tiles)), exclude_idx))

    if not adjacency_constraint:
        # no adjacency imposed, exploring all that not explored earlier
        return list(set(filtered_tidxs)), defaultdict(dict)

    new_filtered_tidxs = filtered_tidxs[:]

    adjacent_tkidxs = []
    for ctidx in current_shell_idx:
        for tkidx in filtered_tidxs:
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


def calc_Tstar(tiles, pInT, pNotInT, seed_tile, thresh_param, adjacency_constraint=True):
    tile_ids_processed = [seed_tile]
    output_tile_ids = [seed_tile]
    # Tstar = Polygon()

    current_frontier = [seed_tile]
    memoized_adjacency = defaultdict(dict)
    tiles_to_explore_next, memoized_adjacency = find_all_tk_in_shell(tiles, current_frontier, memoized_adjacency, tile_ids_processed, adjacency_constraint=adjacency_constraint)

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
            if pInT[tid] >= thresh_param + pNotInT[tid]:
                # include tile
                current_frontier.append(tid)
                output_tile_ids.append(tid)
                # Tstar.union(tiles[tid])

        tiles_to_explore_next, memoized_adjacency = find_all_tk_in_shell(tiles, current_frontier, memoized_adjacency, tile_ids_processed,  adjacency_constraint=adjacency_constraint)

    return output_tile_ids


def testing_cores(indir=ALL_SAMPLES_DIR):
    precisions = []
    recalls = []
    for sample_path in glob.glob('{}/*/'.format(indir)):
        # print '======================================================'
        sample_name = sample_path.split('/')[-2]
        # print 'Processing sample ', sample_name
        # if sample_name != '5worker_rand0':
        #     continue
        # for objid in objects:
        for objid in range(1, 48):
            # if objid != 18:
            #     continue
            # print 'Doing object ', objid
            # print '-----------------------------------------------------'
            try:
                tiles = pickle.load(open('{}vtiles{}.pkl'.format(sample_path, objid)))
                indMat = pickle.load(open('{}indMat{}.pkl'.format(sample_path, objid)))
            except:
                continue

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


def calc_all_Tstars(indir, adjacency_constraint=True):
    # objects = get_obj_to_img_id().keys()
    # num_union_succeeded = 0
    # num_union_failed = 0
    for sample_path in glob.glob('{}/*/'.format(indir)):
        print '======================================================'
        sample_name = sample_path.split('/')[-2]
        num_workers = int(sample_name.split('w')[0])
        sample_num = int(sample_name.split('d')[-1])
        # print 'Processing sample ', sample_name
        # if '/10worker_rand' not in sample_path:
        #     continue
        # for objid in objects:
        for objid in range(1, 48):
            # if objid != 18:
            #     continue
            # print 'Doing object ', objid
            print '-----------------------------------------------------'
            try:
                tiles = pickle.load(open('{}vtiles{}.pkl'.format(sample_path, objid)))
                indMat = pickle.load(open('{}indMat{}.pkl'.format(sample_path, objid)))
            except:
                print 'Failed to load ', objid
            seed_tile = core(tiles, indMat)[0]
            for pInT_iter_path in glob.glob('{}pInT_lst_obj{}_iter*.pkl'.format(sample_path, objid)):
                # load all data
                iter_num = pInT_iter_path.split('.')[-2][-1]
                # print 'Doing iter num ', iter_num
                pInT = pickle.load(open(pInT_iter_path))
                pNotInT = pickle.load(open('{}pNotInT_lst_obj{}_iter{}.pkl'.format(sample_path, objid, iter_num)))

                for thresh_param in [-4.0, -2.0, 0.0, 2.0, 4.0]:
                    print 'Doing ', [num_workers, sample_num, objid, thresh_param, iter_num]
                    output_tile_ids = calc_Tstar(tiles, pInT, pNotInT, seed_tile, thresh_param, adjacency_constraint=adjacency_constraint)
                    if len(output_tile_ids) - len(set(output_tile_ids)) > 0:
                        print 'Num duplicates: ', len(output_tile_ids) - len(set(output_tile_ids))
                    outdir = sample_path + ('' if adjacency_constraint else 'no_') + 'adj/' + 'obj{}/'.format(objid) + 'thresh{}/'.format(int(thresh_param * 10)) + 'iter_{}/'.format(iter_num)
                    if not os.path.isdir(outdir):
                        os.makedirs(outdir)
                    with open('{}/tid_list.pkl'.format(outdir), 'w') as fp:
                        fp.write(pickle.dumps(output_tile_ids))

                    # with open('{}/Tstar.pkl'.format(outdir), 'w') as fp:
                    #     final_valid_tiles = [
                    #         clean_tile(tiles[tid]) for tid in output_tile_ids
                    #         # if tiles[tid].is_valid
                    #     ]
                    #     union_succeeded = True
                    #     # try:
                    #     if True:
                    #         Tstar = cascaded_union(final_valid_tiles)
                    #         num_union_succeeded += 1
                    #     # except:
                    #     else:
                    #         union_succeeded = False
                    #         print 'Failed to union'
                    #         num_union_failed += 1
                    #     if union_succeeded:
                    #         fp.write(pickle.dumps(Tstar))

                    # # sanity check plots
                    # plt.figure()
                    # tids = pickle.load(open('{}/tid_list.pkl'.format(outdir)))
                    # final_tiles = [tiles[tid] for tid in tids]
                    # plot_coords(get_gt(objid), color='red', fill_color='red')
                    # visualizeTilesSeparate(final_tiles, colorful=False)
                    # plot_coords(tiles[seed_tile], color='blue', fill_color='blue')
                    # plt.savefig('{}/out.png'.format(outdir))
                    # plt.close()

            print '-----------------------------------------------------'
    print 'Num succeeded: {), num failed = {}'.format(num_union_succeeded, num_union_failed)


def process_all_PR(indir, adjacency_constraint=True):
    import json
    # files_completed_filename = 'files_completed.json'
    # files_completed = []
    # if os.path.isfile(files_completed_filename):
    #     with open(files_completed_filename, 'r') as fp:
    #         for line in fp:
    #             files_completed.append(json.loads(line))

    # files_failed = []
    # if os.path.isfile('files_failed.json'):
    #     # need to manually maintain this file!
    #     with open('files_failed.json', 'r') as fp:
    #         for line in fp:
    #             files_failed.append(json.loads(line))

    for sample_path in glob.glob('{}/*/'.format(indir)):
        sample_name = sample_path.split('/')[-2]
        num_workers = int(sample_name.split('w')[0])
        sample_num = int(sample_name.split('d')[-1])

        adj_path = sample_path + ('' if adjacency_constraint else 'no_') + 'adj/'

        for objpath in glob.glob('{}{}*/'.format(adj_path, 'obj')):
            objid = int(objpath.split('/')[-2].split('j')[1])

            #if objid != 30:
            #    continue

            tiles = pickle.load(open('{}vtiles{}.pkl'.format(sample_path, objid)))
            GTBB = get_gt(objid)

            for threshpath in glob.glob('{}{}*/'.format(objpath, 'thresh')):
                thresh = int(threshpath.split('/')[-2].split('h')[2])
		if thresh in [-10,10]:
                    continue
                for iterpath in glob.glob('{}{}*/'.format(threshpath, 'iter_')):
                    iter_num = int(iterpath.split('/')[-2].split('_')[1])

                    if iter_num != 5:
                        continue
                    # if num_workers in [10, 15]:
                    #     continue
                    # if num_workers == 20 and sample_num < 3:
                    #     continue
                    # if num_workers == 20 and sample_num == 3 and str(objid) < str(30):
                    #     continue
                    if [num_workers, sample_num, objid, thresh] in [
                            [10, 6, 47, -10]
                            # add any that crash
                    ]:
                        # failed for
                        # ['20worker_rand3', 20, 3, 30, 40, 0]
                        # ['20worker_rand3', 20, 3, 30, 40, 1]
                        continue
                    # finished up to ['20worker_rand3', 20, 3, 30, 40, 0]
                    print [num_workers, sample_num, objid, thresh]

                    # USE IF PR ON ORIGINAL TILE-EM OUTPUT
                    # final_tids = pickle.load(open('{}tid_list.pkl'.format(iterpath)))
                    # final_tiles = [tiles[tid] for tid in final_tids]

                    # USE IF PR ON UNIQIFIED TILES
		    if not os.path.isfile('{}final_unique_tiles.pkl'.format(iterpath)):
                        continue
                    final_tiles = pickle.load(open('{}final_unique_tiles.pkl'.format(iterpath)))

                    # compute PR
                    # if False:
                    #     can_compute_pr = True

                    #     stored_pr_file = '{}tile_intersection_areas{}.pkl'.format(sample_path, objid)
                    #     if os.path.isfile(stored_pr_file):
                    #         int_area_dict = pickle.load(open(stored_pr_file))
                    #     else:
                    #         continue
                    #     total_int_area = 0.0
                    #     total_bb_area = 0.0
                    #     for tidx in final_tids:
                    #         if tidx not in int_area_dict:
                    #             can_compute_pr = False
                    #             break
                    #         else:
                    #             total_int_area += int_area_dict[tidx]
                    #             total_bb_area += tiles[tidx].area

                    #     if can_compute_pr:
                    #         p = total_int_area / total_bb_area
                    #         r = total_int_area / GTBB.area
                    #         with open('{}pr.json'.format(iterpath), 'w') as fp:
                    #             fp.write(json.dumps([p, r]))
                    # else:
                    # if os.path.isfile('{}pr.json'.format(iterpath)):
                    #     # already computed
                    #     continue
                    p, r = precision_and_recall_from_list(final_tiles, GTBB)
                    with open('{}pr.json'.format(iterpath), 'w') as fp:
                        fp.write(json.dumps([p, r]))

def compile_prs_into_single(indir, adjacency_constraint=True):
    import json
    import csv
    with open('{}/COMPILED_PR_{}.csv'.format(indir, ('adj' if adjacency_constraint else 'nadj')), 'w') as csvfile:
        fieldnames = ['num_workers', 'sample_num', 'objid', 'thresh', 'iter_num', 'precision', 'recall']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for sample_path in glob.glob('{}/*/'.format(indir)):
            sample_name = sample_path.split('/')[-2]
            num_workers = int(sample_name.split('w')[0])
            sample_num = int(sample_name.split('d')[-1])

            adj_path = sample_path + ('' if adjacency_constraint else 'no_') + 'adj/'

            for objpath in glob.glob('{}{}*/'.format(adj_path, 'obj')):
                objid = int(objpath.split('/')[-2].split('j')[1])
		if objid in [35,40,41]:
		    continue
                for threshpath in glob.glob('{}{}*/'.format(objpath, 'thresh')):
                    thresh = int(threshpath.split('/')[-2].split('h')[2])
		    if thresh in [-10,10]:
			continue
                    for iterpath in glob.glob('{}{}*/'.format(threshpath, 'iter_')):
                        iter_num = int(iterpath.split('/')[-2].split('_')[1])

                        print [num_workers, sample_num, objid, thresh, iter_num]

                        pr_filename = '{}pr.json'.format(iterpath)
                        if os.path.isfile(pr_filename):
                            [p, r] = json.load(open(pr_filename))

                            writer.writerow(
                                {
                                    'num_workers': num_workers,
                                    'sample_num': sample_num,
                                    'objid': objid,
                                    'thresh': thresh,
                                    'iter_num': iter_num,
                                    'precision': p,
                                    'recall': r
                                }
                            )
                        else:
                            print 'No PR computed.'

def pr_for_all_tiles(indir):
    # to sanity check if we have terrible tiles
    # import json
    import csv
    with open('{}/COMPILED_PR_ALL_TILES.csv'.format(indir), 'w') as csvfile:
        fieldnames = ['num_workers', 'sample_num', 'objid', 'precision', 'recall']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for sample_path in glob.glob('{}/*/'.format(indir)):
            sample_name = sample_path.split('/')[-2]
            num_workers = int(sample_name.split('w')[0])
            sample_num = int(sample_name.split('d')[-1])
            for objid in range(48):
                if [num_workers, sample_num, objid] in [
                    [20, 3, 30],
		    [10, 6, 47]
                ]:
                    # failing set
                    continue
                print [sample_name, num_workers, sample_num, objid]
                try:
                    all_tiles = pickle.load(open('{}vtiles{}.pkl'.format(sample_path, objid)))
                    GTBB = get_gt(objid)
                except:
                    continue

                [p, r] = precision_and_recall_from_list(all_tiles, GTBB)
                writer.writerow(
                    {
                        'num_workers': num_workers,
                        'sample_num': sample_num,
                        'objid': objid,
                        'precision': p,
                        'recall': r
                    }
                )


def compute_and_store_intersection_for_tiles_for_all_obj_sample(indir):
    import json
    tiles_completed_filename = 'tiles_completed_intersection.json'
    tiles_completed = []
    if os.path.isfile(tiles_completed_filename):
        with open(tiles_completed_filename, 'r') as fp:
            for line in fp:
                tiles_completed.append(json.loads(line))

    tiles_failed = []
    if os.path.isfile('tiles_failed_intersection.json'):
        # need to manually maintain this file!
        with open('tiles_failed_intersection.json', 'r') as fp:
            for line in fp:
                tiles_failed.append(json.loads(line))

    for sample_path in glob.glob('{}/*/'.format(indir)):
        sample_name = sample_path.split('/')[-2]
        num_workers = int(sample_name.split('w')[0])
        sample_num = int(sample_name.split('d')[-1])

        for vtiles_file in glob.glob('{}vtiles*.pkl'.format(sample_path)):
            objid = int(vtiles_file.split('/')[-1].split('.')[0].split('s')[1])

            print [sample_name, num_workers, sample_num, objid]

            int_area = dict()  # int_area[tild_id] = intersection area with GTBB
            tiles = pickle.load(open(vtiles_file))
            GTBB = get_gt(objid)

            for tidx, tile in enumerate(tiles):
                if [num_workers, sample_num, objid, tidx] in tiles_completed + tiles_failed:
                    continue
                print [num_workers, sample_num, objid, tidx]
                int_area[tidx] = intersection_area(tile, GTBB)
                with open(tiles_completed_filename, 'a') as fp:
                    fp.write('{}\n'.format(json.dumps([num_workers, sample_num, objid, tidx])))

            with open('{}tile_intersection_areas{}.pkl'.format(sample_path, objid), 'w') as fp:
                fp.write(pickle.dumps(int_area))


if __name__ == '__main__':
    # calc_all_Tstars(ALL_SAMPLES_DIR, adjacency_constraint=False)
    # testing_cores()
    process_all_PR(ALL_SAMPLES_DIR, adjacency_constraint=True)
    # compute_and_store_intersection_for_tiles_for_all_obj_sample(ALL_SAMPLES_DIR)

    compile_prs_into_single(ALL_SAMPLES_DIR, adjacency_constraint=True)

    # pr_for_all_tiles(ALL_SAMPLES_DIR)
