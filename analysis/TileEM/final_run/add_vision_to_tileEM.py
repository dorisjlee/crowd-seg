# visualizing vision tiles and worker / ground truth boxes
# constructing vision baseline using vision tiles and worker / ground truth boxes

import pickle
import glob
import os

from poly_utils import *


def calc_all_viz_added_output(indir, adjacency_constraint=True, plot_stuff=False):
    # objects = get_obj_to_img_id().keys()
    for sample_path in glob.glob('{}/*/'.format(indir)):
        print '======================================================'
        sample_name = sample_path.split('/')[-2]
        num_workers = int(sample_name.split('w')[0])
        sample_num = int(sample_name.split('d')[-1])

        # if '/5worker_rand0' not in sample_path:
        #     continue

        adj_path = sample_path + ('' if adjacency_constraint else 'no_') + 'adj/'

        for objpath in glob.glob('{}{}*/'.format(adj_path, 'obj')):
            objid = int(objpath.split('/')[-2].split('j')[1])

            # if objid != 18:
            #     continue

            # print 'Doing object ', objid
            print '-----------------------------------------------------'

            for threshpath in glob.glob('{}{}*/'.format(objpath, 'thresh')):
                thresh = int(threshpath.split('/')[-2].split('h')[2])

                # if thresh != 0:
                #     continue

                max_iter_num = 0  # only do vision for last iter
                for iterpath in glob.glob('{}{}*/'.format(threshpath, 'iter_')):
                    max_iter_num = int(iterpath.split('/')[-2].split('_')[1])

                print 'Doing ', [num_workers, sample_num, objid, thresh, max_iter_num]

                outdir = '{}{}{}/'.format(threshpath, 'iter_', max_iter_num)

                without_vision_pr_file = '{}/pr.json'.format(outdir)
                if not os.path.isfile(without_vision_pr_file):
                    # only do vision stuff if we were able to calculate pr for the final iteration
                    continue

                final_hybrid_tile_filename = '{}/hybrid_final_tiles.pkl'.format(outdir)
                if os.path.isfile(final_hybrid_tile_filename):
                    # don't repeat ones we have already finished
                    continue

                without_vision_tids = pickle.load(open('{}/tid_list.pkl'.format(outdir)))
                all_tiles = pickle.load(open('{}vtiles{}.pkl'.format(sample_path, objid)))
                without_vision_tiles = [all_tiles[wtidx] for wtidx in without_vision_tids]

                hybrid_final_tiles = get_baseline_for_obj(
                    objid,
                    None,  # prec_threshold if doing reference box viz hybrid
                    w_box_or_tiles=without_vision_tiles,
                    granularity='tile',
                    vtile_constrained=False,
                    expand_thresh=0.8,
                    delete_thresh=0.2
                )

                with open(final_hybrid_tile_filename, 'w') as fp:
                    fp.write(pickle.dumps(hybrid_final_tiles))

                if plot_stuff:
                    # sanity check plots
                    GTBB = get_gt(objid)
                    vision_tiles = get_unconstrained_vision_tiles(objid, reverse=True, rescale=True)
                    plt.figure()
                    final_tiles = pickle.load(open(final_hybrid_tile_filename))
                    plot_coords(GTBB, color='red', fill_color='red', reverse_xy=True)
                    for vtile in vision_tiles:
                        plot_coords(vtile, color='black', reverse_xy=True)
                    visualizeTilesSeparate(final_tiles, colorful=False, default_color='blue', reverse_xy=True)
                    for wtile in without_vision_tiles:
                        plot_coords(wtile, color='yellow', reverse_xy=True)
                    plt.savefig('{}/out.png'.format(outdir))
                    plt.close()

            print '-----------------------------------------------------'


if __name__ == '__main__':
    calc_all_viz_added_output(ALL_SAMPLES_DIR, adjacency_constraint=False, plot_stuff=False)
