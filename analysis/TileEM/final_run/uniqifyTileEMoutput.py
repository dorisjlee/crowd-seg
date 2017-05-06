# visualizing vision tiles and worker / ground truth boxes
# constructing vision baseline using vision tiles and worker / ground truth boxes
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import pickle
import json
import glob
import time
import os

from poly_utils import *


def uniqify_all_tileEM_out(indir,  adjacency_constraint='adj', plot_stuff=False,idx=0):
    # objects = get_obj_to_img_id().keys()
    sample_path_lst= glob.glob('{}/*/'.format(indir))[idx:idx+1]
    print "Working on sample_path_lst: " ,sample_path_lst
    for sample_path in sample_path_lst[:1]:
        print '======================================================'
        sample_name = sample_path.split('/')[-2]
        num_workers = int(sample_name.split('w')[0])
        sample_num = int(sample_name.split('d')[-1])

        #if '/10worker_rand0' not in sample_path:
        #    continue

        adj_path = sample_path + ('mvt' if adjacency_constraint == 'mvt' else 'adj' if adjacency_constraint == 'adj' else 'no_adj') + '/'
	if adjacency_constraint == 'mvt':
	    for mvt_file in glob.glob('{}MVT*.pkl'.format(sample_path)):
                objid = int(mvt_file.split('.')[-2].split('T')[-1])
		if not os.path.isdir('{}obj{}/'.format(adj_path, objid)):
		    os.makedirs('{}obj{}/'.format(adj_path, objid))
        
	for objpath in glob.glob('{}{}*/'.format(adj_path, 'obj')):
            objid = int(objpath.split('/')[-2].split('j')[1])

            if objid != 4:
                continue

            # print 'Doing object ', objid
            print '-----------------------------------------------------'
	    if adjacency_constraint == 'mvt':
		if True: 
		    outdir = objpath  # '{}{}{}/'.format(threshpath, 'iter_', iter_num)
		    final_unique_tile_filename = '{}/final_unique_tiles.pkl'.format(outdir)
		    if os.path.isfile(final_unique_tile_filename):
			# don't repeat ones we have already finished
			print "skipped",final_unique_tile_filename
			continue

		    try:
			MVT_tiles = pickle.load(open('{}MVT{}.pkl'.format(sample_path, objid)))
			all_tiles = pickle.load(open('{}vtiles{}.pkl'.format(sample_path, objid)))
			print len(MVT_tids)
			print len(all_tiles)
		    except:
			print 'Failed to load. Skipping.'

		    #MVT_tiles = [all_tiles[tidx] for tidx in MVT_tids]
		    print 'Old area = ', sum([tile.area for tile in MVT_tiles])
		    print 'len(MVT):',len(MVT_tiles)
		    final_unique_tiles, overlap_area, total_area = uniqify(MVT_tiles, overlap_threshold=0.2)
		    print 'New area = ', sum([tile.area for tile in final_unique_tiles])
		    print 'Num tiles: {}, overlap_area: {}, total_area: {}'.format(
			len(final_unique_tiles), overlap_area, total_area
		     )

		    with open(final_unique_tile_filename, 'w') as fp:
			fp.write(pickle.dumps(final_unique_tiles))

		    with open('{}/overlap_stats.json'.format(outdir), 'w') as fp:
			fp.write(json.dumps([overlap_area, total_area]))
		continue            

	    for threshpath in glob.glob('{}{}*/'.format(objpath, 'thresh')):
                thresh = int(threshpath.split('/')[-2].split('h')[2])

                if thresh in [-10,10]:
                    continue

                for iterpath in glob.glob('{}{}*/'.format(threshpath, 'iter_')):

                    start = time.time()
                    iter_num = int(iterpath.split('/')[-2].split('_')[1])

                    if iter_num != 5:
                        continue

                    print 'Doing ', [num_workers, sample_num, objid, thresh, iter_num]

                    outdir = '{}{}{}/'.format(threshpath, 'iter_', iter_num)

                    final_unique_tile_filename = '{}/final_unique_tiles.pkl'.format(outdir)
                    if os.path.isfile(final_unique_tile_filename):
                        # don't repeat ones we have already finished
			print "skipped",final_unique_tile_filename
                        continue

                    try:
                        tileEM_tids = pickle.load(open('{}/tid_list.pkl'.format(outdir)))
                        all_tiles = pickle.load(open('{}vtiles{}.pkl'.format(sample_path, objid)))
			print len(tileEM_tids)
			print len(all_tiles)
                    except:
                        print 'Failed to load. Skipping.'

                    EM_tiles = [all_tiles[tidx] for tidx in tileEM_tids]
                    print 'Old area = ', sum([tile.area for tile in EM_tiles])
		    print 'len(EM):',len(EM_tiles)
                    final_unique_tiles, overlap_area, total_area = uniqify(EM_tiles, overlap_threshold=0.2)
                    print 'New area = ', sum([tile.area for tile in final_unique_tiles])
                    print 'Num tiles: {}, overlap_area: {}, total_area: {}'.format(
                        len(final_unique_tiles), overlap_area, total_area
                     )

                    with open(final_unique_tile_filename, 'w') as fp:
                        fp.write(pickle.dumps(final_unique_tiles))

                    with open('{}/overlap_stats.json'.format(outdir), 'w') as fp:
                        fp.write(json.dumps([overlap_area, total_area]))

                    if plot_stuff:
                        # sanity check plots
                        # GTBB = get_gt(objid)
                        plt.figure()
                        final_tiles = pickle.load(open(final_unique_tile_filename))
                        visualizeTilesSeparate(final_tiles, colorful=False, default_color='blue', reverse_xy=True)
                        plt.savefig('{}/tileEM_after_uniqify_tiles.png'.format(outdir))
                        plt.close()

                        plt.figure()
                        visualizeTilesSeparate(EM_tiles, colorful=False, default_color='blue', reverse_xy=True)
                        plt.savefig('{}/tileEM_before_uniqify_tiles.png'.format(outdir))
                        plt.close()

                    end = time.time()
                    print 'Time taken: ', round(end - start, 1)

            print '-----------------------------------------------------'


if __name__ == '__main__':
    import sys
    #run_num=int(sys.argv[1])
    #print "Run_num:",run_num
    # adjacency_constraint in ['adj', 'no_adj', 'mvt']
    uniqify_all_tileEM_out(ALL_SAMPLES_DIR, adjacency_constraint='mvt', plot_stuff=False)#,idx=run_num)
