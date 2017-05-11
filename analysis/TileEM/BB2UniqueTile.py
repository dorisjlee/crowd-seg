import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from analysis_toolbox import *
from qualityBaseline import *
from shapely.validation import explain_validity
import shapely
from sample_worker_seeds import sample_specs

img_info,object_tbl,bb_info,hit_info = load_info()

def add_object_to_tiles(tiles, obj):
    if obj == []:
        return

    if type(obj) == shapely.geometry.polygon.Polygon and obj.is_valid and obj.area > 1e-8:
        tiles.append(obj)
    elif ((type(obj) == shapely.geometry.MultiPolygon or type(obj) == shapely.geometry.collection) and obj.is_valid) or type(obj) == list:
        for region in obj:
            if type(region) != shapely.geometry.LineString and region.area > 1e-8:
                tiles.append(region)


def get_intersection_regions(obj1, obj2):
    intersection_regions = []
    if type(obj1) == shapely.geometry.polygon.Polygon and obj1.is_valid:
        obj1 = [obj1]

    if type(obj2) == shapely.geometry.polygon.Polygon and obj2.is_valid:
        obj2 = [obj2]

    for reg1 in obj1:
        for reg2 in obj2:
            int_rs = reg1.intersection(reg2)
            if type(int_rs) != shapely.geometry.MultiPolygon:
                int_rs = [int_rs]
            for int_r in int_rs:
                int_r = int_r.buffer(0)
                if explain_validity(int_r).split("[")[0] == 'Self-intersection':
                    int_r = int_r.buffer(-1e-10)
                add_object_to_tiles(intersection_regions, int_r)
                # intersection_regions.append(int_r)

    return intersection_regions


def get_diff_regions(obj1, obj2):
    diff_regions = []
    if type(obj1) == shapely.geometry.polygon.Polygon and obj1.is_valid:
        obj1 = [obj1]

    if type(obj2) == shapely.geometry.polygon.Polygon and obj2.is_valid:
        obj2 = [obj2]

    for reg1 in obj1:
        diff_1 = reg1
        for reg2 in obj2:
            diff_1 = diff_1.difference(reg2)
        if type(diff_1) != shapely.geometry.MultiPolygon:
            diff_1 = [diff_1]
        for diff in diff_1:
            diff = diff.buffer(0)
            if explain_validity(diff).split("[")[0] == 'Self-intersection':
                diff = diff.buffer(-1e-10)
            add_object_to_tiles(diff_regions, diff)
            # diff_regions.append(diff)

    return diff_regions


def overlap(poly1, poly2):
    if type(poly1) == shapely.geometry.polygon.Polygon:
        poly1 = [poly1]
    if type(poly2) == shapely.geometry.polygon.Polygon:
        poly2 = [poly2]

    inter_area = 0.0
    total_p1_area = sum([a.area for a in poly1])
    total_p2_area = sum([b.area for b in poly2])
    for a in poly1:
        for b in poly2:
            inter_area += intersection_area(a, b)

    # if a.area > b.area:
    #     larger_area = a.area
    # else:
    #     larger_area = b.area

    larger_area = max([total_p1_area, total_p2_area])
    return inter_area / larger_area
def visualizeTilesSeparate(tiles,colorful=True):
#     plt.figure()
    colors=cm.rainbow(np.linspace(0,1,len(tiles)))
    for t,i in zip(tiles,range(len(tiles))):
#         plt.figure()
        if colorful:
            c = colors[i]
        else:
            #c="lime"
            c="blue"
        if type(t)==shapely.geometry.polygon.Polygon:
            # plot_coords(t,color=c,reverse_xy=True,fill_color=c)
            plot_coords(t,color=c,fill_color=c)
            # plot_coords(t,color=c,lw=1)
        elif type(t)==shapely.geometry.MultiPolygon or type(t)==shapely.geometry.collection:
            for region in t:
                if type(t)!=shapely.geometry.LineString:
                    # plot_coords(region,color=c,reverse_xy=True,fill_color=c)
                    plot_coords(region,color=c,fill_color=c)
                    # plot_coords(region,color=c,lw=1)
    #xylocs of the largest tile for estimating the obj size
    xlocs,ylocs = tiles[np.argmax([t.area for t in tiles])].exterior.coords.xy
    # plt.ylim(np.min(ylocs)-50,np.max(ylocs)+50)
    # plt.gca().invert_yaxis()

def create_vtiles(objid,sampleNworkers,random_state,tiles="",PRINT=False,SAVE=False,\
                  tqdm_on=False,EXCLUDE_BBG=True,PLOT=True,load_existing_tiles_from_file=False, overlap_threshold=0.0):
    # Ji_tbl (bb_info) is the set of all workers that annotated object i
    bb_objects = bb_info[bb_info["object_id"]==objid]
    if EXCLUDE_BBG: bb_objects =  bb_objects[bb_objects.worker_id!=3]
    # Sampling Data from Ji table
    if sampleNworkers>0 and sampleNworkers<len(bb_objects):
        bb_objects = bb_objects.sample(n=sampleNworkers,random_state=random_state)
    worker_lst = list(bb_objects.worker_id)
    # Create a list of polygons based on worker BBs
    xylocs = [list(zip(*process_raw_locs([x,y]))) for x,y in zip(bb_objects["x_locs"],bb_objects["y_locs"])]
    BB = []
    for xyloc in xylocs:
        BB.append(Polygon(xyloc).buffer(0))

    #Compute Tiles
    # if load_existing_tiles_from_file:
    #     tiles = pkl.load(open("{0}/tiles{1}.pkl".format(DATA_DIR,objid),'r'))
    #     #worker_lst= pkl.load(open("{0}/worker{1}.pkl".format(DATA_DIR,objid),'r'))
    # elif tiles=="":
    vtiles,overlap_area,total_area=uniqify(BB, overlap_threshold=overlap_threshold, SAVE=False, SAVEPATH=None, PLOT=PLOT)
    if PRINT: print "Overlap ratio:",overlap_area/float(total_area)
    #pkl.dump(vtiles,open("{0}/vtiles{1}.pkl".format(DATA_DIR,objid),'w'))
<<<<<<< HEAD
    return vtiles, BB


def PLOT_fn(step):
    if step > 13 and step < 19:
        return True
    else:
        return False


def uniqify(tiles, overlap_threshold=0.0, SAVE=False, SAVEPATH=None, PLOT=False):
    # TODO: implement
    print "start"
    verified_tiles = []
    overlap_area = 0.0  # rough number

    step = 0

    for tidx in range(len(tiles)):
        print '==================================================================='
        print 'Starting new t'
        t = tiles[tidx]
        t_to_add = t
        # duplicated = False
        verified_tiles_new = verified_tiles[:]
        if PLOT :
            plt.figure()
            plt.title("Before {}".format(tidx))
            if True:
                plot_coords(t, linestyle='--',lw=1, color="red",reverse_xy=True,invert_y=True)
                #print t.area
                if len(verified_tiles)>0:
                    visualizeTilesSeparate(verified_tiles,colorful=False)#, fill_color="lime")
            plt.savefig("{}_before.png".format(tidx))
            plt.close()
        for vtidx in range(len(verified_tiles)):
            print '------------------------------------------------------------------'
            print 'Starting new vt_list'
            vt = verified_tiles[vtidx]
            if type(vt) == shapely.geometry.polygon.Polygon:
                vt_list = [vt]
            for vt in vt_list:
                print 'Next vt in vt_list'
                try:
                    # print "begin geo operations"
                    overlap_score = overlap(vt, t_to_add)
                    if overlap_score >= overlap_threshold:
                    # if True:
                        # print "Duplicate tiles: ", tidx, vtidx, overlap_score, vt.area, t.area
                        # duplicated = True
                        # if overlap_score < 0.99:
                        if True:
                            if PLOT and PLOT_fn(step) and len(verified_tiles_new):
                                print 'Before removing vt'
                                plt.figure()
                                visualizeTilesSeparate(verified_tiles_new, colorful=False)
                                plot_coords(vt, linestyle='--', color='red', lw=2)
                                # plot_coords(t_to_add, linestyle='--', fill_color='orange', hatch='x')
                                plot_coords(t_to_add, linestyle='--', color='orange', lw=2)
                                plt.show()
                                plt.close()

                            print 'Step: ', step
                            step += 1
                            verified_tiles_new.remove(vt)

                            if PLOT and PLOT_fn(step) and len(verified_tiles_new):
                                print 'After removing vt'
                                plt.figure()
                                visualizeTilesSeparate(verified_tiles_new, colorful=False)
                                # plot_coords(t_to_add, linestyle='--', fill_color='orange', hatch='x')
                                plot_coords(t_to_add, linestyle='--', color='orange', lw=2)
                                plt.show()
                                plt.close()

                            print 'Step: ', step
                            step += 1
                            # overlap_region = vt.intersection(t_to_add)
                            overlap_regions = get_intersection_regions(vt, t_to_add)
                            # diff_region = vt.difference(overlap_region)
                            diff_regions = get_diff_regions(vt, overlap_regions)
                            # t_to_add = t_to_add.difference(overlap_region)
                            t_to_add = get_diff_regions(t_to_add, overlap_regions)

                            if PLOT and PLOT_fn(step):
                                print 'Checking overlap, diff, vt, and t_to_add'
                                plt.figure()
                                plot_coords(overlap_regions, linestyle='--', fill_color='black', hatch='-')
                                # plot_coords(t_to_add, linestyle='--', fill_color='orange', hatch='x')
                                plot_coords(t_to_add, linestyle='--', color='orange', lw=2)
                                plot_coords(diff_regions, linestyle='--', fill_color='green', hatch='o')
                                plot_coords(vt, linestyle='--', color='red', lw=2)
                                plt.show()
                                plt.close()

                            if PLOT and PLOT_fn(step) and len(verified_tiles_new):
                                print 'Before adding overlap region'
                                plt.figure()
                                visualizeTilesSeparate(verified_tiles_new, colorful=False)
                                # plot_coords(overlap_region, linestyle='--', fill_color='black', hatch='-')
                                plot_coords(overlap_regions, linestyle='--', fill_color='black', hatch='-')
                                # plot_coords(t_to_add, linestyle='--', fill_color='orange', hatch='x')
                                plot_coords(t_to_add, linestyle='--', color='orange', lw=2)
                                plt.show()
                                plt.close()

                            print 'Step: ', step
                            step += 1
                            # add_object_to_tiles(verified_tiles_new, overlap_region)
                            for ov_r in overlap_regions:
                                add_object_to_tiles(verified_tiles_new, ov_r)

                            if PLOT and PLOT_fn(step):
                                print 'After adding overlap region'
                                plt.figure()
                                visualizeTilesSeparate(verified_tiles_new, colorful=False)
                                # plot_coords(t_to_add, linestyle='--', fill_color='orange', hatch='x')
                                plot_coords(t_to_add, linestyle='--', color='orange', lw=2)
                                plt.show()
                                plt.close()

                            if PLOT and PLOT_fn(step):
                                print 'Before adding diff region'
                                plt.figure()
                                visualizeTilesSeparate(verified_tiles_new, colorful=False)
                                # plot_coords(diff_region, linestyle='--', fill_color='green', hatch='-')
                                plot_coords(diff_regions, linestyle='--', fill_color='green', hatch='o')
                                # plot_coords(t_to_add, linestyle='--', fill_color='orange', hatch='x')
                                plot_coords(t_to_add, linestyle='--', color='orange', lw=2)
                                plt.show()
                                plt.close()

                            print 'Step: ', step
                            step += 1
                            # add_object_to_tiles(verified_tiles_new, diff_region)
                            for df_r in diff_regions:
                                add_object_to_tiles(verified_tiles_new, df_r)

                            if PLOT and PLOT_fn(step):
                                print 'After adding diff region'
                                plt.figure()
                                visualizeTilesSeparate(verified_tiles_new, colorful=False)
                                # plot_coords(t_to_add, linestyle='--', fill_color='orange', hatch='x')
                                plot_coords(t_to_add, linestyle='--', color='orange', lw=2)
                                plt.show()
                                plt.close()

                        if False:
                            plt.figure()
                            plt.title("[{0},{1}]{2}".format(tidx, vtidx, overlap_score))
                            if True:
                            #try:
                                plot_coords(vt, linestyle='--', color='blue', lw=2)
                                plot_coords(t, linestyle='--', color='red', lw=2)
                                # plot_coords(overlap_region, fill_color="lime")
                                # plot_coords(diff_region, fill_color="lime")
                                # plot_coords(t_to_add, fill_color="green")
                                plot_coords(overlap_regions, linestyle='--', fill_color='black', hatch='-')
                                plot_coords(diff_regions, linestyle='--', fill_color='green')
                                plot_coords(t_to_add, linestyle='--', fill_color='orange')
                                plt.show()
                                plt.close()
                            # except(AttributeError):
                            #    print "problem with plotting"
                    else:
                        overlap_area += intersection_area(vt, t)

                    # print "end geo operations"
                except(shapely.geos.TopologicalError):
                    print "Topological Error", tidx, vtidx

        # if not duplicated:
        #     verified_tiles_new.append(t)

        print 'Step: ', step
        step += 1
        if PLOT and PLOT_fn(step) and len(verified_tiles_new):
            print 'Before adding t_to_add'
            plt.figure()
            visualizeTilesSeparate(verified_tiles_new, colorful=False)
            # plot_coords(t_to_add, linestyle='--', fill_color='orange', hatch='x')
            plot_coords(t_to_add, linestyle='--', color='orange', lw=2)
            plt.show()
            plt.close()

        add_object_to_tiles(verified_tiles_new, t_to_add)

        if PLOT and PLOT_fn(step):
            print 'After adding t_to_add'
            plt.figure()
            visualizeTilesSeparate(verified_tiles_new, colorful=False)
            # plot_coords(t_to_add, linestyle='--', fill_color='orange', hatch='-')
            plt.show()
            plt.close()

        verified_tiles = verified_tiles_new[:]
        if PLOT and PLOT_fn(step):
            plt.figure()
            plt.title("After {}".format(tidx))
            if True:
                visualizeTilesSeparate(verified_tiles,colorful=False)#, fill_color="lime")
            plt.savefig("{}_after.png".format(tidx))
            plt.close()
            # plt.show()
            # plt.close()

    if SAVE:
        pickle.dump(verified_tiles, open(SAVEPATH, 'w'))

    total_area = sum([v_t.area for v_t in verified_tiles])
    return verified_tiles, overlap_area, total_area

def plot_coords(obj, color='red', reverse_xy=False, linestyle='-',lw=0, fill_color="", hatch='', show=False, invert_y=False):
    #Plot shapely polygon coord
    if type(obj) != shapely.geometry.MultiPolygon and type(obj) != list:
        obj = [obj]

    for ob in obj:
        if ob.exterior is None:
            print 'Plotting bug: exterior is None (potentially a 0 area tile). Ignoring and continuing...'
            continue
        if reverse_xy:
            x, y = ob.exterior.xy
        else:
            y, x = ob.exterior.xy
        plt.plot(x, y, linestyle, linewidth=lw, color=color, zorder=1)
        if fill_color != "":
            plt.fill_between(x, y, facecolor=fill_color, hatch=hatch, linewidth=lw, alpha=0.5)
    if invert_y:
        plt.gca().invert_yaxis()
def createObjIndicatorMatrix(objid,tiles,worker_lst, PLOT=False,PRINT=False,SAVE=False,EXCLUDE_BBG=True,overlap_threshold=0.8,tile_only=False,tqdm_on=False):

    # Convert set of tiles to indicator matrix for all workers and tiles
    # by checking if the worker's BB contains the tile pieces
    # The indicator matrix is a (N + 1) X M matrix,
    # with first N rows indicator vectors for each annotator and
    # the last row being region sizes
    M = len(tiles)
    N = len(worker_lst)
    if PRINT:
        print "Number of non-overlapping tile regions (M) : ",M
        print "Number of workers (N) : ",N
    indicator_matrix = np.zeros((N+1,M))
    bb_objects = bb_info[bb_info["object_id"]==objid]
    for  wi in range(N):
        worker_id = worker_lst[wi]
        worker_bb_info = bb_objects[bb_objects["worker_id"]==worker_id]
        worker_BB_polygon = Polygon(zip(*process_raw_locs([worker_bb_info["x_locs"].values[0],worker_bb_info["y_locs"].values[0]]))).buffer(0)

        # Check if worker's polygon contains this tile
        for tile_i in range(M):
            tile = tiles[tile_i]
            if worker_BB_polygon.contains(tile.centroid):
                indicator_matrix[wi][tile_i]=1
            else:
                try:
                        tileBB_overlap = tile.intersection(worker_BB_polygon).area/float(tile.area)
                        if tileBB_overlap>=overlap_threshold:
                                indicator_matrix[wi][tile_i]=1
                except(shapely.geos.TopologicalError):
                        pass

    # The last row of the indicator matrix is the tile area
    for tile_i in range(M):
        tile=tiles[tile_i]
        indicator_matrix[-1][tile_i]=tile.area
    # Debug plotting all tiles that have not been voted by workers
    all_unvoted_tiles=np.where(np.sum(indicator_matrix[:-1],axis=0)==0)[0]
    if PRINT:
        print "all unvoted tiles:",all_unvoted_tiles
        print "all unvoted workers:",np.where(np.sum(indicator_matrix,axis=1)==0)[0]
    if PLOT or PRINT:
        print "Object ",objid
        sanity_check(indicator_matrix,PLOT)
    if SAVE:
        pkl.dump(worker_lst,open('{0}/worker{1}.pkl'.format(DATA_DIR,objid),'w'))
        pkl.dump(indicator_matrix,open('{0}/indMat{1}.pkl'.format(DATA_DIR,objid),'w'))
    return worker_lst,tiles,indicator_matrix
def sanity_check(indicator_matrix,PLOT=False):
    print "Check that there are no all-zero rows in indicator matrix:" , len(np.where(np.sum(indicator_matrix,axis=1)==0)[0])==0
    print "Check that there are no all-zero columns in indicator matrix:" , len(np.where(np.sum(indicator_matrix[:-1],axis=0)==0)[0])==0
    if PLOT:
        plt.figure()
        plt.title("Tile Area")
        sorted_indicator_matrix = indicator_matrix[:,indicator_matrix[-1].argsort()]
        plt.semilogy(sorted_indicator_matrix[-1])
        plt.plot(sorted_indicator_matrix[-1])
        plt.figure()
        plt.title("Indicator Matrix")
        #Plot all excluding last row (area)
        #plt.imshow(sorted_indicator_matrix[:-1],cmap="cool",interpolation='none', aspect='auto')
        plt.ylabel("Worker #")
        plt.xlabel("Tile #")
        plt.imshow(indicator_matrix[:-1],cmap="cool",interpolation='none', aspect='auto')
        plt.colorbar()
if __name__=='__main__':
    worker_Nbatches={5:10,10:8,15:6,20:4,25:2,30:1}
    sampleN_lst=worker_Nbatches.keys()
    #sample_lst = sample_specs.keys()[6:]
    #['5workers_rand8', '5workers_rand9', '5workers_rand6', '5workers_rand7', '5workers_rand4', '5workers_rand5', '5workers_rand2', '5workers_rand3', '5workers_rand0', '5workers_rand1', '20worker_rand0', '20worker_rand1', '20worker_rand2', '20worker_rand3', '10workers_rand1', '10workers_rand0', '10workers_rand3', '10workers_rand2', '10workers_rand5', '10workers_rand4', '10workers_rand6', '25worker_rand1', '25worker_rand0', '15workers_rand2', '15workers_rand3', '15workers_rand0', '15workers_rand1', '15workers_rand4', '15workers_rand5', '30worker_rand0']
    #sample_lst = ['5workers_rand8', '5workers_rand9', '5workers_rand6', '5workers_rand7', '5workers_rand4']
    #sample_lst = ['5workers_rand3', '5workers_rand0', '5workers_rand1', '20worker_rand0', '20worker_rand1']
    #sample_lst = ['20worker_rand2', '20worker_rand3', '5workers_rand5', '5workers_rand2']
    #sample_lst = ['15workers_rand1', '15workers_rand4', '15workers_rand5', '30worker_rand0']
    #sample_lst = ['15workers_rand2', '15workers_rand3', '15workers_rand0','15workers_rand3']
    #sample_lst = ['10workers_rand6', '25worker_rand1', '25worker_rand0', '15workers_rand2']
    #sample_lst = ['10workers_rand3', '10workers_rand2', '10workers_rand5', '10workers_rand4']
    #sample_lst = ['20worker_rand2', '20worker_rand3', '10workers_rand1', '10workers_rand0']
    sample_lst = ['5workers_rand4', '5workers_rand5', '5workers_rand2','20worker_rand1']
    # vtiles,BB = create_vtiles(45,10,121,PLOT=True, overlap_threshold=0)
    # vtiles,BB = create_vtiles(10,10,121,PLOT=False, overlap_threshold=0)

    sample = '25worker_rand1'#/vtiles34.pkl
    #sample = '10workers_rand7'
    if True:
    #for sample in sample_lst :
	sampleNworkers=int(sample.split("w")[0])
	print "sampleNworkers:",sampleNworkers
        DATA_DIR='uniqueTiles/'+sample
	if not os.path.isdir(DATA_DIR):
	    print "created: ",sampleNworkers
       	    os.mkdir(DATA_DIR)
	#os.chdir(sample)
	DATA_DIR='uniqueTiles/'+sample
	# Look up seeds
	seed = sample_specs[sample][1]
	print "Seed:",seed
	for objid in range(36,48):#range(1,48):
	    print "Creating unique tiles for ", objid
	    vtiles,worker_lst = create_vtiles(objid,sampleNworkers,seed,PLOT=False, overlap_threshold=0.0)
    	    pkl.dump(vtiles,open("{0}/vtiles{1}.pkl".format(DATA_DIR,objid),'w'))
	    #pkl.dump(workers,open("{0}/workers{1}.pkl".format(DATA_DIR,objid),'w'))
	    print "Creating indicator matrix"
	    createObjIndicatorMatrix(objid,vtiles,worker_lst,PRINT=True,SAVE=True,tqdm_on= True)

    #plt.figure()
    #visualizeTilesSeparate(vtiles,colorful=False)
    #plt.savefig("vtiles.png")
    #plt.close()
