def create_vtiles(objid,sampleNworkers,random_state,tiles="",PRINT=False,SAVE=False,\
                  tqdm_on=False,EXCLUDE_BBG=True,PLOT=True,load_existing_tiles_from_file=False):
    # Ji_tbl (bb_info) is the set of all workers that annotated object i
    bb_objects = bb_info[bb_info["object_id"]==objid]
    if EXCLUDE_BBG: bb_objects =  bb_objects[bb_objects.worker_id!=3]
    # Sampling Data from Ji table
    if sampleNworkers>0 and sampleNworkers<len(bb_objects):
        bb_objects = bb_objects.sample(n=sampleNworkers,random_state=random_state)
    # Create a list of polygons based on worker BBs
    xylocs = [list(zip(*process_raw_locs([x,y]))) for x,y in zip(bb_objects["x_locs"],bb_objects["y_locs"])]
    BB = []
    for xyloc in xylocs:
        BB.append(Polygon(xyloc).buffer(0))

    #Compute Tiles
    if load_existing_tiles_from_file:
        tiles = pkl.load(open("{0}/tiles{1}.pkl".format(DATA_DIR,objid),'r'))
        #worker_lst= pkl.load(open("{0}/worker{1}.pkl".format(DATA_DIR,objid),'r'))
    elif tiles=="":
        tiles = BB2TileExact(objid,BB,tqdm_on=tqdm_on,save_tiles=SAVE)
        vtiles,overlap_area,total_area=uniqify(BB, overlap_threshold=0.2, SAVE=False, SAVEPATH=None, PLOT=PLOT)
        print "Overlap ratio:",overlap_area/float(total_area)
        pkl.dump(vtiles,open("{0}/vtiles{1}.pkl".format(DATA_DIR,objid),'w'))
    return vtiles, BB
def uniqify(tiles, overlap_threshold=0.2, SAVE=False, SAVEPATH=None, PLOT=False):
    # TODO: implement
    print "start"
    verified_tiles = []
    overlap_area = 0.0  # rough number
    for tidx in range(len(tiles)):
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
        for vtidx in range(len(verified_tiles)):
            print "tidx:{}; vtidx{}".format(tidx,vtidx)
            try:
                vt = verified_tiles[vtidx]
            except(IndexError):
                print "last element removed"
            try:
                print "begin geo operations"
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

#                     if PLOT:
#                         plt.figure()
#                         plt.title("[{0},{1}]{2}".format(tidx, vtidx, overlap_score))
#                         if True:
#                         #try:
#                             plot_coords(vt)
#                             plot_coords(t, linestyle='--', color="blue")
#                             plot_coords(overlap_region, fill_color="lime")
                        #except(AttributeError):
                        #    print "problem with plotting"
                else:
                    overlap_area += intersection_area(vt, t)
                    
                print "end geo operations"
            except(shapely.geos.TopologicalError):
                print "Topological Error", tidx, vtidx
        # if not duplicated:
        #     verified_tiles_new.append(t)
        add_object_to_tiles(verified_tiles_new,t_to_add)
        verified_tiles = verified_tiles_new[:]
        if PLOT:
            plt.figure()
            plt.title("After {}".format(tidx))
            if True:
                visualizeTilesSeparate(verified_tiles,colorful=False)#, fill_color="lime")
        
    if SAVE:
        pickle.dump(verified_tiles, open(SAVEPATH, 'w'))

    total_area = sum([v_t.area for v_t in verified_tiles])
    return verified_tiles, overlap_area, total_area

def plot_coords(obj, color='red', reverse_xy=False, linestyle='-',lw=0, fill_color="", show=False, invert_y=False):
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
        plt.plot(x, y, linestyle, linewidth=lw, color=color, zorder=1)
        if fill_color != "":
            plt.fill_between(x, y, facecolor=fill_color,  linewidth=lw, alpha=0.5)
    if invert_y:
        plt.gca().invert_yaxis()

if __main__():
    vtiles,BB = create_vtiles(45,10,121,PLOT=True)
