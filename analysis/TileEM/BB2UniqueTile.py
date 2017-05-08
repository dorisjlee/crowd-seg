from analysis_toolbox import *
from qualityBaseline import *
import shapely
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
                if int_r.area>0:
                    intersection_regions.append(int_r)

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
            if diff.area>0:
                diff_regions.append(diff)

    return diff_regions
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
        return intersection_poly.area
    except:
        print 'intersection failed'
        return 0

def overlap(poly1, poly2):
    #print "poly1:",poly1
    #print "poly2:",poly2

    if type(poly1) == shapely.geometry.polygon.Polygon:
        poly1 = [poly1]
    if type(poly2) == shapely.geometry.polygon.Polygon:
        poly2 = [poly2]
    inter_area = 0.0
    total_p1_area = sum([a.area for a in poly1])
    total_p2_area = sum([b.area for b in poly2])
    for a in poly1:
        for b in poly2:
            # print "a,b:",a,b
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
    #xlocs,ylocs = tiles[np.argmax([t.area for t in tiles])].exterior.coords.xy
    # plt.ylim(np.min(ylocs)-50,np.max(ylocs)+50)
    # plt.gca().invert_yaxis()

def create_vtiles(objid,sampleNworkers,random_state,tiles="",PRINT=False,SAVE=False,\
                  tqdm_on=False,EXCLUDE_BBG=True,PLOT=True,load_existing_tiles_from_file=False, overlap_threshold=0.2):
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
    # if load_existing_tiles_from_file:
    #     tiles = pkl.load(open("{0}/tiles{1}.pkl".format(DATA_DIR,objid),'r'))
    #     #worker_lst= pkl.load(open("{0}/worker{1}.pkl".format(DATA_DIR,objid),'r'))
    # elif tiles=="":
    vtiles,overlap_area,total_area=uniqify(BB, overlap_threshold=overlap_threshold, SAVE=False, SAVEPATH=None, PLOT=PLOT)
    print "Overlap ratio:",overlap_area/float(total_area)
    #pkl.dump(vtiles,open("{0}/vtiles{1}.pkl".format(DATA_DIR,objid),'w'))
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
            plt.savefig("{}_before.png".format(tidx))
            plt.close()
        for vtidx in range(len(verified_tiles)):
            vt = verified_tiles[vtidx]
            print 'Starting vt_list'
            if type(vt) == shapely.geometry.polygon.Polygon:
                vt_list = [vt]
            for vt in vt_list:
                print 'Starting new vt'
                try:
                    # print "begin geo operations"
                    overlap_score = overlap(vt, t_to_add)
                    if overlap_score > overlap_threshold:
                    # if True:
                        # print "Duplicate tiles: ", tidx, vtidx, overlap_score, vt.area, t.area
                        # duplicated = True
                        # if overlap_score < 0.99:
                        
                        if PLOT and len(verified_tiles_new):
                            print 'Before removing vt'
                            plt.figure()
                            visualizeTilesSeparate(verified_tiles_new, colorful=False)
                            plot_coords(vt, linestyle='--', color='black', lw=1)
                            plot_coords(t_to_add, linestyle='--', fill_color='orange', hatch='-')
                            plt.show()
                            plt.close()
                        try:
                            verified_tiles_new.remove(vt)
                        except(ValueError):
                            print "Pass, did not remove vt since vt not in verfified_tiles_new"
                            pass
                        if PLOT and len(verified_tiles_new):
                            print 'After removing vt'
                            plt.figure()
                            visualizeTilesSeparate(verified_tiles_new, colorful=False)
                            plot_coords(t_to_add, linestyle='--', fill_color='orange', hatch='-')
                            plt.show()
                            plt.close()

                        # overlap_region = vt.intersection(t_to_add)
                        overlap_regions = get_intersection_regions(vt, t_to_add)
                        # diff_region = vt.difference(overlap_region)
                        diff_regions = get_diff_regions(vt, overlap_regions)

                        if PLOT and len(verified_tiles_new):
                            print 'Before adding overlap region'
                            plt.figure()
                            print "visualize tile separate"
                            print verified_tiles_new
                            visualizeTilesSeparate(verified_tiles_new, colorful=False)
                            # plot_coords(overlap_region, linestyle='--', fill_color='black', hatch='-')
                            print" overlap_regions"
                            print overlap_regions
                            plot_coords(overlap_regions, linestyle='--', fill_color='black', hatch='-')
                            print "t_to_add"
                            print t_to_add
                            plot_coords(t_to_add, linestyle='--', fill_color='orange', hatch='-')
                            plt.show()
                            plt.close()

                        # add_object_to_tiles(verified_tiles_new, overlap_region)
                        add_object_to_tiles(verified_tiles_new, overlap_regions)

                        if PLOT:
                            print 'After adding overlap region'
                            plt.figure()
                            visualizeTilesSeparate(verified_tiles_new, colorful=False)
                            plot_coords(t_to_add, linestyle='--', fill_color='orange', hatch='-')
                            plt.show()
                            plt.close()

                        # t_to_add = t_to_add.difference(overlap_region)
                        t_to_add = get_diff_regions(t_to_add, overlap_regions)

                        if PLOT:
                            print 'Before adding diff region'
                            plt.figure()
                            visualizeTilesSeparate(verified_tiles_new, colorful=False)
                            # plot_coords(diff_region, linestyle='--', fill_color='green', hatch='-')
                            plot_coords(diff_regions, linestyle='--', fill_color='green', hatch='-')
                            plot_coords(t_to_add, linestyle='--', fill_color='orange', hatch='-')
                            plt.show()
                            plt.close()

                        # add_object_to_tiles(verified_tiles_new, diff_region)
                        add_object_to_tiles(verified_tiles_new, diff_regions)

                        if PLOT:
                            print 'After adding diff region'
                            plt.figure()
                            visualizeTilesSeparate(verified_tiles_new, colorful=False)
                            plot_coords(t_to_add, linestyle='--', fill_color='orange', hatch='-')
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
        if PLOT and len(verified_tiles_new):
            print 'Before adding t_to_add'
            plt.figure()
            visualizeTilesSeparate(verified_tiles_new, colorful=False)
            plot_coords(t_to_add, linestyle='--', fill_color='orange', hatch='-')
            plt.show()
            plt.close()

        add_object_to_tiles(verified_tiles_new, t_to_add)

        if PLOT:
            print 'After adding t_to_add'
            plt.figure()
            visualizeTilesSeparate(verified_tiles_new, colorful=False)
            # plot_coords(t_to_add, linestyle='--', fill_color='orange', hatch='-')
            plt.show()
            plt.close()

        verified_tiles = verified_tiles_new[:]
        if PLOT:
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
    #print "t1:",type(obj)
    #print "tchecl:",type(obj)== shapely.geometry.collection.GeometryCollection
      
    if type(obj) != shapely.geometry.MultiPolygon and type(obj) != list:
        obj = [obj]
    #print type(obj)
    for ob in obj:
        if type(ob)== shapely.geometry.collection.GeometryCollection: 
            print "Cannot plot Geometry collection"
            print ob.area
            pkl.dump(ob,open("geo_collection.pkl",'w')) #these are zero-area objects, throw them away.
            continue
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

# def plot_coords(obj, color='red', reverse_xy=False, linestyle='-',lw=0, fill_color="", show=False, invert_y=False):
#     #Plot shapely polygon coord
#     if type(obj) != shapely.geometry.MultiPolygon:
#         obj = [obj]

#     for ob in obj:
#         if ob.exterior is None:
#             print 'Plotting bug: exterior is None (potentially a 0 area tile). Ignoring and continuing...'
#             continue
#         if reverse_xy:
#             x, y = ob.exterior.xy
#         else:
#             y, x = ob.exterior.xy
#         plt.plot(x, y, linestyle, linewidth=lw, color=color, zorder=1)
#         if fill_color != "":
#             plt.fill_between(x, y, facecolor=fill_color,  linewidth=lw, alpha=0.5)
#     if invert_y:
#         plt.gca().invert_yaxis()

if __name__=='__main__':
    #vtiles,BB = create_vtiles(45,10,121,PLOT=True)
    # vtiles,BB = create_vtiles(10,10,121,PLOT=True)
    vtiles,BB = create_vtiles(43,10,121, overlap_threshold=0,PLOT=False)
    plt.figure()
    visualizeTilesSeparate(vtiles,colorful=False)
    plt.savefig("vtiles.png")
    plt.close()