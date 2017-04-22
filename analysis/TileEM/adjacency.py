import tqdm as tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
from TileEM_plot_toolbox import *
def colorbar_index(ncolors, cmap):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])
                       for i in xrange(N+1) ]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

def visualizeTilesScore(tiles,tidx_score,INT_Z=True,colorful=True):
    '''
    Given a dictionary consisting of {tidx:score} 
    Plot a heatmap of values
    INT_Z: integet z value colormap
    '''
    plt.figure()
    #colors=cm.rainbow(np.linspace(0,1,len(set(tidx_score.values()))+1))

    score = tidx_score.values()
    norm = matplotlib.colors.Normalize(
        vmin=np.min(score),
        vmax=np.max(score))
    c_m = cm.rainbow

    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])

    for tidx,i in tidx_score.iteritems(): 
        t=tiles[tidx]
        if colorful: 
            c=s_m.to_rgba(i)
        else: 
            c="lime"
        if type(t)==shapely.geometry.polygon.Polygon:
            plot_coords(t,color=c,reverse_xy=True,fill_color=c)
        elif type(t)==shapely.geometry.MultiPolygon or type(t)==shapely.geometry.collection:
            for region in t:
                if type(t)!=shapely.geometry.LineString:
                    plot_coords(region,color=c,reverse_xy=True,fill_color=c)

    if INT_Z:
        colorbar_index(ncolors=len(set(score)), cmap=c_m)
    else:
        plt.colorbar(s_m)
    #xylocs of the largest tile for estimating the obj size
    xlocs,ylocs = tiles[np.argmax([t.area for t in tiles])].exterior.coords.xy
    plt.ylim(np.min(ylocs)-50,np.max(ylocs)+50)
    plt.gca().invert_yaxis()
def adjacent(tileA,tileB):
    return tileA.buffer(0.1).overlaps(tileB.buffer(0.1))
def compute_adjacency_score(objid,CHECK=True,SAVE=True):
    tiles = pkl.load(open("vtiles{}.pkl".format(objid),'r'))
    if CHECK: print "Joining tiles "
    tile_area = [t.area for t in tiles]
    top_5percent_largest = np.where(tile_area>np.percentile(tile_area,98))[0]
    jtiles,ptiles = join_tiles(top_5percent_largest,tiles)
    if CHECK: 
        visualizeTilesSeparate(tiles)
        plt.title("Obj{}: Check Centroid inside central tile ".format(objid))
        plt.plot(jtiles.centroid.x,jtiles.centroid.y,'o')
        
    if CHECK: print 'Find the tile index of the tile that contains the centroid'
    central_tidx =  -1
    for i,t in enumerate(tiles):
        if t.contains(jtiles.centroid):
            central_tidx =  i
            break
    if CHECK: 
        plt.figure()
        plt.title("Shape of Central Tile")
        plot_coords(tiles[central_tidx],reverse_xy=True)
        
    tiles_adjacency ={}
    tiles_adjacency[central_tidx]=0
    leftovers=range(len(tiles))
    prev_tiles = [central_tidx]
    d=1
    while (len(leftovers)!=0):
        # prev_tiles = tiles that are at d-1 distance away 
        # Find all tiles that are d distance away 
        next_tiles=[]
        for utidx in prev_tiles:
            ut = tiles[utidx]
            #ut = join_tiles(prev_tiles,tiles)[0]
            #ut = cascaded_union([tiles[tidx] for tidx in prev_tiles])
            if CHECK and d>2: 
                try:
                    visualizeTilesSeparate(ut)
                    plt.title("Shape of Joined Tiles at d-1")
                except(ValueError):
                    pass
            for tidx in leftovers:
                t = tiles[tidx]
                if adjacent(t,ut):
                    #print utidx,tidx
                    tiles_adjacency[tidx]=d
                    next_tiles.append(tidx)
        prev_tiles=next_tiles
        #print "Leftovers before:",len(leftovers)
        leftovers = [tidx for tidx in leftovers if tidx not in  tiles_adjacency.keys()]
        if (len(leftovers)<int(len(tiles)*0.01) and d>100) or (next_tiles==[]):
            #to avoid missed tile causing infinite loop, we stop above 100 iterations if the number of tiles left is smaller than 1% of all tiles
            for tidx in leftovers:
                tiles_adjacency[tidx]=-1
            break
        d+=1
    if CHECK: 
        visualizeTilesScore(tiles,tiles_adjacency)
        plt.title("Tile Adjacency Score Heatmap ")
    if SAVE: pkl.dump(tiles_adjacency,open("adjacency{}.pkl".format(objid),'w'))
    
    return tiles_adjacency
if __name__ == '__main__':
    #DATA_DIR="final_all_tiles"
    DATA_DIR="sampletopworst5"
    os.chdir(DATA_DIR)

    for objid in tqdm(object_lst):
        print "Working on obj:",objid
        tiles_adjacency = compute_adjacency_score(objid,CHECK=False,SAVE=True)