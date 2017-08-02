from tqdm import tqdm
from sample_worker_seeds import sample_specs
import numpy as np 
import pickle as pkl
from PIL import Image, ImageDraw
from time import time 
def tarea_mask(sample, objid):
    # DEMOTED : this goes from uniqified tiles to pixel
    tiles = pkl.load(open("uniqueTiles/{}/vtiles{}.pkl".format(sample,objid)))
    MV = pkl.load(open("pixel_em/{}/obj{}/MV_mask.pkl".format(sample,objid)))
    mega_mask= np.zeros_like(MV)
    t_area = np.array([t.area for t in tiles])
    # sorted by largest to smallest area
    #so that small area assignments for a pixel overrides the large area assignments
    ordered_tiles = np.array(tiles)[np.argsort(t_area)[::-1]]
    tarea_lst = []
    # mask_lst = []
    overlap_count=0
    for tidx,tile in enumerate(ordered_tiles):
    #     print tidx,len(tarea_lst)
        x,y = tile.exterior.xy
        tarea = tile.area
        img = Image.new('L', ( np.shape(MV)[1], np.shape(MV)[0]), 0)
        ImageDraw.Draw(img).polygon(zip(x,y), outline=1, fill=1)
        mask = np.array(img)*(tidx+1) #tarea
        replace_overlap = np.where((mega_mask!=0)&(mask!=0))
        if len(replace_overlap[0])>0:
            overlap_count+=1
            mega_mask[replace_overlap]=0
        mega_mask+=mask
        tarea_lst.append(tarea)
    mega_mask[mega_mask==0]=0 
    # # NOTE: approximate outer area, in pixel units, not in the same units as the rest of polygonal tarea!!!
    approx_outer_area  = mega_mask.size-sum(tarea_lst)
    tarea_lst.insert(0,approx_outer_area)
    tidx_mask = mega_mask.astype('int')
    pkl.dump(tarea_lst,open("pixel_em/{}/obj{}/tarea.pkl".format(sample,objid),'w'))
    pkl.dump(tidx_mask,open("pixel_em/{}/obj{}/tidx_mask.pkl".format(sample,objid),'w'))

def create_tarea_mask(sample,objid):
    mega_mask = pkl.load(open('pixel_em/{}/obj{}/mega_mask.pkl'.format(sample,objid)))
    # Create masks for single valued tiles (so that they are more disconnected)
    from skimage import measure
    from matplotlib import _cntr as cntr
    tiles = [] # list of masks of all the tiles extracted
    tarea_mask = np.zeros_like(mega_mask)
    unique_tile_values = np.unique(mega_mask)
    tarea_lst = []
    # print unique_tile_values
    for tile_value in unique_tile_values[1:]: #exclude 0
        blobs = mega_mask==tile_value
        blobs_labels = measure.label(blobs,background=0)
        for i in np.unique(blobs_labels)[1:]: 
            tile_mask = blobs_labels==i
            tile_pix = np.where(tile_mask==True)
            tiles.append(tile_mask)
	    tarea = mask_area(tile_mask)
            tarea_mask[tile_pix] = tarea
	    tarea_lst.append(tarea)
    outside_area  = np.product(np.shape(tarea_mask))-np.unique(tarea_mask).sum()
    tarea_mask[np.where(tarea_mask==0)]=outside_area
    tarea_lst.append(outside_area)
    pkl.dump(tarea_lst,open("pixel_em/{}/obj{}/tarea.pkl".format(sample,objid),'w'))
    pkl.dump(tarea_mask,open("pixel_em/{}/obj{}/tarea_mask.pkl".format(sample,objid),'w'))
    return tarea_mask
def plot_tarea_mask(tarea_mask):
    from matplotlib.colors import LogNorm
    plt.imshow(tarea_mask, norm=LogNorm(vmin=1, vmax=np.max(tarea_mask)))
    plt.colorbar()
def mask_area(mask):
    return len(np.where(mask)[0])
##############################################################################################
def neighbor_widx(source):
    x=source[0]
    y=source[1]
    return (x+1,y),(x,y+1),(x-1,y),(x,y-1)

def index_item(pix_lst,item):
    for i,pix in enumerate(pix_lst):
        if str(list(pix))==str(list(item)):
             return i 
def create_PixTiles(sample,objid):
    wmap  = pkl.load(open("pixel_em/{}/obj{}/voted_workers_mask.pkl".format(sample,objid)))
    tiles= []
    # Large outside tile 
    x,y = np.where(wmap==0)
    tiles.append(zip(x,y))
    # Pixels voted by at least one worker
    x,y = np.where(wmap!=0)
    potential_pixs = np.array(zip(x,y))
    #Compute the votes on each pixel 
    votes = np.array([len(wmap[tuple(pix)]) for pix in potential_pixs])
    # sort from lowest to highest number of votes  (low votes likely to be lone/small tiles)
    srt_idx = np.argsort(votes)
    srt_potential_pix = potential_pixs[srt_idx]

    tidx=1
    while(len(srt_potential_pix)!=0):
        print "Tile ", tidx
        checked_pixs = []
        #print "srt_potential_pix length:", len(srt_potential_pix)
        source = tuple(srt_potential_pix[0])
        checked_pixs.append(source)
        srt_potential_pix=np.delete(srt_potential_pix,index_item(srt_potential_pix,source),axis=0)
        tiles.append([source])
        voted_workers = wmap[source]
        pidx =1
        #while (checked_pixs!=tiles[tidx]):
        potential_sources = [source]

        while (len(potential_sources)!=0):
    #         print "Pix ", pidx
    #         print "source:",source
    #         print "tiles[tidx]:",tiles[tidx]
    #         print "checked_pixs:",checked_pixs
    #         print "potential_sources:",potential_sources
            at_least_one_connection=False
            for neighbor in neighbor_widx(source):
                if wmap[neighbor] == voted_workers:
                    tiles[tidx].append(neighbor)
                    # Remove added neighbor from potential pixels 
                    found_idx = index_item(srt_potential_pix,neighbor)
                    if found_idx==None:
                        #print "Neighbor already belong to another tile"
                        pass
                    else: 
                        srt_potential_pix=np.delete(srt_potential_pix,found_idx,axis=0)
                        at_least_one_connection=True
                    #checked_pixs.append(neighbor)
    #        if at_least_one_connection==False: 
    #             potential_sources = [i for i in tiles[tidx] if i not in checked_pixs]

            # Identifying potential sources:
            potential_sources=[]
            for i in tiles[tidx]:
                if i not in checked_pixs:
                    potential_sources.append(i)
    #         print "potential_source after:",potential_sources            
            if len(potential_sources)==0:
    #             print "new tile"
                break
            source = potential_sources[0]
            checked_pixs.append(source)
            pidx+=1
        print "final tiles[tidx]:", tiles[tidx]
        print "len(srt_potential_pix):",len(srt_potential_pix)
        tidx+=1 #moving onto the next tile
    pkl.dump(tiles,open("pixel_em/{}/obj{}/tiles.pkl".format(sample,objid),'w'))
 
if __name__=="__main__":
    start = time()
    create_PixTiles('5workers_rand0',1)
    end = time()
    print "Elapsed Time: ", end-start
    #object_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47]
    ##for sample in tqdm(sample_specs.keys()):
    #for sample in tqdm(['5workers_rand0','10workers_rand0','15workers_rand0','20workers_rand0','25workers_rand0','30workers_rand0']):
    #    print sample 
    #    for objid in object_lst:
    #	    print "objid:",objid
    #	    try:
    #            #tarea_mask(sample,objid)
    #	        create_tarea_mask(sample,objid)
    #	    except(IOError):
    #	        pass
